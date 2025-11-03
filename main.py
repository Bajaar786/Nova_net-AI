# backend/main.py
"""
Invoice Insights monorepo backend (FastAPI)
Endpoints:
- POST /upload_csv       -> upload CSV to GCS, returns gs:// uri
- POST /extract_invoice  -> extract structured invoice JSON (via Generative AI), write CSV to GCS
- POST /suggest_query    -> generate SQL suggestion for csv (Generative AI), returns suggested_sql + preview
- POST /confirm_execute  -> execute SQL using BigQuery external table, write result CSV to GCS and return signed URL
- GET  /healthz
Static files: served from frontend/dist (mounted below)
"""
import os
import time
import io
import json
import tempfile
import uuid
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
from google.cloud import storage, bigquery
from fastapi.middleware.cors import CORSMiddleware  # Add this line
import google.auth
import google.auth.transport.requests
import requests
import pandas as pd

# Initialize
GCS_BUCKET = os.environ.get("GCS_BUCKET_NAME", "")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GENAI_MODEL = os.environ.get("GENAI_MODEL", "gemini-1.5-flash")
DEMO_API_KEY = os.environ.get("DEMO_API_KEY", "demo-secret")

if not GCS_BUCKET or not PROJECT_ID:
    # We allow startup without envvars for local linting but will error on requests
    pass

storage_client = storage.Client()
bq_client = bigquery.Client()

app = FastAPI(title="Invoice Insights Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add this to handle OPTIONS preflight requests
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return JSONResponse(status_code=200)

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Serve frontend static files from ../frontend/dist (the Dockerfile copies there)



# --------- Models ----------
class SuggestQueryRequest(BaseModel):
    natural_language_query: str
    gcs_csv_path: str
    max_rows: int = 100

class ExecRequest(BaseModel):
    sql: str
    gcs_csv_path: str
    preferred_output: Optional[str] = "csv"
    filename: Optional[str] = None

class ExtractRequest(BaseModel):
    raw_text: Optional[str] = None
    gcs_input_path: Optional[str] = None
    output_filename: Optional[str] = None

# --------- Helpers ----------
def parse_gs_uri(gs_uri: str):
    # accepts gs://bucket/path or object path (uses GCS_BUCKET)
    if gs_uri.startswith("gs://"):
        _, rest = gs_uri.split("gs://", 1)
        bucket, path = rest.split("/", 1)
        return bucket, path
    if "/" in gs_uri:
        # assume bucket already provided; fallback not expected
        return GCS_BUCKET, gs_uri
    return GCS_BUCKET, gs_uri

def upload_bytes_to_gcs(content: bytes, bucket_name: str, blob_name: str, content_type: str):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content, content_type=content_type)
    return blob

def make_signed_url(blob, expiration_seconds: int = 3600):
    try:
        return blob.generate_signed_url(expiration=expiration_seconds)
    except Exception as e:
        raise RuntimeError(f"Signed URL error: {e}")

# --------- Generative AI (REST with ADC fallback; best-effort)
# --------- Generative AI (Gemini API) ---------
def call_genai(prompt: str, model: Optional[str] = None, temperature: float = 0.0) -> str:
    # Uses Gemini API REST endpoint
    model = model or GENAI_MODEL
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    access_token = credentials.token

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    # Gemini API expects this format
    body = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract text from Gemini response
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0].get("text", "").strip()
        
        raise ValueError(f"Unexpected response format: {data}")
        
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")

# --------- Extract prompt template (JSON-only output)
EXTRACTOR_PROMPT = """
SYSTEM: You are an invoice extraction assistant. Input is raw invoice or OCR text.
YOU MUST OUTPUT JSON ONLY with these keys:
invoice_id, invoice_date (YYYY-MM-DD), vendor, vendor_address, line_items (array of {description, quantity, unit_price, tax, total_price}), subtotal, tax_total, total_amount, currency, notes.
If a field is missing use null or empty string. Numbers must be numeric (no currency symbols). Dates ISO if possible.
INPUT:
\"\"\"{raw_text}\"\"\"
"""

# --------- SQL prompt template (strict)
SQL_PROMPT_TEMPLATE = """
SYSTEM: You are an expert SQL generator for a table named 'df'. OUTPUT ONLY A SINGLE SQL QUERY (no commentary).
SCHEMA:
{schema}
USER_QUESTION:
{user_question}
INSTRUCTIONS:
- Use column names exactly as provided.
- If returning rows, include LIMIT {max_rows}.
- If aggregate, return grouped/aggregated results.
- Output only SQL.
"""

# --------- SQL safety & validators
FORBIDDEN_TOKENS = {"DROP","DELETE","INSERT","UPDATE","CREATE","ALTER","ATTACH","DETACH","PRAGMA","EXEC",";"}
def is_sql_safe(sql: str) -> bool:
    up = sql.upper()
    if ";" in up:
        return False
    for t in FORBIDDEN_TOKENS:
        if t in up:
            return False
    return up.strip().startswith("SELECT") or up.strip().upper().startswith("WITH")

# --------- Endpoints ---------
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file from browser to GCS. Returns gs:// URI."""
    if not GCS_BUCKET:
        raise HTTPException(status_code=500, detail="GCS_BUCKET_NAME not configured")
    
    print(f"Upload received: {file.filename}")  # Add logging
    
    try:
        # Ensure we're at the start of the file
        await file.seek(0)
        
        blob_name = f"uploads/{uuid.uuid4().hex}_{file.filename}"
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        
        # Upload the file content directly
        contents = await file.read()
        blob.upload_from_string(contents, content_type=file.content_type or "text/csv")
        
        gs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
        return {"gcs_uri": gs_uri, "gcs_path": blob_name}
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Add error logging
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_invoice")
async def extract_invoice(req: ExtractRequest):
    """Extract structured invoice JSON and write flattened CSV to GCS; returns gs:// path."""
    if not (req.raw_text or req.gcs_input_path):
        raise HTTPException(status_code=400, detail="Provide raw_text or gcs_input_path")

    # get raw text
    if req.raw_text:
        raw_text = req.raw_text
    else:
        # read from GCS
        try:
            bucket, path = parse_gs_uri(req.gcs_input_path)
            blob = storage_client.bucket(bucket).blob(path)
            if not blob.exists():
                raise HTTPException(status_code=404, detail="Input blob not found")
            raw_text = blob.download_as_text()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read input: {e}")

    prompt = EXTRACTOR_PROMPT.format(raw_text=raw_text)
    try:
        gen = call_genai(prompt, temperature=0.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI extraction error: {e}")

    # parse first JSON object
    try:
        txt = gen.strip()
        first = txt.find("{")
        last = txt.rfind("}")
        if first == -1 or last == -1:
            raise ValueError("No JSON found in AI output")
        json_text = txt[first:last+1]
        parsed = json.loads(json_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {e}. Raw: {gen[:400]}")

    invoice_id = parsed.get("invoice_id") or f"inv_{int(time.time())}"
    line_items = parsed.get("line_items") or []
    rows = []
    if line_items:
        for li in line_items:
            rows.append({
                "invoice_id": invoice_id,
                "invoice_date": parsed.get("invoice_date"),
                "vendor": parsed.get("vendor"),
                "vendor_address": parsed.get("vendor_address"),
                "description": li.get("description"),
                "quantity": li.get("quantity"),
                "unit_price": li.get("unit_price"),
                "tax": li.get("tax"),
                "total_price": li.get("total_price"),
                "subtotal": parsed.get("subtotal"),
                "tax_total": parsed.get("tax_total"),
                "total_amount": parsed.get("total_amount"),
                "currency": parsed.get("currency"),
                "notes": parsed.get("notes")
            })
    else:
        rows.append({
            "invoice_id": invoice_id,
            "invoice_date": parsed.get("invoice_date"),
            "vendor": parsed.get("vendor"),
            "vendor_address": parsed.get("vendor_address"),
            "description": "",
            "quantity": None,
            "unit_price": None,
            "tax": parsed.get("tax_total") or 0,
            "total_price": parsed.get("total_amount"),
            "subtotal": parsed.get("subtotal"),
            "tax_total": parsed.get("tax_total"),
            "total_amount": parsed.get("total_amount"),
            "currency": parsed.get("currency"),
            "notes": parsed.get("notes")
        })

    df = pd.DataFrame(rows)
    filename_base = req.output_filename or f"extracted_{invoice_id}_{int(time.time())}"
    blob_name = f"extracted/{filename_base}.csv"
    try:
        blob = upload_bytes_to_gcs(df.to_csv(index=False).encode("utf-8"), GCS_BUCKET, blob_name, "text/csv")
        signed = make_signed_url(blob, expiration_seconds=3600)
        return {"gs_uri": f"gs://{GCS_BUCKET}/{blob_name}", "signed_url": signed, "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload extracted CSV: {e}")

@app.post("/suggest_query")
async def suggest_query(req: SuggestQueryRequest):
    # Build schema by reading small sample of CSV (autodetect cols)
    try:
        bucket, obj = parse_gs_uri(req.gcs_csv_path)
        tmpf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        storage_client.bucket(bucket).blob(obj).download_to_filename(tmpf.name)
        df = pd.read_csv(tmpf.name, nrows=5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV schema: {e}")

    schema_text = ", ".join(df.columns.tolist())
    prompt = SQL_PROMPT_TEMPLATE.format(schema=schema_text, user_question=req.natural_language_query, max_rows=req.max_rows)
    try:
        sql = call_genai(prompt, temperature=0.0).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI SQL generation error: {e}")
    sql = sql.rstrip(";")
    safe = is_sql_safe(sql)
    preview = []
    est = 0
    if safe:
        # try a light preview using pandas (safer than running full BigQuery)
        try:
            # load small portion to pandas and run sqlite via pandasql? simpler: attempt to run via pandas query if simple SELECT *
            # We'll just return empty preview to avoid executing model SQL locally
            preview = []
            est = 0
        except Exception:
            preview = []
    return {"suggested_sql": sql, "safe": safe, "preview": preview, "estimated_rows": est}

@app.post("/confirm_execute")
async def confirm_execute(body: ExecRequest, x_api_key: Optional[str] = Header(None)):
    # API key guard (optional)
    if DEMO_API_KEY:
        if x_api_key != DEMO_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Validate
    sql = (body.sql or "").strip().rstrip(";")
    if not is_sql_safe(sql):
        raise HTTPException(status_code=400, detail="SQL failed safety validation")

    # create BigQuery external table and run query
    try:
        bucket, path = parse_gs_uri(body.gcs_csv_path)
        # create a temp dataset if not exists
        dataset_id = f"{PROJECT_ID}.invoice_temp"
        try:
            bq_client.get_dataset(dataset_id)
        except Exception:
            bq_client.create_dataset(dataset_id, exists_ok=True)

        table_id = f"ext_{uuid.uuid4().hex[:8]}"
        table_ref = f"{PROJECT_ID}.invoice_temp.{table_id}"

        external_config = bigquery.ExternalConfig("CSV")
        external_config.source_uris = [f"gs://{bucket}/{path}"]
        external_config.options.skip_leading_rows = 1
        external_config.autodetect = True

        table = bigquery.Table(table_ref)
        table.external_data_configuration = external_config
        table = bq_client.create_table(table, exists_ok=True)

        # run query (use table name 'df' in SQL by replacing FROM df with proper ref if needed)
        # Simple convention: if model uses 'df' we adapt by replacing `FROM df` with the table_ref
        sql_to_run = sql.replace("FROM df", f"FROM `{table_ref}`").replace("from df", f"from `{table_ref}`")
        job = bq_client.query(sql_to_run)
        result = job.result()

        # stream results to CSV
        filename_base = body.filename or f"result_{int(time.time())}"
        output_blob_name = f"results/{filename_base}.csv"
        # build CSV in memory
        out_buf = io.StringIO()
        # write header
        header = [schema.name for schema in result.schema]
        out_buf.write(",".join(header) + "\n")
        for row in result:
            values = []
            for f in header:
                v = getattr(row, f)
                values.append("" if v is None else str(v))
            out_buf.write(",".join(values) + "\n")
        content_bytes = out_buf.getvalue().encode("utf-8")
        blob = upload_bytes_to_gcs(content_bytes, GCS_BUCKET, output_blob_name, "text/csv")
        signed = make_signed_url(blob, expiration_seconds=3600)
        return {"gcs_uri": f"gs://{GCS_BUCKET}/{output_blob_name}", "download_url": signed, "row_count": job.num_dml_affected_rows or -1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {e}")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to see all registered routes"""
    routes = []
    for route in app.routes:
        route_info = {
            "path": getattr(route, "path", ""),
            "methods": getattr(route, "methods", []),
            "name": getattr(route, "name", "")
        }
        routes.append(route_info)
    return {"routes": routes}

@app.get("/test")
async def test():
    return {"message": "Test endpoint works"}

# Serve frontend static files from ../frontend/dist (the Dockerfile copies there)
static_dir = os.path.join(os.path.dirname(__file__), "frontend/dist")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    print(f"Serving static files from: {static_dir}")
else:
    print(f"Static directory not found: {static_dir}")
    # In dev/missing build, no static mount; frontend can run with npm dev
    pass