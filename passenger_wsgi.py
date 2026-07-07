import sys
import os
import traceback
import json
import time
import datetime
import asyncio
import cgi
from io import BytesIO

# Set environment variables to restrict OpenBLAS/MKL threads before importing anything else
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(APP_DIR, "wsgi_error.log")

sys.path.insert(0, APP_DIR)

with open(LOG_FILE, "a") as f:
    f.write(f"\n--- STARTUP at {datetime.datetime.now()} ---\n")

# Import services
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(APP_DIR, ".env"))

    from services.duplicate import get_hash, check_duplicates
    from services.plate import extract_plate
    from services.specs import extract_specs
    from services.condition import assess_condition

    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

    with open(LOG_FILE, "a") as f:
        f.write("All services imported successfully.\n")

except Exception as e:
    with open(LOG_FILE, "a") as f:
        f.write(f"IMPORT ERROR:\n{traceback.format_exc()}\n")
    SUPABASE_URL = None
    SUPABASE_KEY = None


# ── CORS helpers ───────────────────────────────────────────────
CORS_HEADERS = [
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ("Access-Control-Allow-Headers", "*"),
]

def json_response(start_response, data, status="200 OK"):
    body = json.dumps(data).encode("utf-8")
    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
    ] + CORS_HEADERS
    start_response(status, headers)
    return [body]

def error_response(start_response, detail, status="500 Internal Server Error"):
    return json_response(start_response, {"detail": detail}, status)


# ── Route: GET /health ─────────────────────────────────────────
def handle_health(environ, start_response):
    return json_response(start_response, {"status": "ok"})


# ── Route: POST /analyse ──────────────────────────────────────
def handle_analyse(environ, start_response):
    start = time.time()

    # Parse multipart form data
    content_type = environ.get("CONTENT_TYPE", "")
    if "multipart/form-data" not in content_type:
        return error_response(start_response, "Content-Type must be multipart/form-data", "400 Bad Request")

    try:
        fs = cgi.FieldStorage(
            fp=environ["wsgi.input"],
            environ=environ,
            keep_blank_values=True,
        )
    except Exception as e:
        return error_response(start_response, f"Failed to parse form data: {e}", "400 Bad Request")

    # Extract images
    image_items = fs["images"] if "images" in fs else []
    if not isinstance(image_items, list):
        image_items = [image_items]

    image_bytes_list = []
    for item in image_items:
        if item.file:
            image_bytes_list.append(item.file.read())

    if not image_bytes_list:
        return error_response(start_response, "No images provided", "400 Bad Request")

    # Extract listing_id
    listing_id = None
    if "listing_id" in fs:
        listing_id = fs["listing_id"].value

    primary_image = image_bytes_list[0]

    # 1. Hashing
    new_hashes = [get_hash(b) for b in image_bytes_list]

    # 2. Fetch existing hashes from Supabase
    existing = []
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{SUPABASE_URL}/rest/v1/products",
                    params={"select": "id,image_hashes"},
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                    },
                )
                if resp.status_code == 200:
                    existing = resp.json()
        except Exception as e:
            with open(LOG_FILE, "a") as f:
                f.write(f"Supabase fetch error: {e}\n")

    if listing_id:
        existing = [l for l in existing if str(l.get("id")) != str(listing_id)]

    # 3. Run analysis
    duplicate_result = check_duplicates(new_hashes, existing)

    # Plate detection across all images
    all_detections = []
    primary_plate = None
    for idx, b in enumerate(image_bytes_list):
        res = extract_plate(b)
        if res.get("bounding_box") or res.get("full_plate"):
            det = {
                "full_plate": res.get("full_plate"),
                "public_prefix": res.get("public_prefix"),
                "hidden_suffix": res.get("hidden_suffix"),
                "bounding_box": res.get("bounding_box"),
                "image_index": idx,
            }
            all_detections.append(det)
            if not primary_plate and det["full_plate"]:
                primary_plate = det

    plate_result = {
        "full_plate": primary_plate["full_plate"] if primary_plate else None,
        "public_prefix": primary_plate["public_prefix"] if primary_plate else None,
        "hidden_suffix": primary_plate["hidden_suffix"] if primary_plate else None,
        "confidence": 0.95 if primary_plate else 0.0,
        "detections": all_detections,
    }

    specs_result = extract_specs(image_bytes_list)
    condition_result = assess_condition(primary_image)

    elapsed = int((time.time() - start) * 1000)

    response_data = {
        "duplicate": duplicate_result,
        "plate": plate_result,
        "specs": specs_result,
        "condition": condition_result,
        "processing_time_ms": elapsed,
    }

    # Convert any Pydantic models or non-serializable objects to dicts
    def to_serializable(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        return obj

    response_data = to_serializable(response_data)
    return json_response(start_response, response_data)


# ── Main WSGI dispatcher ──────────────────────────────────────
def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    with open(LOG_FILE, "a") as f:
        f.write(f"REQUEST: {method} {path} at {datetime.datetime.now()}\n")

    # Handle CORS preflight
    if method == "OPTIONS":
        start_response("204 No Content", CORS_HEADERS)
        return [b""]

    try:
        if path == "/health" and method == "GET":
            return handle_health(environ, start_response)

        if path == "/analyse" and method == "POST":
            return handle_analyse(environ, start_response)

        return error_response(start_response, "Not Found", "404 Not Found")

    except Exception as e:
        with open(LOG_FILE, "a") as f:
            f.write(f"UNHANDLED ERROR:\n{traceback.format_exc()}\n")
        return error_response(start_response, "Internal Server Error")
