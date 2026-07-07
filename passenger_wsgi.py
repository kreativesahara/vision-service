import sys
import os
import traceback
import json
import time
import datetime
from io import BytesIO
from urllib.parse import parse_qs

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
services_loaded = False
load_error_msg = None

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(APP_DIR, ".env"))

    with open(LOG_FILE, "a") as f:
        f.write("Loading services...\n")

    from services.duplicate import get_hash, check_duplicates
    from services.plate import extract_plate
    from services.specs import extract_specs
    from services.condition import assess_condition

    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

    services_loaded = True
    with open(LOG_FILE, "a") as f:
        f.write("All services loaded. App ready.\n")

except Exception as e:
    load_error_msg = traceback.format_exc()
    with open(LOG_FILE, "a") as f:
        f.write(f"IMPORT ERROR:\n{load_error_msg}\n")
    SUPABASE_URL = None
    SUPABASE_KEY = None


# ── CORS headers ───────────────────────────────────────────────
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


# ── Multipart parser (no cgi module needed) ───────────────────
def parse_multipart(environ):
    """Parse multipart/form-data from WSGI environ without the cgi module."""
    content_type = environ.get("CONTENT_TYPE", "")
    if "boundary=" not in content_type:
        return {}, {}

    boundary = content_type.split("boundary=")[1].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    content_length = int(environ.get("CONTENT_LENGTH", 0))
    body = environ["wsgi.input"].read(content_length)

    boundary_bytes = ("--" + boundary).encode("utf-8")
    end_boundary = (boundary_bytes + b"--")

    parts = body.split(boundary_bytes)
    files = {}
    fields = {}

    for part in parts:
        if not part or part.strip() == b"" or part.strip() == b"--":
            continue

        # Remove leading \r\n
        if part.startswith(b"\r\n"):
            part = part[2:]
        # Remove trailing \r\n--
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if part.endswith(b"--"):
            part = part[:-2]
        if part.endswith(b"\r\n"):
            part = part[:-2]

        # Split headers from body
        if b"\r\n\r\n" not in part:
            continue

        header_block, file_data = part.split(b"\r\n\r\n", 1)
        headers_text = header_block.decode("utf-8", errors="replace")

        # Parse Content-Disposition
        name = None
        filename = None
        for line in headers_text.split("\r\n"):
            if "Content-Disposition:" in line:
                if 'name="' in line:
                    name = line.split('name="')[1].split('"')[0]
                if 'filename="' in line:
                    filename = line.split('filename="')[1].split('"')[0]

        if not name:
            continue

        if filename:
            # File field
            if name not in files:
                files[name] = []
            files[name].append(file_data)
        else:
            # Regular field
            fields[name] = file_data.decode("utf-8", errors="replace")

    return files, fields


# ── Route: GET /health ─────────────────────────────────────────
def handle_health(environ, start_response):
    return json_response(start_response, {"status": "ok"})


# ── Route: POST /analyse ──────────────────────────────────────
def handle_analyse(environ, start_response):
    if not services_loaded:
        return error_response(start_response, f"Services failed to load: {load_error_msg}")

    start_time = time.time()

    content_type = environ.get("CONTENT_TYPE", "")
    if "multipart/form-data" not in content_type:
        return error_response(start_response, "Content-Type must be multipart/form-data", "400 Bad Request")

    try:
        files, fields = parse_multipart(environ)
    except Exception as e:
        with open(LOG_FILE, "a") as f:
            f.write(f"Multipart parse error: {traceback.format_exc()}\n")
        return error_response(start_response, f"Failed to parse form data: {e}", "400 Bad Request")

    image_bytes_list = files.get("images", [])
    if not image_bytes_list:
        return error_response(start_response, "No images provided", "400 Bad Request")

    listing_id = fields.get("listing_id")
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

    # Plate detection
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

    elapsed = int((time.time() - start_time) * 1000)

    # Convert Pydantic models to dicts
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

    response_data = to_serializable({
        "duplicate": duplicate_result,
        "plate": plate_result,
        "specs": specs_result,
        "condition": condition_result,
        "processing_time_ms": elapsed,
    })

    return json_response(start_response, response_data)


# ── Main WSGI dispatcher ──────────────────────────────────────
def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    with open(LOG_FILE, "a") as f:
        f.write(f"REQUEST: {method} {path} at {datetime.datetime.now()}\n")

    # CORS preflight
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
