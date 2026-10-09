import sys
import os
import threading
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

# lswsgi embeds Python under the C locale, so stdout and stderr (LiteSpeed writes them to stderr.log) are ASCII and
# printing anything else raises. An em dash in a log line used to throw away detected plates, and Gemini's replies
# (curly quotes, dashes) can break the specs and condition steps the same way.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass


def _log(message):
    with open(LOG_FILE, "a", encoding="utf-8", errors="backslashreplace") as f:
        f.write(message + "\n")


_log(f"\n--- STARTUP at {datetime.datetime.now()} ---")

# Import services
services_loaded = False
load_error_msg = None

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(APP_DIR, ".env"))

    _log("Loading services...")

    # OpenCV ignores the thread variables above and starts one thread per CPU core (96 on the server) on its
    # first parallel call. The hosting account allows 100 processes + threads in total, so that starves PHP (503s).
    import cv2
    cv2.setNumThreads(1)

    from services.duplicate import get_hash, check_duplicates
    from services.plate import extract_plate
    from services.specs import extract_specs
    from services.condition import assess_condition

    services_loaded = True
    _log("All services loaded. App ready.")

except Exception as e:
    load_error_msg = traceback.format_exc()
    _log(f"IMPORT ERROR:\n{load_error_msg}")


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


# ── Parallel analysis steps ────────────────────────────────────
# Steps run side by side, so the seller waits for the slowest step rather than the sum of all of them. Few threads,
# because the hosting account caps processes + threads at 100.
ANALYSIS_THREADS = 3


def run_concurrently(tasks):
    """Run {key: (fn, args)} side by side; returns {key: result, or None if the step raised}. A step the host won't
    give a thread to runs on the request's own thread instead, so a refusal only makes the analysis slower."""
    pending = list(tasks.items())
    lock = threading.Lock()
    results = {}

    def work():
        while True:
            with lock:
                if not pending:
                    return
                key, (fn, args) = pending.pop(0)
            try:
                results[key] = fn(*args)
            except Exception:
                _log(f"Analysis step {key} failed:\n{traceback.format_exc()}")
                results[key] = None

    threads = []
    for _ in range(min(ANALYSIS_THREADS, len(pending))):
        thread = threading.Thread(target=work, daemon=True)
        try:
            thread.start()
        except RuntimeError:
            _log("Could not start an analysis thread; the request thread runs the remaining steps")
            break
        threads.append(thread)

    work()
    for thread in threads:
        thread.join()
    return results


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
        _log(f"Multipart parse error: {traceback.format_exc()}")
        return error_response(start_response, f"Failed to parse form data: {e}", "400 Bad Request")

    image_bytes_list = files.get("images", [])
    if not image_bytes_list:
        return error_response(start_response, "No images provided", "400 Bad Request")

    primary_image = image_bytes_list[0]

    # 1. Hashing
    new_hashes = [get_hash(b) for b in image_bytes_list]

    # 2. Duplicate check. Listings live in Laravel's `cars` table, which stores no image hashes, so there is nothing
    #    to compare against here; the old lookup read a `products` table that doesn't exist and never matched.
    duplicate_result = check_duplicates(new_hashes, [])

    # 3. Run analysis, slowest steps first so they start straight away. Plates are read on every image so each
    #    visible plate can be blurred.
    tasks = {
        "condition": (assess_condition, (primary_image,)),
        "specs": (extract_specs, (image_bytes_list,)),
    }
    tasks.update({("plate", idx): (extract_plate, (b,)) for idx, b in enumerate(image_bytes_list)})
    results = run_concurrently(tasks)

    # Plate detection
    all_detections = []
    primary_plate = None
    for idx in range(len(image_bytes_list)):
        res = results.get(("plate", idx)) or {}
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

    specs_result = results.get("specs")
    condition_result = results.get("condition")

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

    _log(f"REQUEST: {method} {path} at {datetime.datetime.now()}")

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
        _log(f"UNHANDLED ERROR:\n{traceback.format_exc()}")
        return error_response(start_response, "Internal Server Error")
