import sys
import os
import traceback
import datetime
import json

# Set environment variables to restrict OpenBLAS/MKL threads before importing anything else
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(APP_DIR, "wsgi_error.log")

with open(LOG_FILE, "a") as f:
    f.write(f"\n--- STARTUP ATTEMPT at {datetime.datetime.now()} ---\n")

fastapi_loaded = False
load_error = None

try:
    sys.path.insert(0, APP_DIR)
    from a2wsgi import ASGIMiddleware
    from main import app as fastapi_app
    asgi_application = ASGIMiddleware(fastapi_app)
    fastapi_loaded = True
    with open(LOG_FILE, "a") as f:
        f.write("FastAPI app loaded successfully.\n")
except Exception as e:
    load_error = str(e)
    with open(LOG_FILE, "a") as f:
        f.write(f"IMPORT ERROR:\n{traceback.format_exc()}\n")


def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")

    with open(LOG_FILE, "a") as f:
        f.write(f"REQUEST: {environ.get('REQUEST_METHOD')} {path} at {datetime.datetime.now()}\n")

    # Pure WSGI test endpoint
    if path == "/_wsgi_test":
        body = json.dumps({"wsgi": "ok", "fastapi_loaded": fastapi_loaded}).encode("utf-8")
        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ])
        return [body]

    if not fastapi_loaded:
        msg = f"App failed to start: {load_error}\n".encode("utf-8")
        start_response("500 Internal Server Error", [
            ("Content-Type", "text/plain"),
            ("Content-Length", str(len(msg))),
        ])
        return [msg]

    # Intercept a2wsgi response to debug what Passenger rejects
    captured_status = [None]
    captured_headers = [None]

    def capture_start_response(status, headers, exc_info=None):
        captured_status[0] = status
        captured_headers[0] = headers
        with open(LOG_FILE, "a") as f:
            f.write(f"  a2wsgi status: {status}\n")
            f.write(f"  a2wsgi headers: {headers}\n")
        return start_response(status, headers, exc_info)

    try:
        result = asgi_application(environ, capture_start_response)
        # Collect body chunks
        body_parts = []
        for chunk in result:
            body_parts.append(chunk)

        with open(LOG_FILE, "a") as f:
            total_len = sum(len(c) for c in body_parts)
            preview = b"".join(body_parts)[:200]
            f.write(f"  a2wsgi body length: {total_len}\n")
            f.write(f"  a2wsgi body preview: {preview}\n")

        # If a2wsgi didn't call start_response, do it ourselves
        if captured_status[0] is None:
            with open(LOG_FILE, "a") as f:
                f.write("  WARNING: a2wsgi never called start_response!\n")
            body = json.dumps({"error": "ASGI bridge failed silently"}).encode("utf-8")
            start_response("502 Bad Gateway", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        return body_parts

    except Exception as e:
        with open(LOG_FILE, "a") as f:
            f.write(f"REQUEST ERROR for {path}:\n{traceback.format_exc()}\n")
        msg = b"Internal Server Error\n"
        start_response("500 Internal Server Error", [
            ("Content-Type", "text/plain"),
            ("Content-Length", str(len(msg))),
        ])
        return [msg]
