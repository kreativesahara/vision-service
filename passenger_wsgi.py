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

# Write a startup marker
with open(LOG_FILE, "a") as f:
    f.write(f"\n--- STARTUP ATTEMPT at {datetime.datetime.now()} ---\n")

# Try to load the FastAPI app
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

    # Pure WSGI test endpoint — no FastAPI, no a2wsgi
    if path == "/_wsgi_test":
        body = json.dumps({"wsgi": "ok", "python": sys.version, "fastapi_loaded": fastapi_loaded}).encode("utf-8")
        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ])
        return [body]

    # If FastAPI failed to load, return the error
    if not fastapi_loaded:
        msg = f"App failed to start: {load_error}\n".encode("utf-8")
        start_response("500 Internal Server Error", [
            ("Content-Type", "text/plain"),
            ("Content-Length", str(len(msg))),
        ])
        return [msg]

    # Forward to FastAPI via a2wsgi
    try:
        result = asgi_application(environ, start_response)
        with open(LOG_FILE, "a") as f:
            f.write(f"REQUEST OK for {path}\n")
        return result
    except Exception as e:
        with open(LOG_FILE, "a") as f:
            f.write(f"REQUEST ERROR for {path}:\n{traceback.format_exc()}\n")
        msg = b"Internal Server Error\n"
        start_response("500 Internal Server Error", [
            ("Content-Type", "text/plain"),
            ("Content-Length", str(len(msg))),
        ])
        return [msg]
