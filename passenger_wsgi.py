import sys
import os
import traceback
import datetime

# Set environment variables to restrict OpenBLAS/MKL threads before importing anything else
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(APP_DIR, "wsgi_error.log")

# Write a startup marker so we know this file is being executed
with open(LOG_FILE, "a") as f:
    f.write(f"\n--- STARTUP ATTEMPT at {datetime.datetime.now()} ---\n")

try:
    sys.path.insert(0, APP_DIR)

    with open(LOG_FILE, "a") as f:
        f.write("Importing a2wsgi...\n")
    from a2wsgi import ASGIMiddleware

    with open(LOG_FILE, "a") as f:
        f.write("Importing main app...\n")
    from main import app as fastapi_app

    with open(LOG_FILE, "a") as f:
        f.write("Creating WSGI wrapper...\n")
    asgi_application = ASGIMiddleware(fastapi_app)

    with open(LOG_FILE, "a") as f:
        f.write("Startup complete!\n")

    def application(environ, start_response):
        try:
            return asgi_application(environ, start_response)
        except Exception as e:
            with open(LOG_FILE, "a") as f:
                f.write(f"REQUEST ERROR: {traceback.format_exc()}\n")
            start_response('500 Internal Server Error', [('Content-Type', 'text/plain')])
            return [b"Internal Server Error\n"]

except Exception as e:
    # Catch any import-level crash and log it
    with open(LOG_FILE, "a") as f:
        f.write(f"IMPORT ERROR:\n{traceback.format_exc()}\n")

    # Provide a fallback application that returns the error message
    def application(environ, start_response):
        start_response('500 Internal Server Error', [('Content-Type', 'text/plain')])
        msg = f"App failed to start: {e}\n"
        return [msg.encode('utf-8')]
