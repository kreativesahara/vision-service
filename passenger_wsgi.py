import sys
import os

# Set environment variables to restrict OpenBLAS/MKL threads before importing anything else
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from a2wsgi import ASGIMiddleware
import traceback

# Add the application directory to the system path
sys.path.insert(0, os.path.dirname(__file__))

# Import the FastAPI app from main.py
from main import app as fastapi_app

# Wrap the ASGI app into a WSGI application so cPanel Passenger can run it
asgi_application = ASGIMiddleware(fastapi_app)

def application(environ, start_response):
    try:
        return asgi_application(environ, start_response)
    except Exception as e:
        with open(os.path.join(os.path.dirname(__file__), "wsgi_error.log"), "a") as f:
            f.write(traceback.format_exc() + "\n")
        start_response('500 Internal Server Error', [('Content-Type', 'text/plain')])
        return [b"Internal Server Error. Please check wsgi_error.log\n"]
