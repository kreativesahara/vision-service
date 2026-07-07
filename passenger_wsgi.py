import sys
import os
from a2wsgi import ASGIMiddleware

# Add the application directory to the system path
sys.path.insert(0, os.path.dirname(__file__))

# Import the FastAPI app from main.py
from main import app as fastapi_app

# Wrap the ASGI app into a WSGI application so cPanel Passenger can run it
application = ASGIMiddleware(fastapi_app)
