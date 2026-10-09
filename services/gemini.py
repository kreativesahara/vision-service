import os
import threading

from google import genai
from google.genai import types

# Longest a single Gemini call may take, so a stuck call can't keep the seller's form waiting (milliseconds)
TIMEOUT_MS = 30_000

# For reading tasks (badges, a plate, visible damage). Thinking is off: in testing it only added latency. Temperature
# is 0: with the default, the same photos came back as 2007 one run and 2005 the next, at the same confidence.
READING = types.GenerateContentConfig(temperature=0, thinking_config=types.ThinkingConfig(thinking_budget=0))


_client = None
_client_lock = threading.Lock()


def client() -> genai.Client:
    """One Vertex AI client per process, so its auth token is reused across calls instead of fetched for each.

    Created under a lock and kept here for good: analysis steps start at the same moment on separate threads, and a
    second client nothing holds on to gets garbage-collected, which closes its connection mid-request."""
    global _client
    with _client_lock:
        if _client is None:
            _client = genai.Client(
                vertexai=True,
                project=os.getenv('GCP_PROJECT_ID', 'kemotives'),
                location='us-central1',
                http_options=types.HttpOptions(timeout=TIMEOUT_MS),
            )
        return _client


def image_part(image_bytes: bytes) -> types.Part:
    """The form uploads WebP, which used to be labelled JPEG and left to Gemini to sniff."""
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        mime_type = 'image/png'
    elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        mime_type = 'image/webp'
    else:
        mime_type = 'image/jpeg'
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
