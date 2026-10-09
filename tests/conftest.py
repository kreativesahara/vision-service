"""Shared setup: tests run offline, under the production thread caps, and can never reach Google."""
import io
import os
import sys

# The caps passenger_wsgi.py sets in production; they must be in place before numpy and OpenCV load
for _var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

import pytest
from PIL import Image

from fakes import FakeGemini, FakeVision
from services import gemini, plate


@pytest.fixture(autouse=True)
def no_google(monkeypatch):
    """A test that forgets to stub a client fails loudly, instead of calling (and billing) the real Google APIs."""
    def refuse(*args, **kwargs):
        raise RuntimeError('Tests must not call Google: stub the client')

    monkeypatch.setattr(plate.vision, 'ImageAnnotatorClient', refuse)
    monkeypatch.setattr(gemini.genai, 'Client', refuse)
    monkeypatch.setattr(plate, '_vision_client', None)
    monkeypatch.setattr(gemini, '_client', None)


@pytest.fixture
def make_jpeg():
    """Real image bytes, so OpenCV and Pillow decode them as they do uploaded photos."""
    def make(colour=(200, 30, 30), size=(120, 60)):
        buffer = io.BytesIO()
        Image.new('RGB', size, colour).save(buffer, 'JPEG')
        return buffer.getvalue()
    return make


@pytest.fixture
def fake_gemini(monkeypatch):
    def install(reply):
        fake = FakeGemini(reply)
        monkeypatch.setattr(gemini, 'client', lambda: fake)
        return fake
    return install


@pytest.fixture
def fake_vision(monkeypatch):
    def install(reply):
        fake = FakeVision(reply)
        monkeypatch.setattr(plate, '_vision_client', fake)
        return fake
    return install
