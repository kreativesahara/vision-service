"""Regression tests for the 2026-10-09 production fix (vision_service_production_fix.md in the monorepo).

Google Vision and Gemini are replaced with fakes, so these run in CI without credentials."""
import io
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

import passenger_wsgi  # noqa: E402  (loads the services the way LiteSpeed does)
from services import gemini, plate  # noqa: E402


def _slow(value):
    time.sleep(0.5)
    return value


def test_services_load():
    assert passenger_wsgi.services_loaded, passenger_wsgi.load_error_msg


def test_steps_run_side_by_side():
    start = time.time()
    results = passenger_wsgi.run_concurrently({i: (_slow, (i,)) for i in range(4)})

    assert results == {i: i for i in range(4)}
    assert time.time() - start < 1.5  # 3 threads + the request thread: one round, not four


def test_steps_run_on_the_request_thread_when_the_host_refuses_threads(monkeypatch):
    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, 'start', refuse)

    assert passenger_wsgi.run_concurrently({i: (_slow, (i,)) for i in range(3)}) == {0: 0, 1: 1, 2: 2}


def test_steps_finish_when_the_host_grants_only_some_threads(monkeypatch):
    original_start = threading.Thread.start
    granted = []

    def one_then_refuse(self):
        if granted:
            raise RuntimeError("can't start new thread")
        granted.append(self)
        original_start(self)

    monkeypatch.setattr(threading.Thread, 'start', one_then_refuse)

    assert passenger_wsgi.run_concurrently({i: (_slow, (i,)) for i in range(4)}) == {i: i for i in range(4)}


def test_a_failing_step_comes_back_empty_and_the_rest_finish():
    def boom():
        raise ValueError('boom')

    assert passenger_wsgi.run_concurrently({'ok': (_slow, ('fine',)), 'bad': (boom, ())}) == {'ok': 'fine', 'bad': None}


def _token(text, x):
    corners = [(x, 10), (x + 60, 10), (x + 60, 40), (x, 40)]
    vertices = [SimpleNamespace(x=cx, y=cy) for cx, cy in corners]
    return SimpleNamespace(description=text, bounding_poly=SimpleNamespace(vertices=vertices))


def test_a_read_plate_survives_an_ascii_only_log_stream(monkeypatch):
    # The server's stdout is ASCII; an em dash in plate.py's log line used to throw the read plate away
    response = SimpleNamespace(
        error=SimpleNamespace(message=''),
        text_annotations=[SimpleNamespace(description='KAA 123A'), _token('KAA', 10), _token('123A', 80)],
    )
    vision_client = SimpleNamespace(text_detection=lambda image, timeout=None: response)
    monkeypatch.setattr(plate, '_get_client', lambda: vision_client)
    monkeypatch.setattr(plate, '_verify_plate_text_with_gemini', lambda image_bytes, box: None)
    monkeypatch.setattr(sys, 'stdout', io.TextIOWrapper(io.BytesIO(), encoding='ascii'))

    result = plate.extract_plate(b'image bytes')

    assert result['full_plate'] == 'KAA 123A'
    assert result['bounding_box'] is not None


def test_the_app_switches_its_log_streams_to_utf8_under_an_ascii_locale():
    # lswsgi runs Python under the C locale, and the specs and condition steps print Gemini's replies
    probe = 'import passenger_wsgi, sys; print("Gemini says \\u2014 fine"); print(sys.stdout.encoding)'
    child = subprocess.run(
        [sys.executable, '-c', probe],
        cwd=APP_DIR,
        env={**os.environ, 'PYTHONIOENCODING': 'ascii'},
        capture_output=True,
        timeout=180,
    )

    assert child.returncode == 0, child.stderr.decode(errors='replace')
    assert child.stdout.decode('utf-8').splitlines()[-1] == 'utf-8'


def test_steps_starting_together_share_one_gemini_client(monkeypatch):
    # google-genai closes a client when it's garbage-collected, so a second client that nothing keeps hold of
    # fails mid-request with "Cannot send a request, as the client has been closed"
    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            time.sleep(0.05)  # widen the window in which two threads could each create one
            created.append(self)

    monkeypatch.setattr(gemini.genai, 'Client', FakeClient)
    monkeypatch.setattr(gemini, '_client', None)

    clients = passenger_wsgi.run_concurrently({i: (gemini.client, ()) for i in range(4)})

    assert len(created) == 1
    assert all(client is created[0] for client in clients.values())
