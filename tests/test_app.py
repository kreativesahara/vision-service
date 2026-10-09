"""passenger_wsgi.py: the entry point LiteSpeed runs in production."""
import io
import json
import threading

import pytest

import passenger_wsgi as app
from services.duplicate import check_duplicates, get_hash

PLATE_BOX = [{'x': 10, 'y': 10}, {'x': 95, 'y': 10}, {'x': 95, 'y': 30}, {'x': 10, 'y': 30}]


@pytest.fixture(autouse=True)
def log_to_tmp(monkeypatch, tmp_path):
    log_file = tmp_path / 'wsgi_error.log'
    monkeypatch.setattr(app, 'LOG_FILE', str(log_file))
    return log_file


def call(path, method='GET', body=b'', content_type=None):
    environ = {
        'PATH_INFO': path,
        'REQUEST_METHOD': method,
        'CONTENT_LENGTH': str(len(body)),
        'wsgi.input': io.BytesIO(body),
    }
    if content_type:
        environ['CONTENT_TYPE'] = content_type
    response = {}
    chunks = app.application(environ, lambda status, headers: response.update(status=status, headers=dict(headers)))
    raw = b''.join(chunks)
    return response['status'], response['headers'], json.loads(raw) if raw else None


def multipart(images, field='images'):
    """A multipart body the way the add-listing form sends it."""
    boundary = '----kemotivesFormBoundary'
    body = b''.join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="photo{i}.webp"\r\n'
        f'Content-Type: image/webp\r\n\r\n'.encode() + image + b'\r\n'
        for i, image in enumerate(images)
    ) + f'--{boundary}--\r\n'.encode()
    return body, f'multipart/form-data; boundary={boundary}'


@pytest.fixture
def photos(monkeypatch, make_jpeg):
    """Three photos: the second shows the plate, the third a plate too blurry to read but still boxed."""
    images = [make_jpeg((200, 0, 0)), make_jpeg((0, 200, 0)), make_jpeg((0, 0, 200))]
    readings = {
        images[1]: {'full_plate': 'KBX 737U', 'public_prefix': 'KBX', 'hidden_suffix': '737U', 'bounding_box': PLATE_BOX},
        images[2]: {'full_plate': None, 'bounding_box': PLATE_BOX},
    }
    monkeypatch.setattr(app, 'extract_plate', lambda image: readings.get(image, {}))
    monkeypatch.setattr(app, 'extract_specs', lambda images: {'make': 'Lexus', 'field_confidences': {'make': 0.99}})
    monkeypatch.setattr(app, 'assess_condition', lambda image: {'grade': 'good', 'score': 75})
    return images


# ── /analyse ──────────────────────────────────────────────────

def test_analyse_returns_everything_the_add_listing_form_reads(photos):
    status, headers, data = call('/analyse', 'POST', *multipart(photos))

    assert status == '200 OK'
    assert headers['Access-Control-Allow-Origin'] == '*'
    assert set(data) == {'duplicate', 'plate', 'specs', 'condition', 'processing_time_ms'}
    assert (data['plate']['full_plate'], data['plate']['public_prefix'], data['plate']['hidden_suffix']) == \
        ('KBX 737U', 'KBX', '737U')
    assert [(d['image_index'], d['bounding_box']) for d in data['plate']['detections']] == \
        [(1, PLATE_BOX), (2, PLATE_BOX)]  # the form blurs every boxed plate
    assert data['specs']['make'] == 'Lexus'
    assert data['condition']['grade'] == 'good'
    assert data['duplicate']['is_duplicate'] is False
    assert len(data['duplicate']['hashes']) == 3


def test_a_failing_step_does_not_fail_the_analysis(monkeypatch, photos, log_to_tmp):
    def broken(images):
        raise RuntimeError('Vertex AI unavailable')

    monkeypatch.setattr(app, 'extract_specs', broken)

    status, _, data = call('/analyse', 'POST', *multipart(photos))

    assert status == '200 OK'
    assert data['specs'] is None  # the form treats a missing result as "nothing to autofill"
    assert data['plate']['full_plate'] == 'KBX 737U'
    assert 'Analysis step specs failed' in log_to_tmp.read_text()


def test_analyse_rejects_requests_without_photos():
    body, content_type = multipart([b'not used'], field='listing_id')

    assert call('/analyse', 'POST', body, content_type)[0] == '400 Bad Request'
    assert call('/analyse', 'POST', b'{}', 'application/json')[0] == '400 Bad Request'


def test_health_preflight_and_unknown_routes():
    assert call('/health')[::2] == ('200 OK', {'status': 'ok'})

    status, headers, body = call('/analyse', 'OPTIONS')
    assert status == '204 No Content' and body is None
    assert headers['Access-Control-Allow-Methods'] == 'GET, POST, OPTIONS'

    assert call('/nowhere')[0] == '404 Not Found'


# ── Running steps side by side ───────────────────────────────

def test_steps_run_side_by_side():
    # Four steps that can only finish if all four are running at once: three threads plus the request thread
    together = threading.Barrier(4, timeout=5)

    def step(n):
        together.wait()
        return n

    assert app.run_concurrently({n: (step, (n,)) for n in range(4)}) == {0: 0, 1: 1, 2: 2, 3: 3}


def test_never_starts_more_than_the_thread_budget(monkeypatch):
    # The hosting account allows 100 processes + threads in total; one analysis must stay a small slice of that
    real_start = threading.Thread.start
    started = []

    def counting_start(self):
        started.append(self)
        real_start(self)

    monkeypatch.setattr(threading.Thread, 'start', counting_start)

    results = app.run_concurrently({n: (lambda n: n, (n,)) for n in range(10)})

    assert results == {n: n for n in range(10)}
    assert len(started) == app.ANALYSIS_THREADS <= 3


def test_steps_still_finish_when_the_host_refuses_threads(monkeypatch, log_to_tmp):
    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, 'start', refuse)
    ran_on = []

    def step(n):
        ran_on.append(threading.get_ident())
        return n

    assert app.run_concurrently({n: (step, (n,)) for n in range(3)}) == {0: 0, 1: 1, 2: 2}
    assert set(ran_on) == {threading.get_ident()}
    assert 'Could not start an analysis thread' in log_to_tmp.read_text()


def test_uses_the_threads_it_gets_when_the_host_runs_short(monkeypatch):
    real_start = threading.Thread.start
    granted = []

    def first_only(self):
        if granted:
            raise RuntimeError("can't start new thread")
        granted.append(self)
        real_start(self)

    monkeypatch.setattr(threading.Thread, 'start', first_only)
    # Two steps that can only finish together: the one granted thread plus the request thread
    together = threading.Barrier(2, timeout=5)

    def step(n):
        together.wait()
        return n

    assert app.run_concurrently({n: (step, (n,)) for n in range(2)}) == {0: 0, 1: 1}


def test_a_step_that_raises_comes_back_as_none():
    def broken():
        raise ValueError('boom')

    assert app.run_concurrently({'ok': (lambda: 'fine', ()), 'bad': (broken, ())}) == {'ok': 'fine', 'bad': None}


# ── Supporting pieces ─────────────────────────────────────────

def test_photo_hashes_match_for_the_same_photo(make_jpeg):
    photo = make_jpeg((10, 120, 200), size=(320, 240))
    stored = [{'id': 146, 'image_hashes': [get_hash(photo)]}]

    assert check_duplicates([get_hash(photo)], stored)['duplicate_listing_id'] == '146'
    assert check_duplicates([get_hash(photo)], [])['is_duplicate'] is False


def test_local_dev_app_exposes_the_same_routes():
    import main

    assert {'/analyse', '/health'} <= {route.path for route in main.app.routes}
