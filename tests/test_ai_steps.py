import json
import threading
import time

from services import condition, gemini, specs

LEXUS_REPLY = '```json\n' + json.dumps({
    'make': {'value': 'Lexus', 'confidence': 0.99},
    'model': {'value': 'RX 400h', 'confidence': 0.99},
    'year': {'value': '2007', 'confidence': 0.85},
    'engineCapacity': {'value': '3300cc', 'confidence': 0.9},
    'fuelType': {'value': 'Gasoline', 'confidence': 0.95},
    'transmission': {'value': 'Automatic', 'confidence': 0.5},
    'colour': {'value': 'Red', 'confidence': 0.3},
}) + '\n```'


# ── Shared Gemini client ──────────────────────────────────────

def test_one_gemini_client_is_shared_by_steps_starting_together(monkeypatch):
    # A client nothing holds on to is garbage-collected, which closes its connection mid-request. An earlier version
    # created one per thread when steps started at the same moment, and the condition step failed every time.
    created = []

    def construct(**kwargs):
        time.sleep(0.05)  # widen the window in which two threads could both create a client
        created.append(kwargs)
        return object()

    monkeypatch.setattr(gemini.genai, 'Client', construct)
    start = threading.Barrier(6)
    clients = []

    def step():
        start.wait()
        clients.append(gemini.client())

    threads = [threading.Thread(target=step) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 1
    assert len({id(c) for c in clients}) == 1
    assert created[0]['vertexai'] is True
    assert created[0]['http_options'].timeout == gemini.TIMEOUT_MS


def test_reading_config_is_deterministic_without_thinking():
    # With default sampling the same photos came back as 2007 one run and 2005 the next, at the same confidence
    assert gemini.READING.temperature == 0
    assert gemini.READING.thinking_config.thinking_budget == 0


def test_image_parts_carry_the_real_photo_format(make_jpeg):
    png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 8
    webp = b'RIFF' + b'\x00' * 4 + b'WEBP' + b'\x00' * 8

    assert gemini.image_part(png).inline_data.mime_type == 'image/png'
    assert gemini.image_part(webp).inline_data.mime_type == 'image/webp'
    assert gemini.image_part(make_jpeg()).inline_data.mime_type == 'image/jpeg'


# ── Specs ─────────────────────────────────────────────────────

def test_specs_keep_confident_valid_fields(fake_gemini, make_jpeg):
    model = fake_gemini(LEXUS_REPLY)

    result = specs.extract_specs([make_jpeg(), make_jpeg((0, 0, 200))])

    assert (result['make'], result['model'], result['year']) == ('Lexus', 'RX 400h', 2007)
    assert result['engineCapacity'] == '3300'
    assert result['fuelType'] is None  # not one of the form's fuel types
    assert result['transmission'] is None  # below the autofill threshold
    assert result['colour'] == 'Red'  # passed through for the form to judge by its confidence
    assert result['field_confidences']['transmission'] == 0.5
    assert result['autopopulate'] is True

    request = model.requests[0]
    assert request['model'] == specs.SPEC_MODEL
    assert request['config'] is gemini.READING
    assert len(request['contents']) == 3  # the prompt and one part per photo


def test_specs_keep_only_colours_the_form_offers(fake_gemini, make_jpeg):
    fake_gemini(json.dumps({'colour': {'value': 'silver', 'confidence': 0.9}}))
    assert specs.extract_specs([make_jpeg()])['colour'] == 'Silver'

    fake_gemini(json.dumps({'colour': {'value': 'Pearl', 'confidence': 0.9}}))
    assert specs.extract_specs([make_jpeg()])['colour'] is None


def test_specs_fall_back_to_an_empty_result_on_a_bad_reply(fake_gemini, make_jpeg):
    fake_gemini('Sorry, I cannot help with that.')

    result = specs.extract_specs([make_jpeg()])

    assert result['make'] is None
    assert result['autopopulate'] is False
    assert result['field_confidences'] == {}


def test_specs_do_not_autofill_when_nothing_is_confident(fake_gemini, make_jpeg):
    fake_gemini(json.dumps({'make': {'value': 'Toyota', 'confidence': 0.4}, 'trim': {'value': 'G', 'confidence': 0.9}}))

    result = specs.extract_specs([make_jpeg()])

    assert result['make'] is None and result['trim'] == 'G'
    assert result['autopopulate'] is False


# ── Condition ─────────────────────────────────────────────────

def test_condition_parses_the_assessment(fake_gemini, make_jpeg):
    model = fake_gemini('```json\n{"grade": "good", "score": 75, "damage_flags": ["yellowed headlights"], '
                        '"interior_condition": "good", "notes": "Well kept."}\n```')

    result = condition.assess_condition([make_jpeg(), make_jpeg((0, 0, 200)), make_jpeg((90, 90, 90))])

    assert (result['grade'], result['score'], result['damage_flags']) == ('good', 75, ['yellowed headlights'])
    request = model.requests[0]
    assert request['config'] is gemini.READING
    assert len(request['contents']) == 4  # the prompt and every photo, so the inside of the car counts


def test_condition_reports_unknown_when_gemini_fails(fake_gemini, make_jpeg):
    fake_gemini(TimeoutError('deadline exceeded'))

    result = condition.assess_condition([make_jpeg()])

    assert (result['grade'], result['score']) == ('unknown', 0)
