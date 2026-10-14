import io
import threading
import sys
from pathlib import Path

from fakes import word, vision_reply
from services import gemini, plate

SERVICES_DIR = Path(__file__).resolve().parent.parent / 'services'


def kbx_737u():
    return vision_reply(word('KBX', 10, 10), word('737U', 55, 10))


def test_reads_a_kenyan_plate_and_boxes_it_for_blurring(fake_vision, fake_gemini, make_jpeg):
    fake_vision(kbx_737u())
    fake_gemini('KBX 737U')

    result = plate.extract_plate(make_jpeg())

    assert result['full_plate'] == 'KBX 737U'
    assert (result['public_prefix'], result['hidden_suffix']) == ('KBX', '737U')
    assert result['bounding_box'] == [{'x': 10, 'y': 10}, {'x': 95, 'y': 10}, {'x': 95, 'y': 30}, {'x': 10, 'y': 30}]
    assert result['category'] == 'Standard National Registration Plate (LLL NNN L)'


def test_plate_survives_an_ascii_only_log_stream(monkeypatch, fake_vision, make_jpeg):
    # lswsgi runs Python under the C locale. An em dash in plate.py's log line made every detected plate come back
    # null in production, because the failed print was caught as a detection failure.
    monkeypatch.setattr(sys, 'stdout', io.TextIOWrapper(io.BytesIO(), encoding='ascii'))
    fake_vision(kbx_737u())

    assert plate.extract_plate(make_jpeg())['full_plate'] == 'KBX 737U'


def test_service_log_lines_are_ascii():
    # Same reason: keep every print in the services ASCII, so a log line can never abort a step
    offenders = [
        f'{path.name}:{number}'
        for path in SERVICES_DIR.glob('*.py')
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1)
        if 'print(' in line and not line.isascii()
    ]
    assert offenders == []


def test_corrects_common_ocr_mistakes_in_the_number(fake_vision, make_jpeg):
    fake_vision(vision_reply(word('KCG', 10, 10), word('26OP', 55, 10)))

    assert plate.extract_plate(make_jpeg())['full_plate'] == 'KCG 260P'


def test_gemini_reading_of_the_cropped_plate_wins(fake_vision, fake_gemini, make_jpeg):
    fake_vision(kbx_737u())
    model = fake_gemini('KBX 787U')

    assert plate.extract_plate(make_jpeg())['full_plate'] == 'KBX 787U'
    assert model.requests[0]['config'] is gemini.READING


def test_keeps_the_vision_reading_when_the_gemini_check_fails(fake_vision, fake_gemini, make_jpeg):
    fake_vision(kbx_737u())
    fake_gemini(TimeoutError('deadline exceeded'))

    assert plate.extract_plate(make_jpeg())['full_plate'] == 'KBX 737U'


def test_photo_without_a_plate_tries_the_contrast_fallback_then_gives_up(fake_vision, make_jpeg):
    vision = fake_vision(vision_reply(word('LEXUS', 10, 10), word('RX400h', 60, 10)))

    result = plate.extract_plate(make_jpeg())

    assert result['full_plate'] is None and result['bounding_box'] is None
    assert vision.timeouts == [plate.VISION_TIMEOUT, plate.VISION_TIMEOUT]


def test_vision_api_errors_are_logged_not_swallowed(fake_vision, make_jpeg, capsys):
    fake_vision(vision_reply(error='PERMISSION_DENIED: billing is disabled'))

    assert plate.extract_plate(make_jpeg())['full_plate'] is None
    assert 'Vision API error: PERMISSION_DENIED' in capsys.readouterr().out


def test_vision_client_uses_rest_not_grpc(monkeypatch):
    # gRPC runs background threads in every process; the hosting account caps processes + threads at 100
    created = []
    monkeypatch.setattr(plate.vision, 'ImageAnnotatorClient', lambda **kwargs: created.append(kwargs) or object())

    first, second = plate._get_client(), plate._get_client()

    assert created == [{'transport': 'rest'}]
    assert first is second


def test_categorises_plate_formats():
    assert plate.categorize_kenyan_plate('KBA 123G') == 'Standard National Registration Plate (LLL NNN L)'
    assert plate.categorize_kenyan_plate('KD 1234') == 'Standard / Historical KD Dealer Plate'
    assert plate.categorize_kenyan_plate('KDK 123A') == 'New Generation / KDK Digital Series Dealer Plate'
    assert plate.categorize_kenyan_plate('') == 'Invalid plate'


def test_logs_never_hold_the_private_end_of_a_plate(fake_vision, fake_gemini, make_jpeg, capsys):
    fake_vision(kbx_737u())
    fake_gemini('KBX 737U')

    plate.extract_plate(make_jpeg())

    out = capsys.readouterr().out
    assert "Plate 'KBX ****' detected" in out
    assert '737U' not in out
    assert plate.mask_plate('KD 1234') == 'KD ****'
    assert plate.mask_plate(None) == '-'


def test_a_plate_in_several_photos_is_checked_by_gemini_once(fake_vision, fake_gemini, make_jpeg):
    fake_vision(kbx_737u())
    model = fake_gemini('KBX 787U')
    shared = plate.SharedPlateChecks()

    results = [plate.extract_plate(make_jpeg(colour), shared) for colour in [(200, 0, 0), (0, 200, 0), (0, 0, 200)]]

    assert len(model.requests) == 1
    assert [r['full_plate'] for r in results] == ['KBX 787U'] * 3  # every photo gets the checked reading
    assert all(r['bounding_box'] for r in results)  # and its own blur box


def test_photos_checked_at_the_same_moment_still_share_one_check():
    shared = plate.SharedPlateChecks()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_check():
        calls.append(1)
        started.set()
        release.wait(5)
        return ('KBX 737U', 'KBX', '737U')

    results = []
    first = threading.Thread(target=lambda: results.append(shared.verify('KBX 737U', slow_check)))
    first.start()
    started.wait(5)
    second = threading.Thread(target=lambda: results.append(shared.verify('KBX 737U', slow_check)))
    second.start()
    release.set()
    first.join(5)
    second.join(5)

    assert len(calls) == 1
    assert results == [('KBX 737U', 'KBX', '737U')] * 2


def test_a_failed_shared_check_lets_the_next_photo_try_its_own():
    shared = plate.SharedPlateChecks()
    replies = iter([None, ('KBX 737U', 'KBX', '737U')])

    assert shared.verify('KBX 737U', lambda: next(replies)) is None
    assert shared.verify('KBX 737U', lambda: next(replies)) == ('KBX 737U', 'KBX', '737U')
