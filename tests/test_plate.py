import io
import json
import threading
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from fakes import word, vision_reply
from services import gemini, plate

SERVICES_DIR = Path(__file__).resolve().parent.parent / 'services'


def kbx_737u():
    return vision_reply(word('KBX', 10, 10), word('737U', 55, 10))


def kind(request):
    """What a Gemini request was: a plate search of the photo, a search of a close-up of it, or a read of a crop."""
    if request['config'] is gemini.PLATE_READING:
        return 'read'
    return 'close-up' if request['contents'][0] is plate.CLOSE_UP_PROMPT else 'search'


def kinds(model):
    return [kind(request) for request in model.requests]


def gemini_finding(*plates, crop_reading='', close_up=()):
    """Gemini's replies: each plate search finds `plates` ({'box_2d', 'text'}), each close-up search finds `close_up`
    (none by default, so the boxes the searches found stand), and its check of a cropped plate reads crop_reading."""
    def reply(request):
        found = {'search': plates, 'close-up': close_up}.get(kind(request))
        return crop_reading if found is None else json.dumps({'plates': list(found)})
    return reply


def gemini_searching(*searches, crop_reading='', close_up=()):
    """Gemini's replies when its searches disagree: the nth plate search finds the plates in searches[n], or fails
    with it if it's an exception. Close-up searches and crop checks as gemini_finding."""
    found = iter(searches)
    others = gemini_finding(crop_reading=crop_reading, close_up=close_up)

    def reply(request):
        if kind(request) != 'search':
            return others(request)
        answer = next(found)
        return answer if isinstance(answer, Exception) else json.dumps({'plates': list(answer)})
    return reply


# A plate Gemini finds on a 200x100 photo, and the box it becomes: 0-1000 scaled to the photo, then widened by 20%
# of the box's size on every side
FOUND_BOX_2D = [400, 250, 600, 750]
FOUND_BOX = [{'x': 30, 'y': 36}, {'x': 170, 'y': 36}, {'x': 170, 'y': 64}, {'x': 30, 'y': 64}]

# Plates elsewhere on that photo, far enough from each other and from FOUND_BOX not to be blurred as one (plate.NEAR):
# a small one low down (another car behind, or the car's own seen from afar), a big one along the top, and empty
# ground above the plate (where a search put a seller's plate)
LOW_SMALL_BOX_2D = [800, 400, 850, 500]
LOW_SMALL_BOX = [{'x': 76, 'y': 79}, {'x': 104, 'y': 79}, {'x': 104, 'y': 86}, {'x': 76, 'y': 86}]
OTHER_CAR_BOX_2D = [850, 50, 900, 120]
TOP_BIG_BOX_2D = [50, 100, 200, 900]
ABOVE_BOX_2D = [0, 0, 100, 200]


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
    assert model.requests[0]['config'] is gemini.PLATE_READING


def test_keeps_the_vision_reading_when_the_gemini_check_fails(fake_vision, fake_gemini, make_jpeg):
    fake_vision(kbx_737u())
    fake_gemini(TimeoutError('deadline exceeded'))

    assert plate.extract_plate(make_jpeg())['full_plate'] == 'KBX 737U'


def test_photo_without_a_plate_tries_the_contrast_fallback_when_gemini_cannot_search(fake_vision, fake_gemini, make_jpeg):
    vision = fake_vision(vision_reply(word('LEXUS', 10, 10), word('RX400h', 60, 10)))
    fake_gemini(TimeoutError('deadline exceeded'))

    result = plate.extract_plate(make_jpeg())

    assert result['full_plate'] is None and result['bounding_box'] is None and result['bounding_boxes'] == []
    assert vision.timeouts == [plate.VISION_TIMEOUT, plate.VISION_TIMEOUT]


def test_photo_without_a_plate_takes_one_vision_read_and_one_gemini_search(fake_vision, fake_gemini, make_jpeg):
    vision = fake_vision(vision_reply(word('LEXUS', 10, 10)))
    model = fake_gemini(gemini_finding())

    result = plate.extract_plate(make_jpeg())

    assert result['full_plate'] is None and result['bounding_boxes'] == []
    assert vision.timeouts == [plate.VISION_TIMEOUT]  # Gemini saw no plate either: no contrast pass
    assert [request['config'] for request in model.requests] == [gemini.LOCATING]


def test_finds_a_plate_vision_cannot_read_at_an_angle(fake_vision, fake_gemini, make_jpeg, capsys):
    # Vision reads no text at all on a plate seen at an angle; Gemini finds it, then reads its crop as for Vision's
    fake_vision(vision_reply())
    model = fake_gemini(gemini_finding({'box_2d': FOUND_BOX_2D, 'text': 'kcg-26op'}, crop_reading='KCG 260P'))

    result = plate.extract_plate(make_jpeg(size=(200, 100)))

    assert result['bounding_box'] == FOUND_BOX and result['bounding_boxes'] == [FOUND_BOX]
    assert (result['full_plate'], result['public_prefix'], result['hidden_suffix']) == ('KCG 260P', 'KCG', '260P')
    # Two searches (one box can miss), a close-up of the place they found, then the check of the reading
    assert kinds(model) == ['search', 'search', 'close-up', 'read']
    out = capsys.readouterr().out
    assert "Gemini plate search: 1 place(s), 1 found by both searches, 1 not confirmed close up" in out
    assert "Gemini found 1 plate(s) Vision couldn't box: 'KCG ****'" in out
    assert '260P' not in out


def test_blurs_every_plate_gemini_finds_even_ones_it_cannot_read(fake_vision, fake_gemini, make_jpeg):
    fake_vision(vision_reply())
    model = fake_gemini(gemini_finding(
        {'box_2d': FOUND_BOX_2D, 'text': None},  # the car for sale, too far away to read
        {'box_2d': OTHER_CAR_BOX_2D, 'text': None},  # another car behind it
    ))

    result = plate.extract_plate(make_jpeg(size=(200, 100)))

    assert result['bounding_boxes'][0] == FOUND_BOX and len(result['bounding_boxes']) == 2
    assert result['full_plate'] is None and result['confidence'] == 0.0
    assert kinds(model) == ['search', 'search', 'close-up', 'close-up']  # nothing read, so no check


def test_ignores_boxes_that_cannot_be_a_plate(fake_vision, fake_gemini, make_jpeg):
    fake_vision(vision_reply())
    fake_gemini(gemini_finding(
        {'box_2d': [0, 0, 1000, 1000], 'text': 'KCG 260P'},  # the whole photo
        {'box_2d': [600, 500, 400, 600]},  # upside down
        {'box_2d': 'somewhere'},
        'a plate',
    ))

    assert plate.extract_plate(make_jpeg())['bounding_boxes'] == []


def test_a_plate_vision_read_but_could_not_box_gets_gemini_box(fake_vision, fake_gemini, make_jpeg):
    # The full text holds the plate, but none of the words Vision boxed belong to it
    fake_vision(SimpleNamespace(
        error=SimpleNamespace(message=''),
        text_annotations=[SimpleNamespace(description='KBX 737U', bounding_poly=None)],
    ))
    fake_gemini(gemini_finding({'box_2d': FOUND_BOX_2D, 'text': None}, crop_reading='KBX 787U'))

    result = plate.extract_plate(make_jpeg(size=(200, 100)))

    assert result['bounding_boxes'] == [FOUND_BOX]
    assert result['full_plate'] == 'KBX 787U'  # checked on Gemini's crop, as a boxed reading would be


def test_a_plate_gemini_finds_in_several_photos_is_read_once(fake_vision, fake_gemini, make_jpeg):
    fake_vision(vision_reply())
    model = fake_gemini(gemini_finding({'box_2d': FOUND_BOX_2D, 'text': 'KCG 260P'}, crop_reading='KCG 260P'))
    shared = plate.SharedPlateChecks()

    results = [plate.extract_plate(make_jpeg(colour, (200, 100)), shared) for colour in [(200, 0, 0), (0, 200, 0)]]

    assert [r['full_plate'] for r in results] == ['KCG 260P'] * 2
    assert all(r['bounding_boxes'] == [FOUND_BOX] for r in results)
    assert kinds(model) == ['search', 'search', 'close-up', 'read', 'search', 'search', 'close-up']


def test_blurs_where_either_search_puts_the_plate_when_they_disagree(fake_vision, fake_gemini, make_jpeg):
    # On a seller's real photo, the same picture saved a little differently put one search's box above the plate
    fake_vision(vision_reply())
    fake_gemini(gemini_searching([{'box_2d': ABOVE_BOX_2D}], [{'box_2d': FOUND_BOX_2D}]))

    boxes = plate.extract_plate(make_jpeg(size=(200, 100)))['bounding_boxes']

    assert len(boxes) == 2 and FOUND_BOX in boxes


def test_reads_the_plate_both_searches_found_not_a_bigger_box_only_one_found(fake_vision, fake_gemini, make_jpeg):
    # A box only one search found can be on empty ground (above the plate, on a seller's photo): it is still blurred,
    # but the reading is checked on the plate both searches agree on
    fake_vision(vision_reply())
    model = fake_gemini(gemini_searching(
        [{'box_2d': TOP_BIG_BOX_2D, 'text': 'KCG 260P'}, {'box_2d': LOW_SMALL_BOX_2D, 'text': 'KCG 260P'}],
        [{'box_2d': LOW_SMALL_BOX_2D, 'text': 'KCG 260P'}],
        crop_reading='KCG 260P',
    ))

    result = plate.extract_plate(make_jpeg(size=(200, 100)))

    assert result['bounding_box'] == LOW_SMALL_BOX and len(result['bounding_boxes']) == 2
    read = next(request for request in model.requests if kind(request) == 'read')
    assert Image.open(io.BytesIO(read['contents'][1].inline_data.data)).size == (32, 7)  # with the crop's margin
    assert result['full_plate'] == 'KCG 260P'


def test_two_searches_finding_the_same_plate_give_it_one_box(fake_vision, fake_gemini, make_jpeg):
    fake_vision(vision_reply())
    fake_gemini(gemini_searching([{'box_2d': FOUND_BOX_2D}], [{'box_2d': [420, 270, 620, 770]}]))  # a little lower and right

    boxes = plate.extract_plate(make_jpeg(size=(200, 100)))['bounding_boxes']

    assert boxes == [[{'x': 30, 'y': 36}, {'x': 174, 'y': 36}, {'x': 174, 'y': 66}, {'x': 30, 'y': 66}]]  # both, joined


@pytest.mark.parametrize('size, second_look', [((1200, 900), (800, 600)), ((600, 800), (600, 800))],
                         ids=['shrunk to 800', 'a phone upload, re-saved'])
def test_the_second_search_looks_at_another_copy(fake_vision, fake_gemini, make_jpeg, size, second_look):
    # Phones upload at 800 px; shrinking those further lost the plate on sellers' photos
    fake_vision(vision_reply())
    model = fake_gemini(gemini_finding({'box_2d': FOUND_BOX_2D}))
    photo = make_jpeg(size=size)

    plate.extract_plate(photo)

    looked_at = [request['contents'][1].inline_data.data for request in model.requests if kind(request) == 'search']
    assert [Image.open(io.BytesIO(data)).size for data in looked_at] == [size, second_look]
    assert looked_at[1] != photo  # different pixels, not the same search twice


def test_a_close_up_search_puts_the_box_on_the_plate(fake_vision, fake_gemini, make_jpeg):
    # On a seller's front close-up, searches of the whole photo boxed only the top half of the plate; searched close
    # up, the box landed on it. The close-up spans 1.5 box widths to each side and 3 box heights above and below.
    fake_vision(vision_reply())
    model = fake_gemini(gemini_finding({'box_2d': [500, 450, 540, 550]}, close_up=[{'box_2d': [500, 400, 600, 600]}]))

    boxes = plate.extract_plate(make_jpeg(size=(1000, 1000)))['bounding_boxes']

    close_up = next(request for request in model.requests if kind(request) == 'close-up')
    assert Image.open(io.BytesIO(close_up['contents'][1].inline_data.data)).size == (560, 392)
    # The close-up's box (padded as any), placed back on the photo, instead of the search's box
    assert boxes == [[{'x': 422, 'y': 512}, {'x': 578, 'y': 512}, {'x': 578, 'y': 567}, {'x': 422, 'y': 567}]]


@pytest.mark.parametrize('close_up', [[], TimeoutError('deadline exceeded')], ids=['finds nothing', 'fails'])
def test_a_place_the_close_up_search_cannot_confirm_keeps_its_box(fake_vision, fake_gemini, make_jpeg, close_up):
    # A plate left readable costs more than a blurred patch
    fake_vision(vision_reply())

    def reply(request):
        if kind(request) == 'close-up' and isinstance(close_up, Exception):
            return close_up
        return json.dumps({'plates': close_up if kind(request) == 'close-up' else [{'box_2d': FOUND_BOX_2D}]})

    model = fake_gemini(reply)

    assert plate.extract_plate(make_jpeg(size=(200, 100)))['bounding_boxes'] == [FOUND_BOX]
    assert kinds(model) == ['search', 'search', 'close-up']


def test_boxes_each_covering_part_of_a_plate_are_blurred_as_one(fake_vision, fake_gemini, make_jpeg):
    # Two boxes a little apart, one over the plate's top and one over its bottom (neither confirmed close up), are
    # blurred as one box that covers the gap between them too
    fake_vision(vision_reply())
    fake_gemini(gemini_searching([{'box_2d': [400, 250, 500, 750]}], [{'box_2d': [560, 250, 660, 750]}]))

    boxes = plate.extract_plate(make_jpeg(size=(200, 100)))['bounding_boxes']

    assert boxes == [[{'x': 30, 'y': 38}, {'x': 170, 'y': 38}, {'x': 170, 'y': 68}, {'x': 30, 'y': 68}]]


def test_a_failed_first_search_gets_a_second_before_the_contrast_retry(fake_vision, fake_gemini, make_jpeg):
    # A search that timed out used to leave a seller's angled plate to the contrast-enhanced Vision retry, which read
    # it no better than the first Vision read did, so it wasn't blurred
    vision = fake_vision(vision_reply())
    fake_gemini(gemini_searching(TimeoutError('deadline exceeded'), [{'box_2d': FOUND_BOX_2D}]))

    result = plate.extract_plate(make_jpeg(size=(200, 100)))

    assert result['bounding_boxes'] == [FOUND_BOX]
    assert vision.timeouts == [plate.VISION_TIMEOUT]  # no contrast retry needed


def test_a_photo_where_no_search_finds_a_plate_gets_one_search(fake_vision, fake_gemini, make_jpeg):
    fake_vision(vision_reply())
    model = fake_gemini(gemini_finding())

    assert plate.extract_plate(make_jpeg())['bounding_boxes'] == []
    assert len(model.requests) == 1  # interiors and engine bays cost one search, not two


@pytest.mark.parametrize('crop_reading', ['', 'KCG 268P'], ids=['crop check reads nothing', 'crop check reads otherwise'])
def test_keeps_the_box_but_drops_a_reading_only_gemini_made_that_its_crop_check_does_not_confirm(
        fake_vision, fake_gemini, make_jpeg, crop_reading):
    # Gemini guesses at plates it can't really read, a blurred one included (the admin panel checks photos the form
    # already blurred), and a wrong plate would fill in the form and the duplicate check
    fake_vision(vision_reply())
    fake_gemini(gemini_finding({'box_2d': FOUND_BOX_2D, 'text': 'KCG 260P'}, crop_reading=crop_reading))

    result = plate.extract_plate(make_jpeg(size=(200, 100)))

    assert result['bounding_boxes'] == [FOUND_BOX]
    assert result['full_plate'] is None and result['hidden_suffix'] is None and result['confidence'] == 0.0


def vision_read_without_a_box(reading):
    """A reply whose full text holds the plate, but none of whose boxed words belong to it."""
    return SimpleNamespace(
        error=SimpleNamespace(message=''),
        text_annotations=[SimpleNamespace(description=reading, bounding_poly=None)],
    )


def test_checks_the_plate_vision_read_not_another_car_gemini_found(fake_vision, fake_gemini, make_jpeg):
    fake_vision(vision_read_without_a_box('KBX 737U'))
    model = fake_gemini(gemini_finding(
        {'box_2d': TOP_BIG_BOX_2D, 'text': 'KCA 111A'},  # a bigger plate on another car, found first
        {'box_2d': LOW_SMALL_BOX_2D, 'text': 'KBX 737U'},  # the plate Vision read
        crop_reading='KBX 737U',
    ))

    result = plate.extract_plate(make_jpeg(size=(200, 100)))

    assert result['full_plate'] == 'KBX 737U'
    assert result['bounding_box'] == LOW_SMALL_BOX and len(result['bounding_boxes']) == 2  # its own plate first
    read = next(request for request in model.requests if kind(request) == 'read')
    assert Image.open(io.BytesIO(read['contents'][1].inline_data.data)).size == (32, 7)  # with the crop's 10% margin


def test_without_a_matching_reading_the_biggest_plate_is_the_cars_own(fake_vision, fake_gemini, make_jpeg):
    fake_vision(vision_read_without_a_box('KBX 737U'))
    fake_gemini(gemini_finding(
        {'box_2d': OTHER_CAR_BOX_2D, 'text': None},  # small, far behind
        {'box_2d': FOUND_BOX_2D, 'text': None},
        crop_reading='',
    ))

    result = plate.extract_plate(make_jpeg(size=(200, 100)))

    assert result['bounding_box'] == FOUND_BOX
    assert result['full_plate'] == 'KBX 737U'  # the crop check read nothing: Vision's reading stands


def slow_calls(monkeypatch, vision, search_reply):
    """Every Vision and Gemini call runs to its timeout, on a fake clock PHOTO_BUDGET reads. Returns the clock."""
    clock = [0.0]
    monkeypatch.setattr(plate, '_now', lambda: clock[0])
    vision_answer = vision.text_detection

    def slow_vision(image, timeout=None):
        clock[0] += plate.VISION_TIMEOUT
        return vision_answer(image, timeout)

    def slow_gemini(request):
        clock[0] += gemini.PLATE_TIMEOUT_MS / 1000
        return search_reply(request)

    vision.text_detection = slow_vision
    return clock, slow_gemini


def test_a_photo_stays_within_its_budget_when_every_call_runs_to_its_timeout(monkeypatch, fake_vision, fake_gemini, make_jpeg):
    # The admin panel's photo check waits 45 s for the whole analysis (kemotives-laravel
    # PhotoConditionCheck::CHECK_TIMEOUT), the add-listing form 60 s
    assert plate.PHOTO_BUDGET <= 45
    assert gemini.LOCATING.http_options.timeout == gemini.PLATE_READING.http_options.timeout == gemini.PLATE_TIMEOUT_MS
    clock, reply = slow_calls(monkeypatch, fake_vision(vision_reply()),
                              gemini_finding({'box_2d': FOUND_BOX_2D, 'text': 'KCG 260P'}, crop_reading='KCG 260P'))
    model = fake_gemini(reply)

    result = plate.extract_plate(make_jpeg(size=(200, 100)))

    assert clock[0] <= plate.PHOTO_BUDGET
    assert result['bounding_boxes'] == [FOUND_BOX]  # both searches ran, so the plate is blurred
    assert result['full_plate'] is None  # no time left for a close-up search or a check of the reading
    assert kinds(model) == ['search', 'search']


def test_a_photo_whose_searches_all_time_out_still_ends_within_its_budget(monkeypatch, fake_vision, fake_gemini,
                                                                          make_jpeg):
    vision = fake_vision(vision_reply())

    def failing(request):
        raise TimeoutError('deadline exceeded')

    clock, reply = slow_calls(monkeypatch, vision, failing)
    model = fake_gemini(reply)

    assert plate.extract_plate(make_jpeg())['bounding_boxes'] == []
    assert kinds(model) == ['search', 'search']  # the first failed, so the second ran
    assert vision.timeouts == [plate.VISION_TIMEOUT]  # leaving no time for the contrast-enhanced retry
    assert clock[0] <= plate.PHOTO_BUDGET


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
