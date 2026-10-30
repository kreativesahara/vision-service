import json
import re
import threading
import time
import numpy as np
from google.cloud import vision
from services import gemini

_vision_client = None

# Longest a single Google Vision call may take (seconds)
VISION_TIMEOUT = 15

# Longest a photo waits for another photo's Gemini check of the same plate before checking its own (seconds).
# Longer than the Gemini client's own timeout (services/gemini.py), so a stuck check fails first.
SHARED_CHECK_WAIT = 45

# The longest one photo's plate step may take (seconds): the admin panel's photo check waits 45 s for the whole
# analysis (kemotives-laravel PhotoConditionCheck::CHECK_TIMEOUT), the add-listing form 60 s. Calls that only improve
# the answer (a second plate search, a close-up search, a check of a reading, the contrast-enhanced retry) are skipped
# when they could run past it; the first plate search never is.
PHOTO_BUDGET = 45

# The longest a Gemini plate search can take (seconds)
SEARCH_TIME = gemini.PLATE_TIMEOUT_MS / 1000

# The longest a check of a reading can take: it may wait for another photo's check of the same plate, then run its own
CROP_CHECK_TIME = 2 * SEARCH_TIME

# Long side, in pixels, of the copy a second plate search looks at (_second_look): bigger photos are shrunk to it
SECOND_LOOK_SIZE = 800

# How far around a box its close-up search looks: this many box widths to each side, and twice as many box heights above
# and below (a search put one seller's plate box 1.5 plate heights too high)
CLOSE_UP_REACH = 1.5

# Two boxes are blurred as one when they would overlap once each is widened by this share of its own size on every
# side, so a plate two boxes each partly cover is covered whole, gap included
NEAR = 0.5

_now = time.monotonic  # the clock PHOTO_BUDGET runs on; tests replace it

# A Kenyan plate in a reading with everything but A-Z and 0-9 stripped: KXX NNNX, or a KD dealer plate. The number
# allows the letters OCR mistakes for digits, which OCR_DIGITS turns back.
PLATE_PATTERN = re.compile(r'(K[A-Z]{2}|KD)([\dOISZ]{3,4}[A-Z]?)')
OCR_DIGITS = str.maketrans('OISZ', '0152')

# Gemini's plate boxes are widened by this share of their size on every side, so the blur covers the plate's border
# and emblem as well as its characters. At 15% the edge of a box on a real two-line rear plate ran along its first
# letter.
GEMINI_BOX_PADDING = 0.2

# The same for a box a close-up search found (_close_up), which sits on the plate: at GEMINI_BOX_PADDING, a seller said
# the blur looked exaggerated (about twice the plate's area, over the badge and bumper)
CLOSE_UP_PADDING = 0.08

# A "plate" taking more of the photo than this is a misreading (a sign, the whole bumper), not a box to blur
MAX_PLATE_AREA = 0.25

# For photos where Google Vision reads no plate: a plate at an angle, far away or partly in the frame often has no
# text Vision can read, but still has to be blurred. ASCII only, like the rest of this file.
LOCATE_PROMPT = """
Find every vehicle number plate in this photo of a car for sale, including plates seen at an angle, far away, in
shadow or only partly in the frame, and those of other vehicles. Return ONLY JSON:
{"plates": [{"box_2d": [ymin, xmin, ymax, xmax], "text": "KBX 737U"}]}
box_2d is the plate's box, with each coordinate on a 0-1000 scale over the photo. text is the plate's characters,
or null if they can't be read. If no number plate shows, return {"plates": []}.
"""

# For a close-up of the photo around a box a plate search found (_close_up). On a seller's front close-up, searches of
# the whole photo put boxes over only the top half of the plate, or on the pole above it; searched close up, every box
# landed on the plate.
CLOSE_UP_PROMPT = """
This is a close-up cut from a photo of a car for sale. Find every vehicle number plate in it, including plates seen at
an angle or only partly in the frame. Return ONLY JSON: {"plates": [{"box_2d": [ymin, xmin, ymax, xmax]}]}, each
coordinate on a 0-1000 scale over this close-up. If no number plate shows, return {"plates": []}.
"""


def mask_plate(plate: str | None) -> str:
    """A plate as logs may show it: the public prefix, and the private part as stars ('KBX 737U' -> 'KBX ****').
    The marketplace never shows the end of a plate, and the server logs are kept, so they don't hold it either."""
    if not plate:
        return '-'
    prefix, _, rest = str(plate).strip().partition(' ')
    return f"{prefix} {'*' * len(rest)}" if rest else '*' * len(prefix)


class SharedPlateChecks:
    """Gemini's reading of each plate checked during one analysis, keyed by what Google Vision read on the photo.

    A car's plate usually shows in several of its photos, and each used to get its own Gemini check of the same
    characters. With this, the first photo checks it and the others reuse the answer: they wait while that check is
    running rather than starting their own. If it fails, or takes longer than SHARED_CHECK_WAIT, a photo checks its
    own crop instead. Each photo still gets its own blur box. One per analysis request; safe across its threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._checks = {}  # Vision reading -> (Event set when the check finished, [result or None])

    def verify(self, reading: str, check):
        with self._lock:
            entry = self._checks.get(reading)
            first = entry is None
            if first:
                entry = (threading.Event(), [None])
                self._checks[reading] = entry
        finished, result = entry

        if first:
            try:
                result[0] = check()
            finally:
                finished.set()
            return result[0]

        if finished.wait(SHARED_CHECK_WAIT) and result[0] is not None:
            return result[0]
        return check()

def _parse_plate(text) -> tuple[str, str, str] | None:
    """('KBS 865P', 'KBS', '865P') from a reading such as 'kbs-865p', or None when it holds no Kenyan plate."""
    clean = re.sub(r'[^A-Z0-9]', '', str(text or '').upper())
    match = PLATE_PATTERN.search(clean)
    if not match:
        return None
    prefix, suffix = match.group(1), match.group(2).translate(OCR_DIGITS)
    return f"{prefix} {suffix}", prefix, suffix


def _get_client():
    """Lazily initialise and cache the Google Vision client. REST rather than the default gRPC, which runs its own
    background threads in every process on a host that caps the account at 100 processes + threads."""
    global _vision_client
    if _vision_client is None:
        # noinspection PyArgumentList
        _vision_client = vision.ImageAnnotatorClient(transport='rest')  # type: ignore[call-arg]
    return _vision_client


def _preprocess(image_bytes: bytes) -> bytes:
    """Grayscale + CLAHE contrast enhancement fallback."""
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, buf = cv2.imencode('.jpg', enhanced)
    return buf.tobytes()


def _verify_plate_text_with_gemini(image_bytes: bytes, box: list) -> tuple[str, str, str] | None:
    """Use Gemini 2.5 Flash to double check the cropped license plate text."""
    import cv2
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None

        xs = [pt['x'] for pt in box]
        ys = [pt['y'] for pt in box]
        x_min, x_max = max(0, min(xs)), min(img.shape[1], max(xs))
        y_min, y_max = max(0, min(ys)), min(img.shape[0], max(ys))
        
        # Add 10% padding
        pad_x = int((x_max - x_min) * 0.1)
        pad_y = int((y_max - y_min) * 0.1)
        x_min, x_max = max(0, x_min - pad_x), min(img.shape[1], x_max + pad_x)
        y_min, y_max = max(0, y_min - pad_y), min(img.shape[0], y_max + pad_y)

        crop = img[y_min:y_max, x_min:x_max]
        _, buf = cv2.imencode('.png', crop)
        crop_bytes = buf.tobytes()

        prompt = "Read the Kenyan license plate in this image. Return ONLY the text, e.g. 'KBS 865P'. Return nothing else."
        response = gemini.client().models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, gemini.image_part(crop_bytes)],
            config=gemini.PLATE_READING,
        )
        
        return _parse_plate(response.text)

    except Exception as e:
        print(f"Gemini verification failed: {e}")
        
    return None


def _detect_plate(client, content_bytes: bytes, shared_checks: SharedPlateChecks | None = None,
                  fits=lambda seconds: True) -> dict | None:
    """
    Run text detection and extract a Kenyan plate in KXX NNNX format.
    Returns a result dict on success, None if no plate was found.
    `fits` says whether a call that can take so many seconds still fits the photo's PHOTO_BUDGET.
    """
    image = vision.Image(content=content_bytes)
    response = client.text_detection(image=image, timeout=VISION_TIMEOUT)

    if response.error.message:
        print(f"Vision API error: {response.error.message}")
        return None
    if not response.text_annotations:
        return None

    texts = response.text_annotations
    # Strip everything except A-Z and 0-9 to handle dashes, spaces, dots, etc.
    clean = re.sub(r'[^A-Z0-9]', '', texts[0].description.upper())
    
    # Allow common OCR mistakes for numbers (O->0, I->1, S->5, Z->2)
    match = PLATE_PATTERN.search(clean)

    if not match:
        return None

    prefix = match.group(1)
    raw_suffix = match.group(2)
    # Correct the OCR mistakes in the final output
    suffix = raw_suffix.translate(OCR_DIGITS)

    # ── Bounding box: 3-tier strategy ────────────────────────────────────────
    # Tier 1: tokens that directly contain prefix or suffix text
    plate_tokens = {prefix, raw_suffix}
    plate_chars = set(prefix + raw_suffix)

    def _collect_box(token_filter) -> list | None:
        verts = [
            v
            for t in texts[1:]
            for v in t.bounding_poly.vertices
            if token_filter(re.sub(r'[^A-Z0-9]', '', t.description.upper()))
        ]
        if not verts:
            return None
        xs, ys = [v.x for v in verts], [v.y for v in verts]
        return [
            {"x": min(xs), "y": min(ys)},
            {"x": max(xs), "y": min(ys)},
            {"x": max(xs), "y": max(ys)},
            {"x": min(xs), "y": max(ys)},
        ]

    # Tier 1 — strict: token overlaps with prefix or suffix
    box = _collect_box(
        lambda cd: cd and (
            cd in prefix or cd in raw_suffix or
            prefix in cd or raw_suffix in cd
        )
    )

    # Tier 2 — relaxed: token shares ≥2 characters with any part of the full plate
    if not box:
        full_alphanum = prefix + raw_suffix
        box = _collect_box(
            lambda cd: cd and len(cd) >= 2 and any(
                cd[i:i+2] in full_alphanum for i in range(len(cd) - 1)
            )
        )

    # Tier 3 — broadest: any token where ≥50 % of its chars appear in the plate
    if not box:
        box = _collect_box(
            lambda cd: cd and len(cd) >= 2 and (
                sum(1 for c in cd if c in plate_chars) / len(cd)
            ) >= 0.5
        )

    # ASCII only: the server logs through an ASCII stream, and an em dash here used to throw away the detected plate.
    # Masked: the logs are kept, and the end of a plate is private.
    print(f"Plate '{mask_plate(f'{prefix} {suffix}')}' detected - box tier result: {box is not None}")

    # Double check the text using Gemini 2.5 Flash on the cropped region, once per plate per analysis, if there's time
    if box and fits(CROP_CHECK_TIME):
        check = lambda: _verify_plate_text_with_gemini(content_bytes, box)  # noqa: E731
        gemini_result = shared_checks.verify(f"{prefix} {suffix}", check) if shared_checks else check()
        if gemini_result:
            full, g_prefix, g_suffix = gemini_result
            return {
                'full_plate': full,
                'public_prefix': g_prefix,
                'hidden_suffix': g_suffix,
                'confidence': 0.95,
                'bounding_box': box,
                'category': categorize_kenyan_plate(full)
            }

    full_plate = f"{prefix} {suffix}"
    return {
        'full_plate': full_plate,
        'public_prefix': prefix,
        'hidden_suffix': suffix,
        'confidence': 0.95,
        'bounding_box': box,
        'category': categorize_kenyan_plate(full_plate)
    }


def _pixel_box(box_2d, width: int, height: int, padding: float = GEMINI_BOX_PADDING) -> list | None:
    """Gemini's [ymin, xmin, ymax, xmax] on its 0-1000 grid as the four corners in pixels, the way Vision's boxes
    come, widened by `padding` and kept inside the photo. None for a box that can't be a plate."""
    try:
        y_min, x_min, y_max, x_max = (min(max(float(v) / 1000, 0.0), 1.0) for v in box_2d)
    except (TypeError, ValueError):
        return None
    if x_max <= x_min or y_max <= y_min or (x_max - x_min) * (y_max - y_min) > MAX_PLATE_AREA:
        return None
    pad_x, pad_y = (x_max - x_min) * padding, (y_max - y_min) * padding
    left, right = max(0, round((x_min - pad_x) * width)), min(width, round((x_max + pad_x) * width))
    top, bottom = max(0, round((y_min - pad_y) * height)), min(height, round((y_max + pad_y) * height))
    return [{'x': left, 'y': top}, {'x': right, 'y': top}, {'x': right, 'y': bottom}, {'x': left, 'y': bottom}]


def _second_look(img) -> bytes:
    """A copy of a photo for a second plate search, so Gemini sees different pixels: shrunk to SECOND_LOOK_SIZE on its
    long side, or re-saved at its own size when no bigger than that. Phones upload at 800, and shrinking those further
    lost the plate: on two sellers' photos at 600x800, a 3/4 copy fully covered the plate on 0 of 8 searches, a re-saved
    one on 7 (the eighth timed out)."""
    import cv2
    long_side = max(img.shape[:2])
    if long_side > SECOND_LOOK_SIZE:
        scale = SECOND_LOOK_SIZE / long_side
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def _locate_plates_with_gemini(image_bytes: bytes, fits=lambda seconds: True) -> list[dict] | None:
    """Every number plate Gemini finds in a photo: [{'bounding_box', 'reading', 'agreed'}], reading being
    _parse_plate's answer or None, and agreed whether both searches found the plate. [] when no search finds one; None
    when the searches failed.

    One search's box can miss its plate: on a seller's real photo, the same picture saved a little differently put the
    box on the pole above the plate, or over only its top half. So a second search looks at another copy (_second_look;
    when the first finds plates, or fails), and each place either finds is searched again close up, where the plate is
    big enough to box exactly. A place the close-up search can't confirm keeps its box: a missed plate costs more than a
    blurred patch. Calls that won't fit the photo's PHOTO_BUDGET are skipped, keeping the boxes found so far."""
    import cv2
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    height, width = img.shape[:2]
    first = _search_plates(image_bytes, width, height)
    # A first search finding nothing ends it: interiors and engine bays cost one search, not two
    second = _search_plates(_second_look(img), width, height) if first != [] and fits(SEARCH_TIME) else None
    if first is None and second is None:
        return None
    # Each plate says which searches found it, so a box both found can be told from one only one did
    found = _join_plates([{**plate, 'searches': {n}}
                          for n, plates in ((1, first), (2, second)) for plate in plates or []])

    plates, unconfirmed = [], 0
    for place in found:
        close = _close_up(img, place['bounding_box']) if fits(SEARCH_TIME) else None
        unconfirmed += not close
        plates += [{**place, 'bounding_box': box} for box in close] if close else [place]
    plates = _join_plates(plates, near=True)
    if found:
        # A rate to watch: places one search missed, or the close-up couldn't confirm, are where blurs land off plates
        print(f"Gemini plate search: {len(found)} place(s), {sum(len(p['searches']) > 1 for p in found)} found by both "
              f"searches{'' if second is not None else ' (one search ran)'}, {unconfirmed} not confirmed close up")
    return [{'bounding_box': p['bounding_box'], 'reading': p['reading'], 'agreed': len(p['searches']) > 1}
            for p in plates]


def _close_up(img, box: list) -> list | None:
    """The boxes of the plates Gemini finds in a close-up of the photo around `box` (CLOSE_UP_REACH), on the whole
    photo. None when that search fails or finds no plate, so `box` stands."""
    import cv2
    height, width = img.shape[:2]
    left, top, right, bottom = _rect(box)
    reach_x, reach_y = CLOSE_UP_REACH * (right - left), 2 * CLOSE_UP_REACH * (bottom - top)
    x0, x1 = max(0, int(left - reach_x)), min(width, int(right + reach_x))
    y0, y1 = max(0, int(top - reach_y)), min(height, int(bottom + reach_y))
    if x1 <= x0 or y1 <= y0:
        return None
    try:
        close_up = cv2.imencode('.jpg', img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    except cv2.error as e:
        print(f"Couldn't cut a close-up for a plate search: {e}")
        return None
    plates = _search_plates(close_up, x1 - x0, y1 - y0, CLOSE_UP_PROMPT, CLOSE_UP_PADDING)
    if not plates:
        return None
    return [[{'x': corner['x'] + x0, 'y': corner['y'] + y0} for corner in plate['bounding_box']] for plate in plates]


def _search_plates(image_bytes: bytes, width: int, height: int, prompt: str = LOCATE_PROMPT,
                   padding: float = GEMINI_BOX_PADDING) -> list[dict] | None:
    """One Gemini search for every number plate in a copy of a photo `width` x `height`: [{'bounding_box', 'reading'}]
    in that photo's pixels (Gemini's boxes are relative, so a smaller copy's map onto it), widened by `padding`,
    reading being _parse_plate's answer or None. [] when it found none; None when the search itself failed."""
    try:
        response = gemini.client().models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, gemini.image_part(image_bytes)],
            config=gemini.LOCATING,
        )
        found = json.loads(response.text)['plates']
        if not isinstance(found, list):
            raise ValueError('no list of plates in its answer')
    except Exception as e:
        print(f"Gemini plate search failed: {e}")
        return None

    plates = []
    for item in found:
        box = _pixel_box(item.get('box_2d'), width, height, padding) if isinstance(item, dict) else None
        if box:
            plates.append({'bounding_box': box, 'reading': _parse_plate(item.get('text'))})
    return plates


def _box_area(box: list) -> int:
    return (box[1]['x'] - box[0]['x']) * (box[2]['y'] - box[1]['y'])


def _rect(box: list) -> list:
    """A box's [left, top, right, bottom]."""
    return [box[0]['x'], box[0]['y'], box[2]['x'], box[2]['y']]


def _touch(a: list, b: list, near: bool = False) -> bool:
    """Whether two [left, top, right, bottom] rects overlap, or with `near`, would once each is widened by NEAR of its
    own size on every side."""
    if near:
        a, b = ([left - NEAR * (right - left), top - NEAR * (bottom - top),
                 right + NEAR * (right - left), bottom + NEAR * (bottom - top)] for left, top, right, bottom in (a, b))
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _join_plates(plates: list, near: bool = False) -> list:
    """Plates whose boxes overlap (or, with `near`, come within NEAR of each other) joined into one: the box around
    both, the first reading either had, and the searches that found either. The rest kept as they are, so where the
    searches disagree on a plate's place, both places are blurred. In the order they were first found."""
    plates = list(plates)
    joined = True
    while joined:
        joined = False
        for i, a in enumerate(plates):
            # By position: both searches often find exactly the same box, and removing by value took the first copy
            j = next((j for j in range(i + 1, len(plates))
                      if _touch(_rect(a['bounding_box']), _rect(plates[j]['bounding_box']), near)), None)
            if j is not None:
                b = plates.pop(j)
                (l1, t1, r1, b1), (l2, t2, r2, b2) = _rect(a['bounding_box']), _rect(b['bounding_box'])
                left, top, right, bottom = min(l1, l2), min(t1, t2), max(r1, r2), max(b1, b2)
                plates[i] = {
                    'bounding_box': [{'x': left, 'y': top}, {'x': right, 'y': top},
                                     {'x': right, 'y': bottom}, {'x': left, 'y': bottom}],
                    'reading': a['reading'] or b['reading'],
                    'searches': a['searches'] | b['searches'],
                }
                joined = True
                break
    return plates


def _located_result(image_bytes: bytes, located: list, shared_checks: SharedPlateChecks | None, read: dict | None,
                    fits=lambda seconds: True) -> dict:
    """extract_plate's answer for the plates Gemini found: all their boxes, the car's own plate first, and that plate's
    characters, checked on a crop of it as Vision's readings are, if there's time. The car's own plate is the one Gemini
    read as Vision did (`read`: Vision read it but couldn't box it), else the biggest of those both searches found,
    else the biggest; another car's plate never fills in the form, and a box only one search found, which can be on
    empty ground, doesn't win on size alone.

    Vision's reading stands when the crop check fails, as for a plate it boxed. A reading only Gemini made counts only
    when the crop check reads the same: Gemini guesses at plates it can't really read, blurred ones included, and a
    wrong plate would fill in the form, the duplicate check and the admin panel's "readable plate" flag. A plate with
    no reading is still blurred."""
    main = next((plate for plate in located if read and plate['reading'] and plate['reading'][0] == read['full_plate']), None) \
        or max(located, key=lambda plate: (plate['agreed'], _box_area(plate['bounding_box'])))
    boxes = [main['bounding_box'], *(plate['bounding_box'] for plate in located if plate is not main)]

    seen = read['full_plate'] if read else (main['reading'][0] if main['reading'] else None)
    checked = None
    if seen and fits(CROP_CHECK_TIME):
        check = lambda: _verify_plate_text_with_gemini(image_bytes, main['bounding_box'])  # noqa: E731
        checked = shared_checks.verify(seen, check) if shared_checks else check()
    if read:
        parsed = checked or (read['full_plate'], read['public_prefix'], read['hidden_suffix'])
    else:
        parsed = checked if checked and checked[0] == seen else None
    full, prefix, suffix = parsed or (None, None, None)
    print(f"Gemini found {len(boxes)} plate(s) Vision couldn't box: '{mask_plate(full)}'")
    return {
        'full_plate': full,
        'public_prefix': prefix,
        'hidden_suffix': suffix,
        'confidence': 0.95 if full else 0.0,
        'bounding_box': boxes[0],
        'bounding_boxes': boxes,
        'category': categorize_kenyan_plate(full) if full else None,
    }


def extract_plate(image_bytes: bytes, shared_checks: SharedPlateChecks | None = None) -> dict:
    """
    Find the number plates in one photo, to blur them and to fill in the form's registration fields.

    Google Vision reads the photo's text first. A plate at an angle, far away or only partly in the frame often has no
    text Vision can read, so when it reads no plate, or can't box the one it read, Gemini looks for plates instead:
    twice when the first search finds some or fails (one search's box can miss), then close up around each place found
    (_locate_plates_with_gemini), and every box is blurred. Only if the searches fail does Vision try again, on a
    contrast-enhanced copy. Within PHOTO_BUDGET: calls that would run past it are skipped.

    Returns full_plate, public_prefix, hidden_suffix, confidence, bounding_box and category, plus bounding_boxes:
    every plate to blur in the photo, bounding_box being the first.
    Pass the same SharedPlateChecks for every photo of one listing, so its plate is checked by Gemini once.
    """
    started = _now()

    def fits(seconds):
        """Whether a call that can take `seconds` still ends within the photo's PHOTO_BUDGET."""
        return _now() - started + seconds <= PHOTO_BUDGET

    empty = {'full_plate': None, 'public_prefix': None, 'hidden_suffix': None,
             'confidence': 0.0, 'bounding_box': None, 'category': None, 'bounding_boxes': []}
    try:
        client = _get_client()
    except Exception as e:
        print(f"Failed to initialise Google Vision client: {e}")
        return empty

    try:
        result = _detect_plate(client, image_bytes, shared_checks, fits)
    except Exception as e:
        print(f"Plate detection failed: {e}")
        result = None
    if result and result['bounding_box']:
        return {**result, 'bounding_boxes': [result['bounding_box']]}

    located = _locate_plates_with_gemini(image_bytes, fits)
    if located:
        return _located_result(image_bytes, located, shared_checks, result, fits)
    if result or located is not None:
        # A plate Vision read but couldn't box still fills in the form; Gemini finding no plate ends the search
        return {**result, 'bounding_boxes': []} if result else empty

    # Gemini's search failed: Vision again, on a contrast-enhanced copy, if there's time
    if not fits(VISION_TIMEOUT):
        return empty
    try:
        result = _detect_plate(client, _preprocess(image_bytes), shared_checks, fits)
    except Exception as e:
        print(f"Pre-processing fallback failed: {e}")
        return empty
    if not result:
        return empty
    return {**result, 'bounding_boxes': [result['bounding_box']] if result['bounding_box'] else []}


def categorize_kenyan_plate(plate_number: str) -> str:
    """Categorizes a Kenyan license plate into its specific format."""
    if not plate_number:
        return "Invalid plate"
        
    plate_number = plate_number.strip()
    
    # Define Regex Patterns
    standard_kd_pattern = r"(?i)^KD\s\d{3,4}[A-Z]?$"
    digital_kd_pattern = r"(?i)^KD[A-Z]\s\d{3}[A-Z]$"
    standard_national_pattern = r"(?i)^[A-Z]{3}\s\d{3}[A-Z]$"

    if re.match(digital_kd_pattern, plate_number):
        return "New Generation / KDK Digital Series Dealer Plate"
    elif re.match(standard_kd_pattern, plate_number):
        return "Standard / Historical KD Dealer Plate"
    elif re.match(standard_national_pattern, plate_number):
        return "Standard National Registration Plate (LLL NNN L)"
    else:
        return "Does not match any valid Kenyan registration formats provided"