import re
import numpy as np
import os
from google.cloud import vision
from google import genai
from google.genai import types

_vision_client = None

def _get_client():
    """Lazily initialise and cache the Google Vision client."""
    global _vision_client
    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()
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

        client = genai.Client(
            vertexai=True,
            project=os.getenv('GCP_PROJECT_ID', 'kemotives'),
            location='us-central1'
        )
        prompt = "Read the Kenyan license plate in this image. Return ONLY the text, e.g. 'KBS 865P'. Return nothing else."
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, types.Part.from_bytes(data=crop_bytes, mime_type='image/png')]
        )
        
        text = response.text.strip()
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        match = re.search(r'(K[A-Z]{2}|KD)([\dOISZ]{3,4}[A-Z]?)', clean)
        if match:
            prefix = match.group(1)
            suffix = match.group(2).translate(str.maketrans('OISZ', '0152'))
            return f"{prefix} {suffix}", prefix, suffix
            
    except Exception as e:
        print(f"Gemini verification failed: {e}")
        
    return None


def _detect_plate(client, content_bytes: bytes) -> dict | None:
    """
    Run text detection and extract a Kenyan plate in KXX NNNX format.
    Returns a result dict on success, None if no plate was found.
    """
    image = vision.Image(content=content_bytes)
    response = client.text_detection(image=image)

    if response.error.message or not response.text_annotations:
        return None

    texts = response.text_annotations
    # Strip everything except A-Z and 0-9 to handle dashes, spaces, dots, etc.
    clean = re.sub(r'[^A-Z0-9]', '', texts[0].description.upper())
    
    # Allow common OCR mistakes for numbers (O->0, I->1, S->5, Z->2)
    match = re.search(r'(K[A-Z]{2}|KD)([\dOISZ]{3,4}[A-Z]?)', clean)

    if not match:
        return None

    prefix = match.group(1)
    raw_suffix = match.group(2)
    # Correct the OCR mistakes in the final output
    suffix = raw_suffix.translate(str.maketrans('OISZ', '0152'))

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

    print(f"Plate '{prefix} {suffix}' detected — box tier result: {box is not None}")

    # Double check the text using Gemini 2.5 Flash on the cropped region
    if box:
        gemini_result = _verify_plate_text_with_gemini(content_bytes, box)
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


def extract_plate(image_bytes: bytes) -> dict:
    """
    Attempt plate detection; fall back to pre-processed image on miss.
    Returns a dict with full_plate, public_prefix, hidden_suffix, confidence, bounding_box.
    """
    empty = {'full_plate': None, 'public_prefix': None, 'hidden_suffix': None,
             'confidence': 0.0, 'bounding_box': None, 'category': None}
    try:
        client = _get_client()
    except Exception as e:
        print(f"Failed to initialise Google Vision client: {e}")
        return empty

    try:
        result = _detect_plate(client, image_bytes)
        if result:
            return result
    except Exception as e:
        print(f"Plate detection failed: {e}")

    # Fallback: contrast-enhanced grayscale
    try:
        result = _detect_plate(client, _preprocess(image_bytes))
        return result if result else empty
    except Exception as e:
        print(f"Pre-processing fallback failed: {e}")
        return empty


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