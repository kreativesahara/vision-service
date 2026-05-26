import re
import numpy as np
from google.cloud import vision

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
    clean = re.sub(r'[\n\s]', '', texts[0].description.upper())
    match = re.search(r'(K[A-Z]{2})(\d{3}[A-Z]?)', clean)

    if not match:
        return None

    prefix, suffix = match.group(1), match.group(2)

    # Build bounding box from all annotations that belong to the plate
    vertices = [
        v
        for t in texts[1:]
        for v in t.bounding_poly.vertices
        if t.description.upper() in clean and (
            t.description.upper() in prefix or
            t.description.upper() in suffix or
            prefix in t.description.upper() or
            suffix in t.description.upper()
        )
    ]

    box = None
    if vertices:
        xs, ys = [v.x for v in vertices], [v.y for v in vertices]
        box = [
            {"x": min(xs), "y": min(ys)},
            {"x": max(xs), "y": min(ys)},
            {"x": max(xs), "y": max(ys)},
            {"x": min(xs), "y": max(ys)},
        ]

    return {
        'full_plate': f"{prefix} {suffix}",
        'public_prefix': prefix,
        'hidden_suffix': suffix,
        'confidence': 0.95,
        'bounding_box': box,
    }


def extract_plate(image_bytes: bytes) -> dict:
    """
    Attempt plate detection; fall back to pre-processed image on miss.
    Returns a dict with full_plate, public_prefix, hidden_suffix, confidence, bounding_box.
    """
    empty = {'full_plate': None, 'public_prefix': None, 'hidden_suffix': None,
             'confidence': 0.0, 'bounding_box': None}
    try:
        client = _get_client()
    except Exception as e:
        print(f"Failed to initialise Google Vision client: {e}")
        return empty

    result = _detect_plate(client, image_bytes)
    if result:
        return result

    # Fallback: contrast-enhanced grayscale
    try:
        result = _detect_plate(client, _preprocess(image_bytes))
        return result if result else empty
    except Exception as e:
        print(f"Pre-processing fallback failed: {e}")
        return empty