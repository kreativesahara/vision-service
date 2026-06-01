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
    Run text detection and object localization.
    Extract Kenyan plate text if possible. Also find bounding box for blurring.
    """
    image = vision.Image(content=content_bytes)
    features = [
        vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
        vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION)
    ]
    request = vision.AnnotateImageRequest(image=image, features=features)
    response = client.annotate_image(request=request)

    prefix = None
    suffix = None
    full_plate = None
    confidence = 0.0
    text_box = None

    if response.text_annotations:
        texts = response.text_annotations
        # Strip everything except letters and numbers
        clean = re.sub(r'[^A-Z0-9]', '', texts[0].description.upper())
        match = re.search(r'(K[A-Z]{2})(\d{3}[A-Z]?)', clean)

        if match:
            prefix, suffix = match.group(1), match.group(2)
            full_plate = f"{prefix} {suffix}"
            confidence = 0.95

            vertices = []
            for t in texts[1:]:
                t_clean = re.sub(r'[^A-Z0-9]', '', t.description.upper())
                if not t_clean: 
                    continue
                # If this word is part of our plate prefix or suffix, grab its coordinates
                if t_clean in prefix or t_clean in suffix or prefix in t_clean or suffix in t_clean:
                    for v in t.bounding_poly.vertices:
                        vertices.append(v)

            if vertices:
                xs, ys = [v.x for v in vertices], [v.y for v in vertices]
                text_box = [
                    {"x": min(xs), "y": min(ys)},
                    {"x": max(xs), "y": min(ys)},
                    {"x": max(xs), "y": max(ys)},
                    {"x": min(xs), "y": max(ys)},
                ]

    obj_box = None
    import cv2
    nparr = np.frombuffer(content_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        for obj in response.localized_object_annotations:
            print(f"Vision API Detected Object: {obj.name} (score: {obj.score})")
            # Check for License Plate with >= 0.35 confidence
            if obj.name.lower() in ['license plate', 'vehicle registration plate', 'number plate'] and obj.score >= 0.35:
                verts = obj.bounding_poly.normalized_vertices
                if verts:
                    obj_box = [
                        {"x": int(v.x * w), "y": int(v.y * h)} for v in verts
                    ]
                break

    final_box = text_box if text_box else obj_box

    if final_box or full_plate:
        return {
            'full_plate': full_plate,
            'public_prefix': prefix,
            'hidden_suffix': suffix,
            'confidence': confidence,
            'bounding_box': final_box,
        }

    return None


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