import cv2
import numpy as np
from google.cloud import vision
import os
import re

def get_vision_client():
    try:
        return vision.ImageAnnotatorClient()
    except Exception as e:
        print(f"Failed to initialize Google Vision client: {e}")
        return None

def extract_plate(image_bytes: bytes) -> dict:
    client = get_vision_client()
    if not client:
        return {'full_plate': None, 'confidence': 0.0}

    # Strategy 1: Try original image first (often better for colored plates like yellow)
    result = _perform_detection(client, image_bytes)
    if result.get('full_plate'):
        return result

    # Strategy 2: Pre-processing fallback (Grayscale + Contrast Enhancement)
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_img = clahe.apply(gray)
            _, buffer = cv2.imencode('.jpg', contrast_img)
            result = _perform_detection(client, buffer.tobytes())
            return result
    except Exception as e:
        print(f"Pre-processing failed: {e}")

    return {
        'full_plate': None,
        'public_prefix': None,
        'hidden_suffix': None,
        'confidence': 0.0,
    }

def _perform_detection(client, content_bytes):
    try:
        image = vision.Image(content=content_bytes)
        response = client.text_detection(image=image)
        
        if response.error.message:
            return {'full_plate': None}

        texts = response.text_annotations
        if not texts:
            return {'full_plate': None}

        raw_text = texts[0].description
        # Clean text and search for Kenyan plate format (LLL NNNL or LLL NNN)
        clean_text = re.sub(r'[\n\s]', '', raw_text.upper())
        # Standard Kenyan format since 2007: KXX 000X. Older: KXX 000.
        plate_pattern = re.search(r'(K[A-Z]{2})(\d{3}[A-Z]?)', clean_text)

        if plate_pattern:
            prefix = plate_pattern.group(1)
            suffix = plate_pattern.group(2)
            
            # Robust bounding box extraction:
            # Collect all annotations that contribute to the plate text
            all_vertices = []
            # We look for words that are part of the prefix or suffix
            for text in texts[1:]:
                txt = text.description.upper()
                # Check if this annotation is a part of our detected plate
                if txt in clean_text and (txt in prefix or txt in suffix or prefix in txt or suffix in txt):
                    all_vertices.extend(text.bounding_poly.vertices)
            
            box = None
            if all_vertices:
                xs = [v.x for v in all_vertices]
                ys = [v.y for v in all_vertices]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                box = [
                    {"x": min_x, "y": min_y},
                    {"x": max_x, "y": min_y},
                    {"x": max_x, "y": max_y},
                    {"x": min_x, "y": max_y}
                ]
            
            return {
                'full_plate': f"{prefix} {suffix}",
                'public_prefix': prefix,
                'hidden_suffix': suffix,
                'confidence': 0.95,
                'bounding_box': box
            }

    except Exception as e:
        print(f"Internal detection error: {e}")
    
    return {'full_plate': None}