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
    # 1. Pre-processing for better OCR
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Increase contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_img = clahe.apply(gray)
            
            # Encode back to bytes for Google Vision
            _, buffer = cv2.imencode('.jpg', contrast_img)
            processed_bytes = buffer.tobytes()
        else:
            processed_bytes = image_bytes
    except Exception as e:
        print(f"Pre-processing failed, using original: {e}")
        processed_bytes = image_bytes

    client = get_vision_client()
    if not client:
        return {
            'full_plate': None,
            'public_prefix': None,
            'hidden_suffix': None,
            'confidence': 0.0,
        }

    try:
        image = vision.Image(content=processed_bytes)
        # Switched to TEXT_DETECTION as requested (better for sparse text like plates)
        response = client.text_detection(image=image)
        
        if response.error.message:
            print(f"ERROR: Google Vision API returned an error: {response.error.message}")
            return {
                'full_plate': None,
                'public_prefix': None,
                'hidden_suffix': None,
                'confidence': 0.0,
            }

        texts = response.text_annotations
        if not texts:
            return {
                'full_plate': None,
                'public_prefix': None,
                'hidden_suffix': None,
                'confidence': 0.0,
            }

        raw_text = texts[0].description
        print(f"DEBUG: Google Vision detected text:\n{raw_text}")
        
        # Clean text and search for Kenyan plate format
        clean_text = re.sub(r'[\n\s]', '', raw_text.upper())
        plate_pattern = re.search(r'([A-Z]{3})(\d{3}[A-Z])', clean_text)

        if plate_pattern:
            prefix = plate_pattern.group(1)
            suffix = plate_pattern.group(2)
            
            # Find the bounding box for the plate
            # For simplicity, we use the bounding box of the annotation that contains the plate text
            # In TEXT_DETECTION, texts[0] is the whole block, but we can look for specific matches
            box = None
            for text in texts:
                if prefix in text.description.upper() or suffix in text.description.upper():
                    box = [{"x": v.x, "y": v.y} for v in text.bounding_poly.vertices]
                    break
            
            print(f"SUCCESS: Google Vision extracted plate: {prefix} {suffix}")
            return {
                'full_plate': f"{prefix} {suffix}",
                'public_prefix': prefix,
                'hidden_suffix': suffix,
                'confidence': 0.95,
                'bounding_box': box
            }

    except Exception as e:
        print(f"Google Vision Plate extraction error: {e}")

    return {
        'full_plate': None,
        'public_prefix': None,
        'hidden_suffix': None,
        'confidence': 0.0,
    }