import pytesseract
import cv2
import numpy as np
import re
import os

if os.getenv('TESSERACT_CMD'):
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')

def extract_plate(image_bytes: bytes) -> dict:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        # Basic preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        raw_text = pytesseract.image_to_string(
            thresh,
            config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        # Kenyan plate format: [A-Z]{3}\s*\d{3}[A-Z]
        plate_pattern = re.search(r'([A-Z]{3})\s*(\d{3}[A-Z])', raw_text.upper())

        if plate_pattern:
            prefix = plate_pattern.group(1)
            suffix = plate_pattern.group(2)
            return {
                'full_plate': f"{prefix} {suffix}",
                'public_prefix': prefix,
                'hidden_suffix': suffix,
                'confidence': 0.90,
            }
    except Exception as e:
        print(f"Plate extraction error: {e}")

    return {
        'full_plate': None,
        'public_prefix': None,
        'hidden_suffix': None,
        'confidence': 0.0,
    }
