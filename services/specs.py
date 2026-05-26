import json
import os
from google import genai
from google.genai import types
import base64

# --- Vertex AI Client using the new google-genai SDK ---

def _get_client():
    """Initialize the unified GenAI client for Vertex AI."""
    return genai.Client(
        vertexai=True,
        project=os.getenv('GCP_PROJECT_ID', 'kemotives'),
        location='us-central1'
    )

# --- Spec Extraction Logic ---

VALID_FUEL_TYPES = ['Petrol', 'Petrol Hybrid', 'Diesel', 'Diesel Hybrid', 'Hybrid', 'Electric']
VALID_TRANSMISSIONS = ['Automatic', 'Manual', 'CVT']
VALID_DRIVE_SYSTEMS = ['2WD', '4WD', 'AWD']
VALID_CATEGORIES = [
    'Saloon', 'Station Wagon', 'SUV', 'Hatchback', 'Pickup',
    'Van', 'Truck', 'Bus', 'Minivan', 'Coupe', 'Convertible'
]
VALID_CONDITIONS = [
    'New', 'Foreign Used Unregistered', 'Foreign Used Registered',
    'Local Used', 'Reconditioned', 'Certified Pre-Owned'
]

CONFIDENCE_THRESHOLD = 0.90

SPEC_PROMPT = """
You are a professional vehicle inspector analyzing a car photo for a Kenyan vehicle marketplace.
Extract details and return ONLY valid JSON:
{
  "make":           {"value": "Toyota",                    "confidence": 0.97},
  "model":          {"value": "Corolla Fielder",           "confidence": 0.95},
  "year":           {"value": 2019,                        "confidence": 0.80},
  "engineCapacity": {"value": "1500",                      "confidence": 0.70},
  "fuelType":       {"value": "Petrol",                    "confidence": 0.90},
  "transmission":   {"value": "Automatic",                 "confidence": 0.85},
  "driveSystem":    {"value": "4WD",                       "confidence": 0.75},
  "category":       {"value": "Station Wagon",             "confidence": 0.98},
  "condition":      {"value": "Foreign Used Unregistered", "confidence": 0.82}
}
Return ONLY the JSON object.
"""

def extract_specs(image_bytes: bytes) -> dict:
    try:
        client = _get_client()
        
        # Detect mime type
        mime_type = 'image/jpeg'
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = 'image/png'

        print(f"--- Gemini Spec Extraction Start ({client.vertexai=}) ---")
        print(f"Model: gemini-2.5-flash")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                SPEC_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ]
        )
        
        raw = response.text.strip()
        print(f"Raw Output: {raw[:500]}...") # Log first 500 chars

        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        raw = raw.strip()

        per_field = json.loads(raw)

        result = {
            'make': None, 'model': None, 'year': None,
            'engineCapacity': None, 'fuelType': None,
            'transmission': None, 'driveSystem': None,
            'category': None, 'condition': None,
            'colour': None, 'trim': None,
            'confidence': 0.0,
            'field_confidences': {},
            'autopopulate': False,
        }

        enum_validators = {
            'fuelType': VALID_FUEL_TYPES,
            'transmission': VALID_TRANSMISSIONS,
            'driveSystem': VALID_DRIVE_SYSTEMS,
            'category': VALID_CATEGORIES,
            'condition': VALID_CONDITIONS,
        }

        max_confidence = 0.0
        for field, data in per_field.items():
            if field not in result and field not in ['colour', 'trim']:
                continue
            value = data.get('value')
            confidence = float(data.get('confidence', 0.0))
            result['field_confidences'][field] = confidence
            if confidence > max_confidence:
                max_confidence = confidence

            if value is None:
                continue

            if field in ['colour', 'trim']:
                result[field] = value
                continue

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            if field in enum_validators and value not in enum_validators[field]:
                continue

            if field == 'year' and value is not None:
                try: value = int(value)
                except: value = None
            if field == 'engineCapacity' and value is not None:
                value = str(value).replace('cc', '').strip()

            result[field] = value

        result['confidence'] = max_confidence
        result['autopopulate'] = any(
            v is not None for k, v in result.items()
            if k not in ('field_confidences', 'autopopulate', 'confidence', 'colour', 'trim')
        )

        print(f"Parsed Results: {json.dumps({k:v for k,v in result.items() if k != 'field_confidences'}, indent=2)}")
        print("--- Gemini Spec Extraction End ---")
        return result

    except Exception as e:
        print(f"Spec extraction failed: {e}")
        return {
            'make': None, 'model': None, 'year': None,
            'engineCapacity': None, 'fuelType': None,
            'transmission': None, 'driveSystem': None,
            'category': None, 'condition': None,
            'colour': None, 'trim': None,
            'confidence': 0.0,
            'field_confidences': {},
            'autopopulate': False,
        }
