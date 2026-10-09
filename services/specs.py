import json
from services import gemini

# Flash without thinking read the make, model and year the same as 2.5 Pro on a test listing, in ~6s instead of ~28s;
# Pro made this the slowest step the seller waits on. Switch back here if Flash starts misreading models.
SPEC_MODEL = 'gemini-2.5-flash'

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
# The colours the add-listing form offers and the API accepts (kemotives-laravel App\Support\VehicleSpecs::COLOURS)
VALID_COLOURS = [
    'White', 'Silver', 'Grey', 'Black', 'Blue', 'Red', 'Maroon', 'Green',
    'Brown', 'Beige', 'Gold', 'Orange', 'Yellow', 'Purple', 'Other'
]

CONFIDENCE_THRESHOLD = 0.69

SPEC_PROMPT = """
You are a professional vehicle inspector analyzing a car photo for a Kenyan vehicle marketplace.
Carefully analyze the vehicle's body shape, logos, and badges to determine the exact make and model.
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
  "condition":      {"value": "Foreign Used Unregistered", "confidence": 0.82},
  "colour":         {"value": "White",                     "confidence": 0.99},
  "trim":           {"value": "G",                         "confidence": 0.50}
}
colour is the body colour, one of: """ + ', '.join(VALID_COLOURS) + """ (pearl white is White, gunmetal is Grey).
Return ONLY the JSON object.
"""

def extract_specs(images: list[bytes]) -> dict:
    try:
        contents = [SPEC_PROMPT, *(gemini.image_part(img_bytes) for img_bytes in images)]

        print("--- Gemini Spec Extraction Start ---")
        print(f"Model: {SPEC_MODEL}")
        response = gemini.client().models.generate_content(
            model=SPEC_MODEL,
            contents=contents,
            config=gemini.READING,
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

            if field == 'colour':
                # The form fills it in only when it's confident enough; a colour off the list ("Pearl") isn't used
                result['colour'] = next((c for c in VALID_COLOURS if c.lower() == str(value).strip().lower()), None)
                continue

            if field == 'trim':
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
