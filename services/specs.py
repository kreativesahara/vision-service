import openai
import base64
import json
import os

client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Valid options that exactly match the addProduct.jsx selects and DB schema enums
VALID_FUEL_TYPES = ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'LPG']
VALID_TRANSMISSIONS = ['Automatic', 'Manual', 'CVT', 'Semi-Automatic']
VALID_DRIVE_SYSTEMS = ['2WD', '4WD', 'AWD']
VALID_CATEGORIES = [
    'Saloon', 'Station Wagon', 'SUV', 'Hatchback', 'Pickup',
    'Van', 'Truck', 'Bus', 'Minivan', 'Coupe', 'Convertible'
]
VALID_CONDITIONS = [
    'New', 'Foreign Used Unregistered', 'Foreign Used Registered',
    'Local Used', 'Reconditioned', 'Certified Pre-Owned'
]

SPEC_PROMPT = """
You are a professional vehicle inspector analyzing a car photo for a Kenyan vehicle marketplace.

Extract the following details and return ONLY valid JSON — no markdown, no explanation, just the JSON object:

{
  "make": "Toyota",
  "model": "Corolla Fielder",
  "year": 2019,
  "engineCapacity": "1500",
  "fuelType": "Petrol",
  "transmission": "Automatic",
  "driveSystem": "4WD",
  "category": "Station Wagon",
  "condition": "Foreign Used Unregistered",
  "colour": "Pearl White",
  "trim": "X Grade",
  "confidence": 0.92
}

Field rules:
- make: Car manufacturer (e.g. Toyota, Subaru, Mazda, Honda, Nissan, Mitsubishi, BMW, Mercedes-Benz)
- model: Full model name (e.g. "Corolla Fielder", "Jimny Sierra", "CX-5")
- year: 4-digit year estimated from styling cues
- engineCapacity: Engine size in cc as a STRING without units (e.g. "1500", "2000", "2400")
- fuelType: One of exactly: Petrol, Diesel, Hybrid, Electric, LPG
- transmission: One of exactly: Automatic, Manual, CVT, Semi-Automatic
- driveSystem: One of exactly: 2WD, 4WD, AWD
- category: One of exactly: Saloon, Station Wagon, SUV, Hatchback, Pickup, Van, Truck, Bus, Minivan, Coupe, Convertible
- condition: One of exactly: New, Foreign Used Unregistered, Foreign Used Registered, Local Used, Reconditioned, Certified Pre-Owned
- colour: Colour of the vehicle (advisory only, not in DB)
- trim: Trim/grade level if visible (advisory only)
- confidence: 0.0 to 1.0 — how confident you are in these details from the image

Use null for any field you cannot determine. Return ONLY the JSON object.
"""

def extract_specs(image_bytes: bytes) -> dict:
    b64_image = base64.standard_b64encode(image_bytes).decode('utf-8')

    # Detect media type
    media_type = 'image/jpeg'
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        media_type = 'image/png'
    elif image_bytes[:4] == b'RIFF':
        media_type = 'image/webp'

    try:
        response = client.chat.completions.create(
            model='gpt-4o',
            max_tokens=600,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:{media_type};base64,{b64_image}"
                        }
                    },
                    {
                        'type': 'text',
                        'text': SPEC_PROMPT,
                    }
                ]
            }]
        )

        raw = response.choices[0].message.content.strip()
        # Strip any accidental markdown code fences
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]

        specs = json.loads(raw.strip())

        # Sanitize: enforce that enum fields only contain known values
        if specs.get('fuelType') not in VALID_FUEL_TYPES:
            specs['fuelType'] = None
        if specs.get('transmission') not in VALID_TRANSMISSIONS:
            specs['transmission'] = None
        if specs.get('driveSystem') not in VALID_DRIVE_SYSTEMS:
            specs['driveSystem'] = None
        if specs.get('category') not in VALID_CATEGORIES:
            specs['category'] = None
        if specs.get('condition') not in VALID_CONDITIONS:
            specs['condition'] = None

        # Coerce engineCapacity to string (no units)
        if specs.get('engineCapacity') is not None:
            specs['engineCapacity'] = str(specs['engineCapacity']).replace('cc', '').strip()

        # Coerce year to int
        if specs.get('year') is not None:
            specs['year'] = int(specs['year'])

        specs['autopopulate'] = specs.get('confidence', 0) >= 0.75
        return specs

    except Exception as e:
        print(f"Spec extraction failed: {e}")
        return {
            'make': None,
            'model': None,
            'year': None,
            'engineCapacity': None,
            'fuelType': None,
            'transmission': None,
            'driveSystem': None,
            'category': None,
            'condition': None,
            'colour': None,
            'trim': None,
            'confidence': 0.0,
            'autopopulate': False,
        }
