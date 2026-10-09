import json
from services import gemini

# --- Condition Assessment Logic ---

CONDITION_PROMPT = """
You are a certified vehicle condition inspector. Analyze this car photo and return ONLY valid JSON:
{
  "grade": "excellent", 
  "score": 90,
  "damage_flags": ["minor scratch on rear bumper"],
  "interior_condition": "good",
  "notes": "Overall well maintained."
}
Grade must be one of: excellent, good, fair, poor.
Score must be 0-100.
Return ONLY the JSON object.
"""

def assess_condition(image_bytes: bytes) -> dict:
    try:
        print(f"--- Gemini Condition Assessment Start ---")
        print(f"Model: gemini-2.5-flash")
        response = gemini.client().models.generate_content(
            model='gemini-2.5-flash',
            contents=[CONDITION_PROMPT, gemini.image_part(image_bytes)],
            config=gemini.READING,
        )
        
        raw = response.text.strip()
        print(f"Raw Output: {raw[:500]}...")
        
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        raw = raw.strip()
        
        res = json.loads(raw)
        print(f"Parsed Result: {json.dumps(res, indent=2)}")
        print("--- Gemini Condition Assessment End ---")
        return res
    except Exception as e:
        print(f"Condition assessment failed: {e}")
        return {
            'grade': 'unknown',
            'score': 0,
            'damage_flags': [],
            'interior_condition': None,
            'notes': 'Assessment unavailable',
        }
