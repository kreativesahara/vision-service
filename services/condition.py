import json
from services import gemini

# --- Condition Assessment Logic ---

# Sent every photo of the listing, not just the first, so the inside of the car is judged when the seller
# photographed it. The notes are saved with the listing and read by buyers, so they talk about the car.
CONDITION_PROMPT = """
You are a certified vehicle condition inspector. These are all the photos a seller took of one car for a
Kenyan vehicle marketplace: the outside, and often the inside. Assess the car from all of them together and
return ONLY valid JSON:
{
  "grade": "good",
  "score": 78,
  "damage_flags": ["minor scratch on rear bumper"],
  "interior_condition": "good",
  "notes": "Well maintained, with light wear for its age."
}
grade is one of: excellent, good, fair, poor. score is 0-100.
damage_flags lists the visible damage or wear, each item short.
interior_condition is one of excellent, good, fair, poor, judged from the photos of the inside; null if no photo shows the inside.
notes is one or two sentences for buyers about the car itself. Never mention the photos or images, or what they do or don't show.
Return ONLY the JSON object.
"""

def assess_condition(images: list[bytes]) -> dict:
    try:
        print(f"--- Gemini Condition Assessment Start ---")
        print(f"Model: gemini-2.5-flash, {len(images)} photos")
        response = gemini.client().models.generate_content(
            model='gemini-2.5-flash',
            contents=[CONDITION_PROMPT, *(gemini.image_part(image_bytes) for image_bytes in images)],
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
