import openai
import base64
import json
import os

client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

CONDITION_PROMPT = """
You are a certified vehicle condition inspector. Analyze this car photo and return ONLY valid JSON:
{
  "grade": "good",
  "score": 74,
  "damage_flags": ["minor scratch on rear bumper"],
  "interior_condition": "good",
  "notes": "Overall well maintained."
}
"""

def assess_condition(image_bytes: bytes) -> dict:
    b64_image = base64.standard_b64encode(image_bytes).decode('utf-8')
    try:
        response = client.chat.completions.create(
            model='gpt-4o',
            max_tokens=400,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/jpeg;base64,{b64_image}"
                        }
                    },
                    {'type': 'text', 'text': CONDITION_PROMPT}
                ]
            }]
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Condition assessment failed: {e}")
        return {
            'grade': 'unknown',
            'score': 0,
            'damage_flags': [],
            'interior_condition': None,
            'notes': 'Assessment unavailable',
        }
