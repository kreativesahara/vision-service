import os
import json
from google import genai
from google.genai import types

def _get_client():
    return genai.Client(
        vertexai=True,
        project=os.getenv('GCP_PROJECT_ID', 'kemotives'),
        location='us-central1'
    )

def test_gemini():
    client = _get_client()
    # We will pass a dummy image just to see if the API responds
    # Let's read an image from the filesystem if possible, or just create a 1x1 png
    image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    
    prompt = """
You are a vehicle analyzer. Extract the Kenyan license plate from this image.
Return JSON ONLY:
{
  "full_plate": "KCG 260P",
  "public_prefix": "KCG",
  "hidden_suffix": "260P",
  "bounding_box": [
    {"x": 10, "y": 10},
    {"x": 100, "y": 10},
    {"x": 100, "y": 50},
    {"x": 10, "y": 50}
  ]
}
If no plate is found, return empty strings and null for bounding_box.
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type='image/png')
            ]
        )
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    test_gemini()
