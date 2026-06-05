import os
import json
from google import genai
from google.genai import types

def test_gemini():
    client = genai.Client(
        vertexai=True,
        project=os.getenv('GCP_PROJECT_ID', 'kemotives'),
        location='us-central1'
    )
    
    image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    
    prompt = "Find the Kenyan license plate in this image. Return a JSON object with 'plate' (the text) and 'box' [ymin, xmin, ymax, xmax] as normalized coordinates (0-1000)."
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type='image/png')]
        )
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    test_gemini()
