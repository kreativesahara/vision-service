import httpx
from PIL import Image
import io
import json
import asyncio

async def test_vision():
    print("1. Creating a dummy test image...")
    img = Image.new('RGB', (800, 600), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    print(f"   Created {len(img_bytes)} bytes image.")

    print("2. Sending to Vision Service (http://localhost:8000/analyse)...")
    try:
        files = {
            'images': ('car.jpg', img_bytes, 'image/jpeg')
        }
        async with httpx.AsyncClient() as client:
            res = await client.post('http://localhost:8000/analyse', files=files, timeout=60.0)
            
            if res.status_code == 200:
                print("\n=== VISION ANALYSIS SUCCESS ===")
                print(json.dumps(res.json(), indent=2))
            else:
                print(f"\n=== VISION ANALYSIS FAILED (Status: {res.status_code}) ===")
                print(res.text)
    except Exception as e:
        print(f"\nError connecting to vision service: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision())
