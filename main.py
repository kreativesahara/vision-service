import time
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import httpx
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

from models import VisionResponse
from services.duplicate import get_hash, check_duplicates
from services.plate import extract_plate
from services.specs import extract_specs
from services.condition import assess_condition

app = FastAPI(title='Vehicle Vision Microservice')

# Use a wild card or environment variable for CORS in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=['POST', 'GET'],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

@app.post('/analyse', response_model=VisionResponse)
async def analyse_vehicle(
    images: List[UploadFile] = File(...),
    listing_id: str = None
):
    start = time.time()
    if not images:
        raise HTTPException(status_code=400, detail='No images provided')

    image_bytes_list = [await img.read() for img in images]
    primary_image = image_bytes_list[0]

    # 1. Perceptual hashing for duplicate check
    new_hashes = [get_hash(b) for b in image_bytes_list]

    # 2. Mock or Fetch existing hashes from Supabase
    existing = []
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/products",
                    params={'select': 'id,image_hashes'},
                    headers={
                        'apikey': SUPABASE_KEY,
                        'Authorization': f'Bearer {SUPABASE_KEY}',
                    },
                    timeout=5.0
                )
                if resp.status_code == 200:
                    existing = resp.json()
        except Exception as e:
            print(f"Supabase fetch error: {e}")

    if listing_id:
        existing = [l for l in existing if str(l.get('id')) != str(listing_id)]

    # 3. Parallel Execution of AI checks
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    # Plate detection: try all images until one works
    async def get_plate_result():
        for idx, b in enumerate(image_bytes_list):
            res = await loop.run_in_executor(executor, extract_plate, b)
            if res.get('full_plate'):
                res['image_index'] = idx
                return res
        return {
            'full_plate': None,
            'public_prefix': None,
            'hidden_suffix': None,
            'confidence': 0.0,
            'bounding_box': None,
            'image_index': None
        }

    duplicate_task = loop.run_in_executor(executor, check_duplicates, new_hashes, existing)
    plate_task = get_plate_result()
    specs_task = loop.run_in_executor(executor, extract_specs, primary_image)
    condition_task = loop.run_in_executor(executor, assess_condition, primary_image)

    duplicate_result, plate_result, specs_result, condition_result = await asyncio.gather(
        duplicate_task, plate_task, specs_task, condition_task
    )

    elapsed = int((time.time() - start) * 1000)

    return VisionResponse(
        duplicate=duplicate_result,
        plate=plate_result,
        specs=specs_result,
        condition=condition_result,
        processing_time_ms=elapsed,
    )

@app.get('/health')
def health():
    return {'status': 'ok'}
