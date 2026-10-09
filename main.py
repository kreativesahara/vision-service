import os

# Native thread pools default to one thread per CPU core (96 on the server); cap them before numpy loads,
# as passenger_wsgi.py does in production
for _var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import time
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# OpenCV ignores the variables above and needs its own cap
import cv2
cv2.setNumThreads(1)

from models import VisionResponse, PlateResult, PlateDetection
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

@app.post('/analyse', response_model=VisionResponse)
async def analyse_vehicle(images: List[UploadFile] = File(...)):
    start = time.time()
    if not images:
        raise HTTPException(status_code=400, detail='No images provided')

    image_bytes_list = [await img.read() for img in images]

    # Duplicate listings are spotted by the Laravel API on submit (App\Services\DuplicateListingCheck), as in passenger_wsgi.py

    # Parallel Execution of AI checks
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    # Plate detection: process ALL images to ensure all are blurred for privacy
    async def get_plate_result():
        all_detections = []
        primary_plate = None
        for idx, b in enumerate(image_bytes_list):
            res = await loop.run_in_executor(executor, extract_plate, b)
            if res.get('bounding_box') or res.get('full_plate'):
                det = PlateDetection(
                    full_plate=res.get('full_plate'),
                    public_prefix=res.get('public_prefix'),
                    hidden_suffix=res.get('hidden_suffix'),
                    bounding_box=res.get('bounding_box'),
                    image_index=idx
                )
                all_detections.append(det)
                if not primary_plate and det.full_plate:
                    primary_plate = det
        
        if not all_detections:
            return PlateResult(
                full_plate=None,
                public_prefix=None,
                hidden_suffix=None,
                confidence=0.0,
                detections=[]
            )
        
        return PlateResult(
            full_plate=primary_plate.full_plate if primary_plate else None,
            public_prefix=primary_plate.public_prefix if primary_plate else None,
            hidden_suffix=primary_plate.hidden_suffix if primary_plate else None,
            confidence=0.95 if primary_plate else 0.0,
            detections=all_detections
        )

    plate_task = get_plate_result()
    specs_task = loop.run_in_executor(executor, extract_specs, image_bytes_list)
    condition_task = loop.run_in_executor(executor, assess_condition, image_bytes_list)

    plate_result, specs_result, condition_result = await asyncio.gather(
        plate_task, specs_task, condition_task
    )

    elapsed = int((time.time() - start) * 1000)

    return VisionResponse(
        plate=plate_result,
        specs=specs_result,
        condition=condition_result,
        processing_time_ms=elapsed,
    )

@app.get('/health')
def health():
    return {'status': 'ok'}
