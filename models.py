# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import Optional, List

class DuplicateResult(BaseModel):
    is_duplicate: bool
    duplicate_listing_id: Optional[str]
    confidence: float
    hashes: List[str]

class PlateDetection(BaseModel):
    full_plate: str
    public_prefix: str
    hidden_suffix: str
    bounding_box: List[dict]
    image_index: int

class PlateResult(BaseModel):
    full_plate: Optional[str]
    public_prefix: Optional[str]    # Shown publicly e.g. KBB
    hidden_suffix: Optional[str]    # Gated behind payment e.g. 675B
    confidence: float
    detections: List[PlateDetection] = []

class SpecResult(BaseModel):
    # --- Core fields matching addProduct.jsx `values` state ---
    make: Optional[str]             # e.g. "Toyota"
    model: Optional[str]            # e.g. "Corolla Fielder"
    year: Optional[int]             # e.g. 2019 (maps to `yom` in DB)
    engineCapacity: Optional[str]   # e.g. "1500" (maps to `engine_capacity`)
    fuelType: Optional[str]         # e.g. "Petrol" (maps to `fuel_type`)
    transmission: Optional[str]     # e.g. "Automatic"
    driveSystem: Optional[str]      # e.g. "4WD" (maps to `drive_system`)
    category: Optional[str]         # e.g. "Station Wagon" (maps to `category`)
    condition: Optional[str]        # e.g. "Foreign Used Unregistered"

    # --- Advisory / display fields ---
    colour: Optional[str]           # e.g. "Pearl White" (not in DB, shown in UI)
    trim: Optional[str]             # e.g. "X Grade" (not in DB, advisory only)

    # --- Vision metadata ---
    confidence: float               # 0.0 to 1.0
    autopopulate: bool              # True when confidence >= 0.75

class ConditionResult(BaseModel):
    grade: str                      # excellent / good / fair / poor
    score: int                      # 0-100
    damage_flags: List[str]         # e.g. ['front bumper crack', 'paint fade']
    interior_condition: Optional[str]
    notes: Optional[str]

class VisionResponse(BaseModel):
    duplicate: DuplicateResult
    plate: PlateResult
    specs: SpecResult
    condition: ConditionResult
    processing_time_ms: int
