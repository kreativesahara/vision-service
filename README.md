# Vehicle Vision Microservice — AI & Image Processing Architecture

This document provides a comprehensive overview of how **AI and Image Processing** are implemented within the `vision-service` microservice.

---

## 1. High-Level Architecture & Orchestration

When vehicle images are submitted to the service (via the FastAPI endpoint `POST /analyse` in [`main.py`](./main.py#L32-L125) or the WSGI handler in [`passenger_wsgi.py`](./passenger_wsgi.py#L150-L253)), the system orchestrates four distinct inspection pipelines.

To achieve high throughput and low latency, the tasks are dispatched in parallel using Python's `asyncio.gather` and a `ThreadPoolExecutor` (configured with 4 workers):
1. **Duplicate Detection** (runs across all uploaded images)
2. **License Plate Recognition & Masking** (runs across all uploaded images)
3. **Vehicle Specification Extraction** (runs across all uploaded images collectively)
4. **Condition & Damage Assessment** (runs on the primary image)

---

## 2. The 4 Pillars of AI & Image Processing

### A. Perceptual Hashing & Duplicate Detection ([`services/duplicate.py`](./services/duplicate.py))

Instead of relying on heavy deep learning embeddings for duplicate checking, the codebase uses **Classical Computer Vision (Perceptual Hashing)** via `Pillow` and `imagehash` for fast, deterministic similarity searching:
* **Algorithm**: Each image is converted to RGB and evaluated using a combined hash: **Average Hash (`ahash`)** and **Difference Hash (`dhash`)** at a 16x16 resolution.
* **Matching**: Hashes are stored as `ahash:dhash` strings in Supabase. When a new listing is submitted, the system computes the Hamming distance between the new image hashes and stored listing hashes.
* **Thresholding**: A Hamming distance threshold of `8` (`DUPLICATE_THRESHOLD`) is used. A distance $\le 8$ flags the image as a duplicate and calculates a linear confidence score:
  $$\text{Confidence} = 1 - \left(\frac{\text{Distance}}{8}\right)$$

---

### B. Hybrid License Plate Recognition & Categorization ([`services/plate.py`](./services/plate.py))

License plate detection uses a **3-stage Hybrid Pipeline** combining classical image processing, optical character recognition (OCR), and Vision LLMs:

1. **OCR Text Detection & Fallback Preprocessing**:
   * Uses the **Google Cloud Vision API** (`ImageAnnotatorClient.text_detection`) to extract text annotations and bounding boxes.
   * **CLAHE Fallback**: If standard detection misses the plate, OpenCV (`cv2`) converts the image to grayscale and applies Contrast Limited Adaptive Histogram Equalization (**CLAHE**) with a clip limit of 2.0 to enhance plate contrast before retrying OCR.
2. **Kenyan Plate Parsing & Error Correction**:
   * Extracts alphanumeric tokens and matches them against standard Kenyan formats (`KXX NNNX` or `KD` dealer plates).
   * Automatically corrects common OCR alphanumeric substitutions (e.g., mapping `O` $\rightarrow$ `0`, `I` $\rightarrow$ `1`, `S` $\rightarrow$ `5`, `Z` $\rightarrow$ `2`).
3. **3-Tier Bounding Box Localization & Gemini Verification**:
   * Locates the plate's bounding polygon using 3 fallback heuristics (Strict word match $\rightarrow$ Substring/Bigram overlap $\rightarrow$ 50% character overlap).
   * **LLM Verification**: Once the bounding box is found, OpenCV crops the plate region (with a 10% padding margin) and sends the cropped image to **Gemini 2.5 Flash** (via the Google GenAI Vertex AI client) with a strict prompt to double-check and confirm the exact plate characters.
4. **Privacy & Categorization**:
   * Splitting the plate into a `public_prefix` (e.g., `KBB`) and a `hidden_suffix` (e.g., `675B`) allows the marketplace to show public teasers while gating full registration numbers behind paywalls or user verification.
   * Categorizes the plate into historical/digital dealer series (`KD`) or standard national series.

---

### C. Multimodal Specification Extraction ([`services/specs.py`](./services/specs.py))

To auto-populate listing details for sellers, the service uses high-reasoning multimodal AI (**Gemini 2.5 Pro** via Vertex AI):
* **Multi-Image Ingestion**: Passes all uploaded vehicle photos (supporting JPEG and PNG byte detection) in a single prompt context so the model can inspect multiple angles of the vehicle simultaneously.
* **Prompt Engineering & Structured JSON**: Instructs the model to act as a professional Kenyan vehicle inspector, analyzing body shape, badges, and trim levels to return a structured JSON object containing exact attributes and individual confidence scores.
* **Validation & Thresholding**:
  * Applies a strict `CONFIDENCE_THRESHOLD` of **0.69**. Any extracted field below this confidence score is discarded.
  * Validates extracted fields against marketplace domain enums (`VALID_FUEL_TYPES`, `VALID_TRANSMISSIONS`, `VALID_DRIVE_SYSTEMS`, `VALID_CATEGORIES`, `VALID_CONDITIONS`).
  * Sets `autopopulate = True` if reliable data is extracted, allowing the frontend form to automatically fill in the vehicle's make, model, year, engine capacity, transmission, and body type.

---

### D. Condition & Damage Assessment ([`services/condition.py`](./services/condition.py))

For rapid automated grading, the service evaluates the vehicle's physical condition using **Gemini 2.5 Flash**:
* Analyzes the primary vehicle image to return a normalized condition report.
* Outputs an overall `grade` (`excellent`, `good`, `fair`, or `poor`), a numerical quality `score` (`0-100`), a list of specific `damage_flags` (e.g., `"minor scratch on rear bumper"`), and inspector notes.

---

## 3. Key Technical & Infrastructure Optimizations

* **Thread Contention Management**: In [`passenger_wsgi.py:L11-L15`](./passenger_wsgi.py#L11-L15), environment variables (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`) are explicitly set to `"1"` before importing OpenCV or NumPy. This prevents CPU thread oversubscription and deadlocks when running NumPy/OpenCV inside WSGI web server workers and Python thread pools.
* **Model Tiering (Cost vs. Performance)**:
  * **Gemini 2.5 Pro** is selectively reserved for *Specification Extraction* where deep visual reasoning (distinguishing car trims, sub-models, and reading badges across multiple angles) is required.
  * **Gemini 2.5 Flash** is used for *Condition Assessment* and *Plate Verification*, where high speed, lower latency, and cost-efficiency are prioritized over complex reasoning.
* **Unified SDK Migration**: The codebase uses the modern, unified `google-genai` SDK (`from google import genai`), connecting directly to Google Cloud Vertex AI backend (`vertexai=True`, `location='us-central1'`).
