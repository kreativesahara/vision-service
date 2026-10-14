# Vehicle Vision Microservice — AI & Image Processing Architecture

This document provides a comprehensive overview of how **AI and Image Processing** are implemented within the `vision-service` microservice.

---

## 1. High-Level Architecture & Orchestration

When vehicle images are submitted to the service (via the FastAPI endpoint `POST /analyse` in [`main.py`](./main.py#L32-L125) or the WSGI handler in [`passenger_wsgi.py`](./passenger_wsgi.py#L150-L253)), the system orchestrates three inspection pipelines.

To achieve high throughput and low latency, the tasks are dispatched in parallel using Python's `asyncio.gather` and a `ThreadPoolExecutor` (configured with 4 workers):
1. **License Plate Recognition & Masking** (runs across all uploaded images)
2. **Vehicle Specification Extraction** (runs across all uploaded images collectively)
3. **Condition & Damage Assessment** (runs on the first three images together, so an interior photo among them counts)

Duplicate listings are no longer checked here. The Laravel API spots them when a listing is submitted (same registration, or two or more of the same photos; `App\Services\DuplicateListingCheck` in `kemotives-laravel`), because it knows the seller and the live listings and this public endpoint does not.

---

## 2. The 3 Pillars of AI & Image Processing

### A. Hybrid License Plate Recognition & Categorization ([`services/plate.py`](./services/plate.py))

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
   * **Once per plate**: a car's plate usually shows in several photos. `SharedPlateChecks` (one per analysis request) lets the first photo run the Gemini check and the others reuse its answer, waiting while it runs instead of repeating it; every photo still gets its own blur box. If that check fails, or takes longer than `SHARED_CHECK_WAIT` (45 s), a photo checks its own crop. A listing with its plate in 4 photos makes 1 Gemini call instead of 4.
4. **Privacy & Categorization**:
   * Splitting the plate into a `public_prefix` (e.g., `KBB`) and a `hidden_suffix` (e.g., `675B`) lets the marketplace show the prefix while keeping the rest private.
   * **Logs never hold the private part**: plates are logged masked (`Plate 'KBX ****' detected`), through `mask_plate()`. The server keeps its logs, and a plate is personal data. A test fails if a detected plate's suffix reaches the log.
   * Categorizes the plate into historical/digital dealer series (`KD`) or standard national series.

---

### B. Multimodal Specification Extraction ([`services/specs.py`](./services/specs.py))

To auto-populate listing details for sellers, the service uses multimodal AI (**Gemini 2.5 Flash** via Vertex AI; `SPEC_MODEL` in `services/specs.py`):
* **Multi-Image Ingestion**: Passes all uploaded vehicle photos (supporting JPEG and PNG byte detection) in a single prompt context so the model can inspect multiple angles of the vehicle simultaneously.
* **Prompt Engineering & Structured JSON**: Instructs the model to act as a professional Kenyan vehicle inspector, analyzing body shape, badges, and trim levels to return a structured JSON object containing exact attributes and individual confidence scores.
* **Validation & Thresholding**:
  * Applies a strict `CONFIDENCE_THRESHOLD` of **0.69**. Any extracted field below this confidence score is discarded.
  * Validates extracted fields against marketplace domain enums (`VALID_FUEL_TYPES`, `VALID_TRANSMISSIONS`, `VALID_DRIVE_SYSTEMS`, `VALID_CATEGORIES`, `VALID_CONDITIONS`).
  * Sets `autopopulate = True` if reliable data is extracted, allowing the frontend form to automatically fill in the vehicle's make, model, year, engine capacity, transmission, and body type.

---

### C. Condition & Damage Assessment ([`services/condition.py`](./services/condition.py))

For rapid automated grading, the service evaluates the vehicle's physical condition using **Gemini 2.5 Flash**:
* Analyzes the first three photos of the listing together, so a photo of the inside among them counts towards `interior_condition` (null when none shows the inside).
* Outputs an overall `grade` (`excellent`, `good`, `fair`, or `poor`), a numerical quality `score` (`0-100`), a list of specific `damage_flags` (e.g., `"minor scratch on rear bumper"`), and notes for buyers about the car (never about the photos). The marketplace saves these with the listing.
* If Gemini fails, the grade comes back as `unknown` with score `0`; the add-listing form doesn't save that, and the admin panel's "Listing checks" page lists the listing as not analysed.

---

## 3. Key Technical & Infrastructure Optimizations

* **Thread Contention Management**: In [`passenger_wsgi.py:L11-L15`](./passenger_wsgi.py#L11-L15), environment variables (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`) are explicitly set to `"1"` before importing OpenCV or NumPy. This prevents CPU thread oversubscription and deadlocks when running NumPy/OpenCV inside WSGI web server workers and Python thread pools.
* **Model choice**: every step uses **Gemini 2.5 Flash** with thinking off and temperature 0 (`services/gemini.py`). Specs used Gemini 2.5 Pro until 2026-10-09; on the test listing Flash read the same make, model and engine in 3–11 s instead of ~28 s. To switch back, set `SPEC_MODEL` in `services/specs.py`.
* **The add-listing form decides what to trust**: fields at 0.69 confidence or more are filled in, and those under 0.85 are marked "please check". The year is always marked "please check", whatever its confidence: a model looks the same across the years it was made, so the photos can't settle it (`ALWAYS_CONFIRM` in `kemotives-nextjs/src/components/forms/addProduct.jsx`).
* **Unified SDK Migration**: The codebase uses the modern, unified `google-genai` SDK (`from google import genai`), connecting directly to Google Cloud Vertex AI backend (`vertexai=True`, `location='us-central1'`).

---

## 4. Tests and CI

The suite in [`tests/`](./tests) runs offline: every Google client is replaced by a fake, so it needs no credentials and costs nothing. It includes regression tests for the 2026-10-09 production bugs (OpenCV's thread per core, the ASCII log stream, the shared Gemini client).

```bash
pip install -r requirements-dev.txt
pytest                                      # tests/ only; the test_*.py files in the root are manual scripts
ruff check --select E9,F63,F7,F82 .         # syntax errors and undefined names
```

[`.github/workflows/ci.yml`](./.github/workflows/ci.yml) runs the same checks on Python 3.13 (the server's version) for every push and pull request to `main`. It also fails if a credential file or key is committed.

---

## 5. Which code is the server running?

Deploys are a manual `git pull` on the server (see `vision_service_production_fix.md` §7 in the docs repo), so it can fall behind `main`. `GET /health` reports the commit the running app was deployed at, read from the checkout's `.git` when the app starts:

```bash
curl -s https://vision.kemotives.co.ke/health    # {"status": "ok", "commit": "6a68446"}
git log -1 --format=%h                           # in this repo: the same commit means the server is up to date
```

`"commit": null` means the app couldn't read `.git` (for example, files copied in by hand rather than pulled). The startup line in `~/vision/wsgi_error.log` shows the commit too. The app picks up a new commit only when it restarts, so check after the restart step of the deploy.
