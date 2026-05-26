# Vision Service: Running & Testing Guide

This guide provides a step-by-step procedure to run and test the Vehicle Vision Microservice, which handles AI-powered spec extraction, license plate recognition, and duplicate detection.

---

## 1. Prerequisites

Before running the service, ensure the following are installed and configured:

*   **Python 3.10+**: Make sure Python is in your system PATH.
*   **Tesseract OCR**: 
    *   [Download Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
    *   Ensure the installation path (usually `C:\Program Files\Tesseract-OCR`) is added to your System Environment Variables (PATH).
*   **API Keys**:
    *   Open `vision-service/.env` and ensure `OPENAI_API_KEY` is valid.
    *   *Note: If the quota is exceeded, spec extraction will return null values but the service will still run.*

---

## 2. Running the Service

Open your terminal (e.g., Git Bash or PowerShell) and run the following:

1.  **Navigate to the directory**:
    ```bash
    cd vision-service
    ```

2.  **Start the server**:
    Activate the virtual environment and run the ASGI server using `uvicorn`:
    ```bash
    source ../.venv/Scripts/activate
    uvicorn main:app --port 8000 --reload
    ```
    The service is now running at `http://localhost:8000`.

---

## 3. Testing via the Frontend (Recommended)

The `addProduct.jsx` form is now integrated with this service. To test the full workflow:

1.  **Start the Client**: Ensure your React frontend is running (`npm run client`).
2.  **Open the Add Vehicle Page**: Navigate to the "Add New Vehicle" form in your browser.
3.  **Upload Images**:
    *   Select one or more clear photos of a vehicle.
    *   **Wait for AI Scan**: You will see a loader: *"AI is scanning vehicle details..."*
4.  **Verify Results**:
    *   **Autopopulate**: Once complete, fields like **Make, Model, Year, Engine, and Fuel Type** will be filled automatically.
    *   **Registration**: If the plate is clear, the **Prefix** and **Suffix** will be populated.
    *   **Duplicate Warning**: If the car is already in the database, a warning will appear.

---

## 4. Testing via Command Line (Script)

To verify the API independently of the frontend, use the provided test script:

1.  **Run the test**:
    Ensure your virtual environment is activated, then run the test script:
    ```bash
    python test_vision.py
    ```
    This script sends a dummy image to the service and prints the JSON response to your terminal.

---

## 5. Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **"Tesseract is not installed"** | Ensure Tesseract is installed and `tesseract.exe` is in your system PATH. Restart your terminal after adding it. |
| **"Insufficient Quota (429)"** | Your OpenAI API key has run out of credits. Spec extraction will fail, but the rest of the service (Duplicate check, Plate OCR) will work. |
| **Connection Refused** | Ensure the service is running on port `8000`. Check the uvicorn terminal for errors. |
| **Slow Analysis** | The AI analysis (OpenAI Vision) typically takes 5-10 seconds depending on network speed. |
