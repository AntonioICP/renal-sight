# ------------------------ IMPORTS ------------------------
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow import keras

from renal_sight.ml_logic.predict import predict_with_confidence
from renal_sight.ml_logic.report import generate_report

# ---------------------- LOAD MODEL -----------------------
# Load model once at startup — not on every request
MODEL_PATH = os.environ.get("MODEL_PATH", "models/efficientnet_best.keras")
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global model
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    yield
    print("Shutting down...")

# ---------------------- APP SETUP -----------------------
app = FastAPI(
    title="Renal Sight API",
    description="AI-assisted kidney stone detection and clinical report generation from CT imaging",
    version="0.1.0",
    lifespan=lifespan
)

# Allow all origins for development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- ENDPOINTS -----------------------

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status"  : "ok",
        "message" : "Renal Sight API is running"
    }

@app.get("/health")
def health():
    """Check if model is loaded and ready."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status" : "ok",
        "model"  : MODEL_PATH
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a CT scan image and get:
    - prediction (Stone / Non-Stone)
    - confidence score
    - clinical report
    """
    # File Type Validation
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image."
        )

    # Read image bytes
    image_bytes = await file.read()

    # Prediction
    try:
        label, confidence = predict_with_confidence(model, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Report
    try:
        report = generate_report(image_bytes, label, confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

    # Results
    return {
        "prediction" : label,
        "confidence" : confidence,
        "report"     : report
    }
