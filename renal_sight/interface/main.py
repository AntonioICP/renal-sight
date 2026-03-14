# ------------------------ IMPORTS ------------------------
import os
from tensorflow import keras
from PIL import Image
import numpy as np
import tensorflow as tf

from renal_sight.ml_logic.preprocessing import preprocess_image_for_backbone
from renal_sight.ml_logic.predict import predict_with_confidence
from renal_sight.ml_logic.report import generate_report

# --------------------- CONFIGURATION ---------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/efficientnet_best.keras")

# ---------------------- LOAD MODEL -----------------------
def get_model():
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    return model

# ----------------------- PREDICTION ----------------------
def predict(image_input):
    model = get_model()

    label, confidence = predict_with_confidence(model, image_input)
    report = generate_report(image_input, label, confidence)

    return {
        "prediction"  : label,
        "confidence"  : confidence,
        "report"      : report
    }


# ----------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m renal_sight.interface.main <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = predict(image_bytes)

    print(f"\\nPrediction  : {result['prediction']}")
    print(f"Confidence  : {result['confidence']}%")
    print(f"\\n--- CLINICAL REPORT ---\\n")
    print(result['report'])
