# ------------------------ IMPORTS ------------------------
import numpy as np
import tensorflow as tf
from PIL import Image
from renal_sight.ml_logic.preprocessing import preprocess_image_for_backbone

# --------------------- CONFIGURATION ---------------------
LABEL_NAMES = {0: "Non-Stone", 1: "Stone"}
THRESHOLD   = 0.5

# ------------------ PREDICT FROM TENSOR ------------------
def predict_with_confidence(model, imgage_input):
    if isinstance(imgage_input, bytes):
        img_tensor = preprocess_image_for_backbone(imgage_input, backbone="efficientnet")
    elif isinstance(imgage_input, Image.Image):
        img_array = np.array(imgage_input.resize((224, 224))).astype("float32")
        img_tensor = tf.expand_dims(img_array, axis=0)
    else:
        img_tensor = imgage_input

    prob = model.predict(img_tensor, verbose=0)
    prob = float(prob.flatten()[0])

    if prob >= THRESHOLD:
        label = "Stone"
        confidence = prob
    else:
        label = "Non-Stone"
        confidence = 1 - prob

    return label, round(confidence * 100, 2)
