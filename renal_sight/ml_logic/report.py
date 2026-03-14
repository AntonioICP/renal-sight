import io
import base64
import ollama
from PIL import Image
import tensorflow as tf
import numpy as np

def generate_report(image_input, prediction_label: str, confidence_score: float):
    # Convert to PIL for report generation
    if isinstance(image_input, bytes):
        img_pil = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, tf.Tensor):
        img_pil = Image.fromarray(
            image_input.numpy().squeeze().astype("uint8")
        ).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img_pil = Image.fromarray(
            image_input.squeeze().astype("uint8")
        ).convert("RGB")
    else:
        img_pil = image_input  # already PIL

    # Convert PIL image to base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = f"""You are a clinical radiology assistant specialized in kidney stone detection.

    An AI model has analyzed this CT scan and detected the following:
    - Detection result : {prediction_label}
    - Model confidence : {confidence_score}%

    Based on the image and the AI result above, generate a concise professional clinical report.
    Structure it as:

    FINDINGS:
    IMPRESSION:
    RECOMMENDATION:

    Rules:
    - Do NOT invent details not provided.
    - Do NOT contradict the AI detection result
    - Keep it under 200 words
    - Use formal medical language
    """

    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_base64]
        }]
    )

    return response["message"]["content"]
