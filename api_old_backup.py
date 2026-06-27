import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

SMOOTH = 1e-6

def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1.0 - (
        (2. * intersection + SMOOTH) /
        (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + SMOOTH)
    )

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (
        (2. * intersection + SMOOTH) /
        (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + SMOOTH)
    )

def iou_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)

def precision_m(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred > 0.5, tf.float32))
    tp = tf.keras.backend.sum(y_true_f * y_pred_f)
    fp = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
    return (tp + SMOOTH) / (tp + fp + SMOOTH)

def recall_m(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred > 0.5, tf.float32))
    tp = tf.keras.backend.sum(y_true_f * y_pred_f)
    fn = tf.keras.backend.sum(y_true_f * (1 - y_pred_f))
    return (tp + SMOOTH) / (tp + fn + SMOOTH)

def focal_dice_loss(y_true, y_pred, gamma=2.0):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = tf.keras.backend.exp(-bce)
    focal = tf.keras.backend.mean((1 - bce_exp) ** gamma * bce)
    return focal + dice_loss(y_true, y_pred)

CUSTOM_OBJECTS = {
    'focal_dice_loss': focal_dice_loss,
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'dice_coef': dice_coef,
    'iou_coef': iou_coef,
    'precision_m': precision_m,
    'recall_m': recall_m,
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_unet.keras')

app = FastAPI(title="BreastCare AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def load_ml_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    print(f"Model loaded from {MODEL_PATH}")

IMAGE_SHAPE = (256, 256)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, IMAGE_SHAPE)
    img = img.astype(np.float32) / 255.0
    return img

def mask_to_base64(mask: np.ndarray) -> str:
    mask_uint8 = (mask * 255).astype(np.uint8)
    pil_img = Image.fromarray(mask_uint8, mode='L')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def classify_by_mask(mask: np.ndarray) -> tuple:
    lesion_ratio = float(mask.mean())
    if lesion_ratio < 0.01:
        return "Normal", 1.0 - lesion_ratio
    elif lesion_ratio < 0.15:
        return "Jinak", min(lesion_ratio * 5, 0.95)
    else:
        return "Ganas", min(lesion_ratio * 3, 0.98)

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    lesion_ratio: float
    mask_base64: str
    all_results: list

@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    img = preprocess_image(image_bytes)

    input_tensor = np.expand_dims(img, axis=0)
    pred_mask = model.predict(input_tensor, verbose=0)[0, :, :, 0]
    binary_mask = (pred_mask > 0.5).astype(np.float32)

    prediction, confidence = classify_by_mask(binary_mask)
    lesion_ratio = float(binary_mask.mean())

    labels = ["Normal", "Jinak", "Ganas"]
    confidences = []
    for label in labels:
        if label == prediction:
            confidences.append(round(confidence, 4))
        elif label == "Normal":
            confidences.append(round(max(0.01, 1.0 - confidence - 0.1), 4))
        else:
            confidences.append(round(max(0.01, (1.0 - confidence) * 0.5), 4))

    total = sum(confidences)
    confidences = [round(c / total, 4) for c in confidences]

    all_results = [
        {"label": labels[i], "confidence": confidences[i]}
        for i in range(3)
    ]

    mask_b64 = mask_to_base64(binary_mask)

    return PredictResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        lesion_ratio=round(lesion_ratio, 4),
        mask_base64=mask_b64,
        all_results=all_results,
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
