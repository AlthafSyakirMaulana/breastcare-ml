"""BERSERI FastAPI backend: classifier + U-Net lesion segmentation.

This API combines two independently trained models:
1. ResNet101 classifier (PyTorch): Normal / Benign / Malignant.
2. U-Net segmenter (TensorFlow/Keras): predicted lesion mask and overlay.

The classifier produces the diagnosis label. The U-Net only localizes a
suspected lesion area; it does not decide Normal / Benign / Malignant.

Expected project structure:
    breastcare-ml/
    ├── api.py
    ├── models/
    │   ├── breast_classifier_approach1_best.pt
    │   └── unet_unbalanced_best.keras
    └── ...

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000

Environment variables:
    CLASSIFIER_MODEL_PATH  Path to classifier .pt checkpoint.
    UNET_MODEL_PATH        Path to U-Net .keras model.
    ALLOWED_ORIGINS        Comma-separated frontend origins.
    MASK_THRESHOLD         U-Net mask threshold (default 0.5).
    MAX_UPLOAD_MB          Maximum uploaded image size in MB (default 10).
"""

from __future__ import annotations

import base64
import io
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal

import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision import models, transforms


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

UNET_INPUT_SIZE = (256, 256)  # (width, height)
DEFAULT_CLASSIFIER_PATH = MODEL_DIR / "breast_classifier_approach1_best.pt"
DEFAULT_UNET_PATH = MODEL_DIR / "unet_unbalanced_best.keras"

CLASSIFIER_MODEL_PATH = Path(
    os.getenv("CLASSIFIER_MODEL_PATH", str(DEFAULT_CLASSIFIER_PATH))
)
UNET_MODEL_PATH = Path(os.getenv("UNET_MODEL_PATH", str(DEFAULT_UNET_PATH)))
MASK_THRESHOLD = float(os.getenv("MASK_THRESHOLD", "0.5"))
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

if not 0.0 <= MASK_THRESHOLD <= 1.0:
    raise ValueError("MASK_THRESHOLD must be between 0 and 1.")

ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if item.strip()
]

DISPLAY_LABELS = {
    "normal": "Normal",
    "benign": "Jinak",
    "malignant": "Ganas",
}
DISPLAY_ORDER = ["normal", "benign", "malignant"]

# Models are loaded once at startup.
classifier_model: nn.Module | None = None
unet_model: tf.keras.Model | None = None
classifier_class_names: List[str] = []
classifier_transform: transforms.Compose | None = None
classifier_device: torch.device | None = None
inference_lock = threading.Lock()


# -----------------------------------------------------------------------------
# API schemas
# -----------------------------------------------------------------------------
class ClassProbability(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    raw_label: Literal["normal", "benign", "malignant"]
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: List[ClassProbability]
    model: str = "ResNet101 classifier (Approach 1)"


class SegmentationResult(BaseModel):
    model: str = "U-Net unbalanced"
    mask_threshold: float = Field(..., ge=0.0, le=1.0)
    mask_area_ratio: float = Field(..., ge=0.0, le=1.0)
    mean_mask_probability: float = Field(..., ge=0.0, le=1.0)
    positive_pixel_count: int = Field(..., ge=0)
    output_width: int = Field(..., ge=1)
    output_height: int = Field(..., ge=1)


class PredictResponse(BaseModel):
    # Backward-compatible fields for the existing frontend.
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    lesion_ratio: float = Field(..., ge=0.0, le=1.0)
    all_results: List[ClassProbability]
    mask_base64: str
    mask_data_url: str
    overlay_base64: str
    overlay_data_url: str

    # Clear structured outputs for a newer frontend.
    classification: ClassificationResult
    segmentation: SegmentationResult
    note: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    classifier_loaded: bool
    unet_loaded: bool
    classifier_model_path: str
    unet_model_path: str
    device: str


# -----------------------------------------------------------------------------
# FastAPI setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="BERSERI Breast Care AI API",
    version="2.0.0",
    description=(
        "Breast ultrasound classification (Normal/Jinak/Ganas) plus "
        "U-Net predicted lesion-area segmentation. Outputs are screening support, "
        "not a definitive clinical diagnosis."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def get_torch_device() -> torch.device:
    """Choose CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_classifier(num_classes: int) -> nn.Module:
    """Rebuild the exact ResNet101 head used by code_classifier.py.

    weights=None avoids downloading ImageNet weights at API startup. The trained
    checkpoint contains all final weights.
    """
    network = models.resnet101(weights=None)
    network.fc = nn.Linear(network.fc.in_features, num_classes)
    return network


def load_classifier_checkpoint() -> None:
    """Load the ResNet101 Approach-1 classifier and matching preprocessing."""
    global classifier_model, classifier_class_names, classifier_transform, classifier_device

    if not CLASSIFIER_MODEL_PATH.is_file():
        raise RuntimeError(
            f"Classifier checkpoint not found: {CLASSIFIER_MODEL_PATH}. "
            "Run code_classifier.py first or set CLASSIFIER_MODEL_PATH."
        )

    classifier_device = get_torch_device()
    try:
        checkpoint: Dict[str, Any] = torch.load(
            CLASSIFIER_MODEL_PATH,
            map_location=classifier_device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not support weights_only.
        checkpoint = torch.load(CLASSIFIER_MODEL_PATH, map_location=classifier_device)

    required = {"state_dict", "class_names", "image_size", "normalization"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(
            "Classifier checkpoint has an unexpected format. Missing keys: "
            f"{sorted(missing)}"
        )

    class_names = [str(name).lower() for name in checkpoint["class_names"]]
    supported = set(DISPLAY_LABELS)
    if set(class_names) != supported:
        raise RuntimeError(
            "Classifier classes must contain exactly normal, benign, and malignant. "
            f"Found: {class_names}"
        )

    image_size = int(checkpoint["image_size"])
    normalization = checkpoint["normalization"]
    mean = normalization["mean"]
    std = normalization["std"]

    network = build_classifier(num_classes=len(class_names)).to(classifier_device)
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    network.eval()

    # This exactly matches the evaluation transform in code_classifier.py.
    classifier_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    classifier_model = network
    classifier_class_names = class_names

    print(
        "Loaded classifier from "
        f"{CLASSIFIER_MODEL_PATH} on {classifier_device.type}; classes={class_names}"
    )


def load_unet_checkpoint() -> None:
    """Load the U-Net only for inference."""
    global unet_model

    if not UNET_MODEL_PATH.is_file():
        raise RuntimeError(
            f"U-Net model not found: {UNET_MODEL_PATH}. "
            "Run code_ml.py first or set UNET_MODEL_PATH."
        )

    # compile=False skips training-only losses and metrics at deployment.
    network = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
    input_shape = tuple(network.input_shape[1:])
    expected_shape = (UNET_INPUT_SIZE[1], UNET_INPUT_SIZE[0], 3)
    if input_shape != expected_shape:
        raise RuntimeError(
            f"Unexpected U-Net input shape {input_shape}; expected {expected_shape}."
        )

    unet_model = network
    print(f"Loaded U-Net segmentation model from {UNET_MODEL_PATH}")


@app.on_event("startup")
def startup_load_models() -> None:
    """Load both deployed models once when the server starts."""
    load_classifier_checkpoint()
    load_unet_checkpoint()


# -----------------------------------------------------------------------------
# Image processing helpers
# -----------------------------------------------------------------------------
def encode_png(image: np.ndarray, mode: str) -> str:
    """Encode a grayscale or RGB NumPy array as raw Base64 PNG."""
    buffer = io.BytesIO()
    Image.fromarray(image, mode=mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def read_uploaded_image(image_bytes: bytes) -> Image.Image:
    """Read one uploaded image safely, correct EXIF orientation, and convert to RGB."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file is not a valid image.",
        ) from exc


def prepare_unet_input(image: Image.Image) -> np.ndarray:
    """Match U-Net training preprocessing: RGB, 256×256, float32 in [0, 1]."""
    rgb = np.asarray(image, dtype=np.uint8)
    resized = cv2.resize(rgb, UNET_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


def predict_classifier(image: Image.Image) -> tuple[str, float, List[ClassProbability]]:
    """Return class label, top probability, and probabilities for all three classes."""
    if classifier_model is None or classifier_transform is None or classifier_device is None:
        raise HTTPException(status_code=503, detail="Classifier model is not loaded yet.")

    tensor = classifier_transform(image).unsqueeze(0).to(classifier_device)
    with inference_lock, torch.no_grad():
        logits = classifier_model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    probability_map = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(classifier_class_names)
    }
    best_raw_label = max(probability_map, key=probability_map.get)

    ordered_results = [
        ClassProbability(
            label=DISPLAY_LABELS[class_name],
            confidence=round(float(probability_map[class_name]), 6),
        )
        for class_name in DISPLAY_ORDER
    ]
    return best_raw_label, float(probability_map[best_raw_label]), ordered_results


def predict_unet_mask(image: Image.Image) -> np.ndarray:
    """Return the U-Net lesion probability mask at 256×256."""
    if unet_model is None:
        raise HTTPException(status_code=503, detail="U-Net model is not loaded yet.")

    model_input = prepare_unet_input(image)
    batch = np.expand_dims(model_input, axis=0)
    with inference_lock:
        prediction = unet_model.predict(batch, verbose=0)

    return np.clip(prediction[0, :, :, 0], 0.0, 1.0).astype(np.float32)


def create_overlay(original_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Create a semi-transparent red lesion overlay on the original image."""
    overlay = original_rgb.astype(np.float32).copy()
    lesion_pixels = binary_mask.astype(bool)
    red = np.array([255, 0, 0], dtype=np.float32)
    overlay[lesion_pixels] = 0.65 * overlay[lesion_pixels] + 0.35 * red
    return np.clip(overlay, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {
        "service": "BERSERI Breast Care AI API",
        "classifier": "ResNet101: Normal / Jinak / Ganas",
        "segmenter": "U-Net: predicted lesion-area mask and overlay",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        classifier_loaded=classifier_model is not None,
        unet_loaded=unet_model is not None,
        classifier_model_path=str(CLASSIFIER_MODEL_PATH),
        unet_model_path=str(UNET_MODEL_PATH),
        device=classifier_device.type if classifier_device else "unknown",
    )


@app.get("/model-info")
def model_info() -> dict:
    return {
        "classification": {
            "model": "ResNet101 classifier (Approach 1)",
            "input": "original ultrasound image only; no ground-truth mask overlay",
            "classes": [DISPLAY_LABELS[name] for name in DISPLAY_ORDER],
        },
        "segmentation": {
            "model": "U-Net unbalanced",
            "input_shape": [256, 256, 3],
            "output": "one probability mask: background vs predicted lesion region",
            "mask_threshold": MASK_THRESHOLD,
        },
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Classify one ultrasound image and return a U-Net mask plus overlay."""
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type and file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG, and WebP image uploads are supported.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        maximum_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum upload size is {maximum_mb} MB.",
        )

    image = read_uploaded_image(image_bytes)
    source_rgb = np.asarray(image, dtype=np.uint8)

    # Diagnosis comes from the classifier, not from lesion-mask size.
    raw_label, confidence, all_results = predict_classifier(image)

    # U-Net gives lesion localization only.
    probability_mask = predict_unet_mask(image)
    binary_mask_256 = (probability_mask >= MASK_THRESHOLD).astype(np.uint8)

    source_height, source_width = source_rgb.shape[:2]
    binary_mask_original = cv2.resize(
        binary_mask_256,
        (source_width, source_height),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_png = (binary_mask_original * 255).astype(np.uint8)
    overlay_rgb = create_overlay(source_rgb, binary_mask_original)

    mask_base64 = encode_png(mask_png, mode="L")
    overlay_base64 = encode_png(overlay_rgb, mode="RGB")
    positive_values = probability_mask[binary_mask_256 == 1]

    lesion_ratio = float(binary_mask_256.mean())
    mean_probability = float(
        positive_values.mean() if positive_values.size else probability_mask.mean()
    )

    classification = ClassificationResult(
        raw_label=raw_label,  # type: ignore[arg-type]
        label=DISPLAY_LABELS[raw_label],
        confidence=round(confidence, 6),
        probabilities=all_results,
    )
    segmentation = SegmentationResult(
        mask_threshold=MASK_THRESHOLD,
        mask_area_ratio=round(lesion_ratio, 6),
        mean_mask_probability=round(mean_probability, 6),
        positive_pixel_count=int(binary_mask_256.sum()),
        output_width=source_width,
        output_height=source_height,
    )

    return PredictResponse(
        prediction=classification.label,
        confidence=classification.confidence,
        lesion_ratio=segmentation.mask_area_ratio,
        all_results=all_results,
        mask_base64=mask_base64,
        mask_data_url=f"data:image/png;base64,{mask_base64}",
        overlay_base64=overlay_base64,
        overlay_data_url=f"data:image/png;base64,{overlay_base64}",
        classification=classification,
        segmentation=segmentation,
        note=(
            "The Normal/Jinak/Ganas label is generated by the classifier. "
            "The red overlay is a predicted lesion-area segmentation from U-Net. "
            "This application is screening support and not a definitive clinical diagnosis."
        ),
    )
