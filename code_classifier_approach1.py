#!/usr/bin/env python3
"""Train a three-class BUSI breast-ultrasound classifier (Approach 1).

This script intentionally uses ORIGINAL ultrasound images only.
It does NOT blend or concatenate ground-truth lesion masks with input images.

Classes:
    benign, malignant, normal

Outputs:
    models/breast_classifier_approach1_best.pt   # deploy this checkpoint
    models/breast_classifier_approach1_final.pt  # final epoch checkpoint
    outputs_classifier/splits.csv
    outputs_classifier/test_metrics.json
    outputs_classifier/test_classification_report.csv
    outputs_classifier/test_confusion_matrix.csv

The training loop has NO early stopping. It always completes EPOCHS epochs,
while separately saving the checkpoint with the best validation macro-F1.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = int(os.getenv("SEED", "42"))
IMAGE_SIZE = int(os.getenv("CLASSIFIER_IMAGE_SIZE", "224"))
BATCH_SIZE = int(os.getenv("CLASSIFIER_BATCH_SIZE", "8"))
EPOCHS = int(os.getenv("CLASSIFIER_EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("CLASSIFIER_LR", "5e-5"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

CLASS_NAMES = ["benign", "malignant", "normal"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs_classifier"
BEST_MODEL_PATH = MODEL_DIR / "breast_classifier_approach1_best.pt"
FINAL_MODEL_PATH = MODEL_DIR / "breast_classifier_approach1_final.pt"


def find_busi_root() -> Path:
    """Locate Dataset_BUSI_with_GT from an environment variable or common paths."""
    candidates: List[Path] = []

    env_path = os.getenv("BUSI_DATASET_DIR")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path("/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"),
            Path("/kaggle/input/breast-ultrasound-images-dataset"),
            BASE_DIR / "data" / "raw" / "Dataset_BUSI_with_GT",
            BASE_DIR / "data" / "raw",
        ]
    )

    for candidate in candidates:
        if (candidate / "benign").is_dir() and (candidate / "malignant").is_dir() and (candidate / "normal").is_dir():
            return candidate

    raise FileNotFoundError(
        "BUSI dataset not found. Set BUSI_DATASET_DIR to the folder containing "
        "the benign, malignant, and normal directories."
    )


def set_seed(seed: int) -> None:
    """Set random seeds for more reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_mask_filename(filename: str) -> bool:
    """Return True for BUSI lesion-mask files such as '_mask.png' or '_mask_1.png'."""
    stem = Path(filename).stem.lower()
    return "_mask" in stem


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    """Collect original ultrasound-image paths only; all ground-truth masks are excluded."""
    rows: List[Dict[str, str]] = []

    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for image_path in sorted(class_dir.glob("*.png")):
            if is_mask_filename(image_path.name):
                continue
            rows.append({"image_path": str(image_path), "label": class_name})

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No original PNG images were found in the BUSI dataset.")

    print("Dataset class counts:")
    print(manifest["label"].value_counts().reindex(CLASS_NAMES))
    return manifest


def split_manifest(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a stratified 80% train / 10% validation / 10% test split."""
    train_df, temp_df = train_test_split(
        manifest,
        test_size=0.20,
        random_state=SEED,
        stratify=manifest["label"],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    # Defensive checks: no image path may appear in more than one split.
    train_paths = set(train_df["image_path"])
    validation_paths = set(validation_df["image_path"])
    test_paths = set(test_df["image_path"])
    assert train_paths.isdisjoint(validation_paths)
    assert train_paths.isdisjoint(test_paths)
    assert validation_paths.isdisjoint(test_paths)

    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


class UltrasoundClassificationDataset(Dataset):
    """Dataset that returns an original ultrasound image and its class index."""

    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), CLASS_TO_INDEX[row["label"]]


def get_transforms() -> Dict[str, transforms.Compose]:
    """Return transforms; augmentation is applied to TRAIN images only."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        ),
        "evaluation": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        ),
    }


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(num_classes: int) -> nn.Module:
    """Create the ResNet101 classifier used by Approach 1."""
    weights = models.ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Train for one epoch when optimizer is supplied; otherwise evaluate."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    return avg_loss, accuracy, np.asarray(y_true), np.asarray(y_pred)


def save_checkpoint(path: Path, model: nn.Module, epoch: int, validation_macro_f1: float) -> None:
    """Save a portable state_dict checkpoint plus the preprocessing metadata."""
    payload = {
        "architecture": "resnet101",
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "epoch": epoch,
        "validation_macro_f1": validation_macro_f1,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def main() -> None:
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_root = find_busi_root()
    print(f"Using dataset: {dataset_root}")

    manifest = build_manifest(dataset_root)
    train_df, validation_df, test_df = split_manifest(manifest)

    split_frames = [
        train_df.assign(split="train"),
        validation_df.assign(split="validation"),
        test_df.assign(split="test"),
    ]
    pd.concat(split_frames, ignore_index=True).to_csv(OUTPUT_DIR / "splits.csv", index=False)

    for name, frame in [("Train", train_df), ("Validation", validation_df), ("Test", test_df)]:
        print(f"\n{name}: {len(frame)} images")
        print(frame["label"].value_counts().reindex(CLASS_NAMES))

    transform_map = get_transforms()
    datasets = {
        "train": UltrasoundClassificationDataset(train_df, transform_map["train"]),
        "validation": UltrasoundClassificationDataset(validation_df, transform_map["evaluation"]),
        "test": UltrasoundClassificationDataset(test_df, transform_map["evaluation"]),
    }

    device = get_device()
    pin_memory = device.type == "cuda"
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(name == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )
        for name, dataset in datasets.items()
    }

    print(f"\nDevice: {device}")
    model = make_model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_validation_f1 = -1.0
    history: List[Dict[str, float]] = []

    # No EarlyStopping: training completes every configured epoch.
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_accuracy, val_true, val_pred = run_epoch(model, loaders["validation"], criterion, device)
        val_macro_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_macro_f1,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_accuracy:.4f}, macro-F1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_validation_f1:
            best_validation_f1 = val_macro_f1
            save_checkpoint(BEST_MODEL_PATH, model, epoch, val_macro_f1)
            print(f"  Saved best classifier checkpoint: {BEST_MODEL_PATH.name}")

    save_checkpoint(FINAL_MODEL_PATH, model, EPOCHS, history[-1]["validation_macro_f1"])
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # Test only once after selecting the best checkpoint from validation data.
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    best_model = make_model(num_classes=len(checkpoint["class_names"])).to(device)
    best_model.load_state_dict(checkpoint["state_dict"])

    test_loss, test_accuracy, test_true, test_pred = run_epoch(best_model, loaders["test"], criterion, device)
    test_macro_f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)
    report = classification_report(
        test_true,
        test_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_true, test_pred, labels=list(range(len(CLASS_NAMES))))

    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / "test_classification_report.csv")
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")

    metrics = {
        "checkpoint": str(BEST_MODEL_PATH),
        "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
        "best_epoch": int(checkpoint["epoch"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "class_names": CLASS_NAMES,
        "split": "stratified 80/10/10 at the image-file level",
        "approach": "Approach 1: original ultrasound images only; no ground-truth mask overlay",
    }
    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n" + "=" * 70)
    print("CLASSIFIER TEST RESULTS")
    print("=" * 70)
    print(f"Best checkpoint epoch : {metrics['best_epoch']}")
    print(f"Test loss             : {test_loss:.4f}")
    print(f"Test accuracy         : {test_accuracy:.4f}")
    print(f"Test macro F1         : {test_macro_f1:.4f}")
    print(f"Deploy checkpoint     : {BEST_MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
