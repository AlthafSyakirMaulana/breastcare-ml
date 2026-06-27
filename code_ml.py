#!/usr/bin/env python3
"""Unbalanced U-Net training for BUSI breast-ultrasound image segmentation.

Key differences from the balanced version:
- No malignant oversampling or class balancing.
- No EarlyStopping: training completes all configured epochs.
- Normal images are included as empty-mask examples.
- ModelCheckpoint still saves the best validation Dice checkpoint.
- Validation and test sets are never used during training.

Outputs:
    models/unet_unbalanced_best.keras   # deploy this checkpoint
    models/unet_unbalanced_final.keras  # final epoch checkpoint
    outputs_unet/training_history.csv
    outputs_unet/test_metrics.csv
"""

import os, re, time, warnings, glob
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import map_coordinates, gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.layers import (
    Conv2D, Input, MaxPool2D,
    Conv2DTranspose, concatenate,
    Dropout, BatchNormalization, Cropping2D
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'outputs_unet')
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SHAPE = (256, 256)
SMOOTH = 1e-6

# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def find_dataset_root():
    env_path = os.getenv('BUSI_DATASET_DIR')
    candidates = [env_path] if env_path else []
    candidates += [
        '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT',
        '/kaggle/input/breast-ultrasound-images-dataset',
        '/kaggle/input/busi-dataset/Dataset_BUSI_with_GT',
        '/kaggle/input/busi-dataset',
        '/app/data/raw/Dataset_BUSI_with_GT',
        '/app/data/raw',
    ]
    for pattern in ['/kaggle/input/*/Dataset_BUSI_with_GT', '/kaggle/input/*/*']:
        candidates += glob.glob(pattern)
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, 'benign')):
            return path
    for root, dirs, _ in os.walk('/kaggle/input'):
        if 'benign' in dirs:
            return root
    if os.path.isdir('/app/data'):
        for root, dirs, _ in os.walk('/app/data'):
            if 'benign' in dirs:
                return root
    return None

img_dir = find_dataset_root()
if img_dir is None:
    raise FileNotFoundError("BUSI dataset not found. Set BUSI_DATASET_DIR.")

print(f"Dataset: {img_dir}")

# ---------------------------------------------------------------------------
# Load file paths (exclude mask files from image list)
# ---------------------------------------------------------------------------
def load_raw_paths(image_dir):
    img_paths, mask_paths, labels = [], [], []
    for cls in ['benign', 'malignant', 'normal']:
        cls_dir = os.path.join(image_dir, cls)
        for f in sorted(os.listdir(cls_dir)):
            if 'mask' in f:
                continue
            full_img = os.path.join(cls_dir, f)
            full_mask = full_img.replace('.png', '_mask.png')
            if not os.path.exists(full_img):
                continue
            if not os.path.exists(full_mask):
                full_mask = None
            img_paths.append(full_img)
            mask_paths.append(full_mask)
            labels.append(cls)
    labels = np.array(labels)
    print(f"Total samples: {len(img_paths)}")
    print(f"Distribution: { {k: list(labels).count(k) for k in sorted(set(labels))} }")
    return np.array(img_paths), np.array(mask_paths), labels

raw_img_paths, raw_mask_paths, raw_labels = load_raw_paths(img_dir)

# ---------------------------------------------------------------------------
# Stratified 80/10/10 split
# ---------------------------------------------------------------------------
paths_train, paths_temp, masks_train, masks_temp, lbl_train, lbl_temp = train_test_split(
    raw_img_paths, raw_mask_paths, raw_labels,
    test_size=0.20, random_state=42, stratify=raw_labels
)
paths_val, paths_test, masks_val, masks_test, lbl_val, lbl_test = train_test_split(
    paths_temp, masks_temp, lbl_temp,
    test_size=0.50, random_state=42, stratify=lbl_temp
)

total = len(raw_img_paths)
print(f"\nTrain: {len(paths_train)} ({len(paths_train)/total*100:.1f}%)")
print(f"Val:   {len(paths_val)} ({len(paths_val)/total*100:.1f}%)")
print(f"Test:  {len(paths_test)} ({len(paths_test)/total*100:.1f}%)")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_split(img_paths, mask_paths, labels, image_shape=IMAGE_SHAPE, split_name='Split'):
    images_list, masks_list, labels_out = [], [], []
    for img_path, mask_path, lbl in zip(img_paths, mask_paths, labels):
        try:
            img = plt.imread(img_path)
        except Exception:
            continue
        img = cv2.resize(img, image_shape)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        img = img.astype(np.float32)
        if img.max() > 1:
            img = img / 255.0
        if mask_path is None:
            mask = np.zeros(image_shape, dtype=np.float32)
        else:
            try:
                mask = plt.imread(mask_path)
            except Exception:
                continue
            mask = cv2.resize(mask, image_shape, interpolation=cv2.INTER_NEAREST)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        images_list.append(img)
        masks_list.append(mask)
        labels_out.append(lbl)
    images_array = np.array(images_list, dtype=np.float32)
    masks_array = np.array(masks_list, dtype=np.float32)
    labels_out = np.array(labels_out)
    print(f"[{split_name}] After preprocessing: {len(images_array)} samples")
    return images_array, masks_array, labels_out

X_train, y_train, lbl_train = preprocess_split(paths_train, masks_train, lbl_train, split_name='Train')
X_val, y_val, lbl_val = preprocess_split(paths_val, masks_val, lbl_val, split_name='Val')
X_test, y_test, lbl_test = preprocess_split(paths_test, masks_test, lbl_test, split_name='Test')

# ---------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------
def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1.0 - ((2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH))

def focal_dice_loss(y_true, y_pred, gamma=2.0):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = K.exp(-bce)
    focal = K.mean((1 - bce_exp) ** gamma * bce)
    return focal + dice_loss(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def iou_coef(y_true, y_pred):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)

def precision_m(y_true, y_pred):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred > 0.5, tf.float32))
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    return (tp + SMOOTH) / (tp + fp + SMOOTH)

def recall_m(y_true, y_pred):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred > 0.5, tf.float32))
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    return (tp + SMOOTH) / (tp + fn + SMOOTH)

METRICS = [dice_coef, iou_coef, precision_m, recall_m, 'accuracy']

# ---------------------------------------------------------------------------
# U-Net architecture
# ---------------------------------------------------------------------------
def conv_block(x, filters, kernel_size=(3, 3), use_batch_norm=True, dropout=0.0):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal',
               activation='relu', kernel_regularizer=l2(1e-4))(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal',
               activation='relu', kernel_regularizer=l2(1e-4))(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    return x

def crop_concat(upsampled, skip):
    up_h, up_w = K.int_shape(upsampled)[1:3]
    sk_h, sk_w = K.int_shape(skip)[1:3]
    dh, dw = sk_h - up_h, sk_w - up_w
    if dh != 0 or dw != 0:
        skip = Cropping2D(((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2)))(skip)
    return concatenate([upsampled, skip])

def build_unet(input_shape, num_filters=64, dropout=0.3, use_bn=True):
    inputs = Input(input_shape)
    c1 = conv_block(inputs, num_filters, use_batch_norm=use_bn, dropout=0.0)
    p1 = MaxPool2D((2, 2))(c1)
    c2 = conv_block(p1, num_filters * 2, use_batch_norm=use_bn, dropout=0.0)
    p2 = MaxPool2D((2, 2))(c2)
    c3 = conv_block(p2, num_filters * 4, use_batch_norm=use_bn, dropout=0.1)
    p3 = MaxPool2D((2, 2))(c3)
    c4 = conv_block(p3, num_filters * 8, use_batch_norm=use_bn, dropout=0.2)
    p4 = MaxPool2D((2, 2))(c4)
    c5 = conv_block(p4, num_filters * 16, use_batch_norm=use_bn, dropout=0.3)
    u6 = Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = crop_concat(u6, c4)
    c6 = conv_block(u6, num_filters * 8, use_batch_norm=use_bn, dropout=0.2)
    u7 = Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = crop_concat(u7, c3)
    c7 = conv_block(u7, num_filters * 4, use_batch_norm=use_bn, dropout=0.1)
    u8 = Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = crop_concat(u8, c2)
    c8 = conv_block(u8, num_filters * 2, use_batch_norm=use_bn, dropout=0.0)
    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = crop_concat(u9, c1)
    c9 = conv_block(u9, num_filters, use_batch_norm=use_bn, dropout=0.0)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return Model(inputs, outputs, name='U-Net')

model = build_unet((256, 256, 3), num_filters=64, dropout=0.3)
model.summary()

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
EPOCHS = int(os.getenv('EPOCHS', '120'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss=focal_dice_loss,
    metrics=METRICS
)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'unet_unbalanced_best.keras')
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'unet_unbalanced_final.keras')

callbacks = [
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_dice_coef', mode='max',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_dice_coef', mode='max', patience=8,
                      factor=0.5, min_lr=1e-7, verbose=1),
]

# ---------------------------------------------------------------------------
# Training (no EarlyStopping — completes all epochs)
# ---------------------------------------------------------------------------
t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)
train_time = time.time() - t0
print(f"\nTraining time: {train_time/60:.2f} minutes")

# Save final model
model.save(FINAL_MODEL_PATH)
print(f"Final model saved: {FINAL_MODEL_PATH}")

# ---------------------------------------------------------------------------
# Evaluate best model on test set
# ---------------------------------------------------------------------------
best_model = load_model(BEST_MODEL_PATH, custom_objects={
    'focal_dice_loss': focal_dice_loss, 'dice_loss': dice_loss,
    'dice_coef': dice_coef, 'iou_coef': iou_coef,
    'precision_m': precision_m, 'recall_m': recall_m,
})
results = best_model.evaluate(X_test, y_test, verbose=0)
metric_names = ['Loss', 'Dice', 'IoU', 'Precision', 'Recall', 'Accuracy']
print("\n" + "=" * 55)
print("U-NET TEST SET RESULTS")
print("=" * 55)
for name, value in zip(metric_names, results):
    print(f"{name:<12}: {value:.4f}")
print("=" * 55)

# Save training history
import pandas as pd
pd.DataFrame(history.history).to_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False)
print(f"\nHistory saved to {OUTPUT_DIR}/training_history.csv")
print(f"Best model: {BEST_MODEL_PATH}")
