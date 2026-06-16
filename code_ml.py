#!/usr/bin/env python
# coding: utf-8
"""
Full U-Net training and evaluation pipeline for BUSI breast-ultrasound
image segmentation. This file was converted from the original notebook
and translated into English.

For website inference, import the lightweight helper from
``unet_web_inference.py`` instead of importing this training script.
"""

# # 🩺 Breast Cancer Image Segmentation — U-Net

# ## 1. Import Dependencies



import matplotlib
matplotlib.use('Agg')

import os, re, time, warnings, glob
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf

OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data augmentation support (elastic deformation)
from scipy.ndimage import map_coordinates, gaussian_filter

# Dataset splitting and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Layers used to build the U-Net model
from tensorflow.keras.layers import (
    Conv2D, Input, MaxPool2D,
    Conv2DTranspose, concatenate,
    Dropout, BatchNormalization, Cropping2D
)

# Build and load models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

# Training callbacks
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# Image data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Disable unnecessary warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Display the TensorFlow version
print("TensorFlow:", tf.__version__)


# ## 2. Locate the BUSI dataset on Kaggle



# Find the directory containing the BUSI dataset on Kaggle
def find_dataset_root():
    """Find the BUSI dataset root.

    Set the BUSI_DATASET_DIR environment variable when running outside Kaggle.
    """
    env_path = os.getenv('BUSI_DATASET_DIR')
    candidates = [env_path] if env_path else []
    candidates += [
        '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT',
        '/kaggle/input/breast-ultrasound-images-dataset',
        '/kaggle/input/busi-dataset/Dataset_BUSI_with_GT',
        '/kaggle/input/busi-dataset',
    ]

    # Search additional dataset directories on Kaggle
    for pattern in ['/kaggle/input/*/Dataset_BUSI_with_GT', '/kaggle/input/*/*']:
        candidates += glob.glob(pattern)

    # Check whether the directory contains the benign class
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, 'benign')):
            return path

    # Search the entire input directory if it has not been found
    for root, dirs, _ in os.walk('/kaggle/input'):
        if 'benign' in dirs:
            return root

    return None

# Find the dataset path
img_dir = find_dataset_root()

# Check whether the dataset exists
if img_dir is None:
    print("❌ Dataset not found!")
    raise FileNotFoundError(
        "BUSI dataset not found. Set BUSI_DATASET_DIR to the dataset root "
        "or attach the dataset in Kaggle."
    )
else:
    print(f"✅ Dataset found: {img_dir}")

    # Image class directories
    benign_path    = os.path.join(img_dir, 'benign')
    malignant_path = os.path.join(img_dir, 'malignant')
    normal_path    = os.path.join(img_dir, 'normal')

    # Read the image file lists
    benign_images    = os.listdir(benign_path)
    malignant_images = os.listdir(malignant_path)
    normal_images    = os.listdir(normal_path)

    # Combine all images
    images = benign_images + malignant_images + normal_images

    # Count images in each class
    print(f"Total files : {len(images)}")
    print(f"Benign      : {len(benign_images)}")
    print(f"Malignant   : {len(malignant_images)}")
    print(f"Normal      : {len(normal_images)}")


# ## 3. Randomly select three breast ultrasound images 



import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# --------------------------------------------------
# Find the mask corresponding to an image
# --------------------------------------------------
def find_mask(img_path):
    return img_path.replace(".png", "_mask.png")

# --------------------------------------------------
# Display images and masks
# --------------------------------------------------
def plot_samples(num_samples=3):

    fig, axes = plt.subplots(num_samples, 2,
                             figsize=(12, 5*num_samples))

    selected_idx = np.random.choice(
        len(images),
        num_samples,
        replace=False
    )

    for i, idx in enumerate(selected_idx):

        img_path = images[idx]

        initial = img_path.split('_')
        if len(initial) > 1:
            img_path = initial[0] + '.png'

        typ = img_path.split(' ')[0]

        full_img_path = os.path.join(
            img_dir,
            typ,
            img_path
        )

        full_mask_path = find_mask(full_img_path)

        img = Image.open(full_img_path)
        mask = Image.open(full_mask_path)

        axes[i, 0].imshow(ImageOps.invert(img))
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

# Display three samples
plot_samples(3)


# ## 4. Load Raw Data (Collect File Paths)



# Input image size for the model
IMAGE_SHAPE = (256, 256)


def load_raw_paths(image_dir, images):
    """
    Step 1 — Collect only file paths and class labels.
    Images are not loaded into RAM, resized, or normalized yet.
    Returns:
    - img_paths  : list of original image paths
    - mask_paths : list of corresponding mask paths
    - labels     : benign / malignant / normal labels
    """
    img_paths  = []
    mask_paths = []
    labels     = []

    for image in images:
        if 'mask' in image:
            continue

        typ = image.split(' ')[0]
        full_img  = os.path.join(image_dir, typ, image)
        full_mask = find_mask(full_img)

        # Skip the sample if the image or mask file does not exist
        if not os.path.exists(full_img) or not os.path.exists(full_mask):
            continue

        img_paths.append(full_img)
        mask_paths.append(full_mask)
        labels.append(typ)

    labels = np.array(labels)

    print(f"Total valid samples : {len(img_paths)}")
    print(f"Label distribution     : { {k: list(labels).count(k) for k in sorted(set(labels))} }")

    return img_paths, mask_paths, labels


# Collect raw file paths
raw_img_paths, raw_mask_paths, raw_labels = load_raw_paths(img_dir, images)


# ## 5. Split Data 80 / 10 / 10 (Using File Paths)



import numpy as np
from sklearn.model_selection import train_test_split

# Convert lists to arrays for easier indexing
raw_img_paths  = np.array(raw_img_paths)
raw_mask_paths = np.array(raw_mask_paths)

# Split into 80% training and 20% temporary data
# Stratification preserves the benign / malignant / normal proportions
(
    paths_train, paths_temp,
    masks_train, masks_temp,
    lbl_train,   lbl_temp
) = train_test_split(
    raw_img_paths, raw_mask_paths, raw_labels,
    test_size=0.20,
    random_state=42,
    stratify=raw_labels
)

# Split the temporary set into 10% validation and 10% test data
(
    paths_val,  paths_test,
    masks_val,  masks_test,
    lbl_val,    lbl_test
) = train_test_split(
    paths_temp, masks_temp, lbl_temp,
    test_size=0.50,
    random_state=42,
    stratify=lbl_temp
)

total = len(raw_img_paths)

print(f"Train : {len(paths_train):4d} samples ({len(paths_train)/total*100:.1f}%)")
print(f"Val   : {len(paths_val):4d} samples ({len(paths_val)/total*100:.1f}%)")
print(f"Test  : {len(paths_test):4d} samples ({len(paths_test)/total*100:.1f}%)")

print("\nLabel distribution for each split:")
for name, lbl in [('Train', lbl_train), ('Val', lbl_val), ('Test', lbl_test)]:
    counts = {k: list(lbl).count(k) for k in sorted(set(lbl))}
    print(f"  {name:6s}: {counts}")


# ## 6. Data Preprocessing (Applied Separately to Each Split)



import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_split(img_paths, mask_paths, labels,
                     image_shape=IMAGE_SHAPE,
                     fg_threshold=0.005,
                     split_name='Split'):
    """
    Step 2 — Read, resize, normalize, and filter images by foreground area.
    Apply preprocessing independently to each split (train / validation / test)
    to prevent data leakage.
    Returns:
    - images_array, masks_array, labels_out
    """
    images_list = []
    masks_list  = []
    labels_out  = []

    for img_path, mask_path, lbl in zip(img_paths, mask_paths, labels):
        try:
            img  = plt.imread(img_path)
            mask = plt.imread(mask_path)
        except Exception:
            continue

        # Resize
        img  = cv2.resize(img, image_shape)
        mask = cv2.resize(mask, image_shape,
                          interpolation=cv2.INTER_NEAREST)

        # Convert grayscale images to three channels
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        img = img.astype(np.float32)
        if img.max() > 1:
            img = img / 255.0

        # Normalize the mask
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        # Remove images whose lesion region is too small (<0.5%)
        if mask.mean() < fg_threshold:
            continue

        images_list.append(img)
        masks_list.append(mask)
        labels_out.append(lbl)

    images_array = np.array(images_list, dtype=np.float32)
    masks_array  = np.array(masks_list,  dtype=np.float32)
    labels_out   = np.array(labels_out)

    print(f"[{split_name}] After preprocessing and filtering: {len(images_array)} samples")
    print(f"  Image shape  : {images_array.shape}")
    print(f"  Mask shape : {masks_array.shape}")
    print(f"  Label distribution  : { {k: list(labels_out).count(k) for k in sorted(set(labels_out))} }")

    return images_array, masks_array, labels_out


# ── Preprocess each split separately ──
X_train, y_train, lbl_train = preprocess_split(
    paths_train, masks_train, lbl_train, split_name='Train')

X_val, y_val, lbl_val = preprocess_split(
    paths_val, masks_val, lbl_val, split_name='Val')

X_test, y_test, lbl_test = preprocess_split(
    paths_test, masks_test, lbl_test, split_name='Test')


# ## 6b. Check the Train / Validation / Test Distributions



# ── Check the lesion-area ratio distribution (foreground ratio) ──
# Purpose: verify that train, validation, and test data have similar distributions

def fg_stats(masks, name):
    # Calculate the foreground-pixel ratio (lesion area) for each mask
    r = masks.mean(axis=(1, 2, 3))

    # Summarize the foreground-ratio distribution
    print(f"{name:8s} | mean={r.mean():.4f}  std={r.std():.4f}  "
          f"min={r.min():.4f}  max={r.max():.4f}")

    return r

print("Lesion-pixel ratio (foreground ratio) per sample")
print("-" * 60)

# Calculate statistics for each data split
r_tr  = fg_stats(y_train, 'Train')
r_val = fg_stats(y_val,   'Val')
r_te  = fg_stats(y_test,  'Test')

# Plot distribution comparisons between splits
fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(r_tr,  bins=20, alpha=0.6, label='Train',
        color='steelblue', edgecolor='k')

ax.hist(r_val, bins=20, alpha=0.6, label='Val',
        color='darkorange', edgecolor='k')

ax.hist(r_te,  bins=20, alpha=0.6, label='Test',
        color='green', edgecolor='k')

ax.set_xlabel('Lesion-pixel ratio')
ax.set_ylabel('Number of images')
ax.set_title('Train / Validation / Test Distribution (Foreground Ratio)')
ax.legend()

plt.tight_layout()
plt.show()

# ── Kolmogorov–Smirnov statistical test ──
# Compare the training and test distributions
from scipy import stats

ks_stat, ks_p = stats.ks_2samp(r_tr, r_te)

print(f"\nKS-test Train vs Test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")

# Conclusion
if ks_p < 0.05:
    print("⚠️ p < 0.05 → the training and test distributions differ (this may bias the model)")
else:
    print("✅ p ≥ 0.05 → the training and test distributions are similar")


# ## 7. Data Augmentation



# ── Elastic Deformation (nonlinear image deformation) ──
# Increase data diversity by deforming images and masks together

def elastic_transform(image, mask, alpha=720, sigma=24, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    h, w = image.shape[:2]

    dx = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    indices = (np.clip(y + dy, 0, h - 1).flatten(),
               np.clip(x + dx, 0, w - 1).flatten())

    def warp_channel(ch):
        return map_coordinates(ch, indices, order=1, mode='reflect').reshape(h, w)

    if image.ndim == 3:
        warped_img = np.stack(
            [warp_channel(image[..., c]) for c in range(image.shape[-1])],
            axis=-1
        )
    else:
        warped_img = warp_channel(image)

    warped_mask = warp_channel(mask[..., 0])
    warped_mask = (warped_mask > 0.5).astype(np.float32)[..., np.newaxis]

    return warped_img.astype(np.float32), warped_mask


# ── Data augmentation (without oversampling) ──
# Purpose:
# - Increase the amount of training data
# - Reduce overfitting

def augment_batch(images, masks, elastic_prob=0.2):

    aug_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='reflect'
    )

    img_gen  = ImageDataGenerator(**aug_args)
    mask_gen = ImageDataGenerator(**aug_args)

    aug_images, aug_masks = [], []
    elas_images, elas_masks = [], []

    # ── Augment the original data ──
    for img, msk in zip(images, masks):

        seed_val = np.random.randint(0, 10000)

        aug_img = img_gen.random_transform(img, seed=seed_val)
        aug_msk = mask_gen.random_transform(msk, seed=seed_val)

        factor = np.random.uniform(0.75, 1.25)

        aug_img = np.clip(aug_img * factor, 0, 1).astype(np.float32)
        aug_msk = (aug_msk > 0.5).astype(np.float32)

        aug_images.append(aug_img)
        aug_masks.append(aug_msk)

        if np.random.rand() < elastic_prob:
            e_img, e_msk = elastic_transform(img, msk)
            elas_images.append(e_img)
            elas_masks.append(e_msk)

    aug_images = np.array(aug_images, dtype=np.float32)
    aug_masks  = np.array(aug_masks, dtype=np.float32)

    # ── Combine the data ──
    parts_img = [images, aug_images]
    parts_msk = [masks, aug_masks]

    if elas_images:
        parts_img.append(np.array(elas_images, dtype=np.float32))
        parts_msk.append(np.array(elas_masks, dtype=np.float32))

    X_out = np.concatenate(parts_img, axis=0)
    y_out = np.concatenate(parts_msk, axis=0)

    shuffle_idx = np.random.permutation(len(X_out))

    print(f"Original samples: {len(images)} | After augmentation: {len(X_out)}")

    return X_out[shuffle_idx], y_out[shuffle_idx]


# ── Apply augmentation ──
X_train_full, y_train_full = augment_batch(
    X_train, y_train,
    elastic_prob=0.2
)


# ── Display the results ──
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

n = min(4, len(X_train))

for i in range(n):

    axes[0, i].imshow(X_train[i], cmap='gray')
    axes[0, i].set_title('Original image')
    axes[0, i].axis('off')

    axes[1, i].imshow(X_train_full[i], cmap='gray')
    axes[1, i].set_title('Augmented image')
    axes[1, i].axis('off')

    axes[2, i].imshow(y_train_full[i, :, :, 0], cmap='gray')
    axes[2, i].set_title('Mask (label)')
    axes[2, i].axis('off')

plt.suptitle('Augmentation Results',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# ## 8. Dice Loss & Metrics (Dice · IoU · Precision · Recall · Accuracy)



SMOOTH = 1e-6

# ── Dice Loss ──
# Measure the difference between the predicted mask and the ground-truth mask
def dice_loss(y_true, y_pred):

    # Flatten into a one-dimensional vector
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)

    # Calculate the intersection
    intersection = K.sum(y_true_f * y_pred_f)

    # Dice Loss = 1 - Dice Coefficient
    return 1.0 - (
        (2. * intersection + SMOOTH) /
        (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)
    )


# ── Combined BCE and Dice loss ──
# BCE learns pixel-wise predictions, while Dice optimizes segmentation overlap
def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# =========================
# ── EVALUATION METRICS ──
# =========================


# Dice Coefficient (predicted-region overlap)
def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (
        (2. * intersection + SMOOTH) /
        (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)
    )


# IoU (Intersection over Union)
# measure predicted-region agreement with the ground truth
def iou_coef(y_true, y_pred):

    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    union = (
        K.sum(y_true_f) +
        K.sum(y_pred_f) -
        intersection
    )

    return (intersection + SMOOTH) / (union + SMOOTH)


# Precision (precision of lesion-region predictions)
def precision_m(y_true, y_pred):

    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred > 0.5, tf.float32))

    tp = K.sum(y_true_f * y_pred_f)  # True Positive
    fp = K.sum((1 - y_true_f) * y_pred_f)  # False Positive

    return (tp + SMOOTH) / (tp + fp + SMOOTH)


# Recall (ability to correctly detect lesion regions)
def recall_m(y_true, y_pred):

    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred > 0.5, tf.float32))

    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))

    return (tp + SMOOTH) / (tp + fn + SMOOTH)


# ── Metrics used when compiling the model ──
METRICS = [
    dice_coef,       # region overlap
    iou_coef,        # Intersection over Union
    precision_m,     # precision of lesion-region predictions
    recall_m,        # lesion-region detection ability
    'accuracy'       # pixel accuracy
]

print("Configured loss: Focal + Dice | Metrics: Dice, IoU, Precision, Recall, Accuracy")


# ## 9. Build the U-Net Model



from tensorflow.keras.regularizers import l2


# ── Convolution block ──
# Contains two convolution layers, batch normalization, and optional dropout
# Purpose: extract image features

def conv_block(x, filters, kernel_size=(3, 3),
               use_batch_norm=True, dropout=0.0):

    x = Conv2D(filters, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               activation='relu',
               kernel_regularizer=l2(1e-4))(x)

    if use_batch_norm:
        x = BatchNormalization()(x)

    x = Conv2D(filters, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               activation='relu',
               kernel_regularizer=l2(1e-4))(x)

    if use_batch_norm:
        x = BatchNormalization()(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    return x


# ── Merge the skip connection (encoder ↔ decoder) ──
# Crop tensors when their dimensions do not match, then concatenate them

def crop_concat(upsampled, skip):

    up_h, up_w = K.int_shape(upsampled)[1:3]
    sk_h, sk_w = K.int_shape(skip)[1:3]

    dh, dw = sk_h - up_h, sk_w - up_w

    if dh != 0 or dw != 0:
        skip = Cropping2D(((dh // 2, dh - dh // 2),
                           (dw // 2, dw - dw // 2)))(skip)

    return concatenate([upsampled, skip])


# ── Build the U-Net Model ──
# Input → Encoder → Bottleneck → Decoder → Output mask

def build_unet(input_shape, num_filters=64,
               dropout=0.3, use_bn=True):

    inputs = Input(input_shape)

    # =========================
    # ── ENCODER (Downsampling)
    # =========================
    c1 = conv_block(inputs, num_filters,
                    use_batch_norm=use_bn, dropout=0.0)
    p1 = MaxPool2D((2, 2))(c1)

    c2 = conv_block(p1, num_filters * 2,
                    use_batch_norm=use_bn, dropout=0.0)
    p2 = MaxPool2D((2, 2))(c2)

    c3 = conv_block(p2, num_filters * 4,
                    use_batch_norm=use_bn, dropout=0.1)
    p3 = MaxPool2D((2, 2))(c3)

    c4 = conv_block(p3, num_filters * 8,
                    use_batch_norm=use_bn, dropout=0.2)
    p4 = MaxPool2D((2, 2))(c4)

    # =========================
    # ── BOTTLENECK (Deep layer)
    # =========================
    c5 = conv_block(p4, num_filters * 16,
                    use_batch_norm=use_bn, dropout=0.3)

    # =========================
    # ── DECODER (Upsampling)
    # =========================
    u6 = Conv2DTranspose(num_filters * 8, (2, 2),
                         strides=(2, 2), padding='same')(c5)
    u6 = crop_concat(u6, c4)
    c6 = conv_block(u6, num_filters * 8,
                    use_batch_norm=use_bn, dropout=0.2)

    u7 = Conv2DTranspose(num_filters * 4, (2, 2),
                         strides=(2, 2), padding='same')(c6)
    u7 = crop_concat(u7, c3)
    c7 = conv_block(u7, num_filters * 4,
                    use_batch_norm=use_bn, dropout=0.1)

    u8 = Conv2DTranspose(num_filters * 2, (2, 2),
                         strides=(2, 2), padding='same')(c7)
    u8 = crop_concat(u8, c2)
    c8 = conv_block(u8, num_filters * 2,
                    use_batch_norm=use_bn, dropout=0.0)

    u9 = Conv2DTranspose(num_filters, (2, 2),
                         strides=(2, 2), padding='same')(c8)
    u9 = crop_concat(u9, c1)
    c9 = conv_block(u9, num_filters,
                    use_batch_norm=use_bn, dropout=0.0)

    # ── Output layer ──
    # sigmoid → binary segmentation (0: background, 1: lesion)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return Model(inputs, outputs, name='U-Net')


# ── Initialize the model ──
model = build_unet((256, 256, 3),
                   num_filters=64,
                   dropout=0.3)

# Display the model architecture
model.summary()


# ## 10. Visualize the U-Net Architecture



# Optional visualization dependencies must be installed separately:
# pip install pydot graphviz




# ── Plot the U-Net architecture ──
# show_shapes=True  → display tensor shapes for every layer
# show_layer_names=True → display each layer name
# rankdir='TB' → draw from top to bottom
# dpi=60 → image resolution

try:
    tf.keras.utils.plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        dpi=60,
        to_file=os.path.join(OUTPUT_DIR, 'unet_architecture.png'),
    )
except Exception as exc:
    print(f"Model architecture plot was skipped: {exc}")


# ## 11. Configure U-Net Training



# ── Training configuration ──
EPOCHS     = 120        # maximum number of full-dataset training epochs
PATIENCE   = 15         # epochs to wait before early stopping
BATCH_SIZE = 32         # larger batch size for faster training (~40%)


# ── Focal + Dice Loss ──
# Focal: focuses on hard-to-learn samples
# Dice: optimizes segmentation overlap
def focal_dice_loss(y_true, y_pred, gamma=2.0):

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # reduce the weight of easy samples
    bce_exp = K.exp(-bce)

    focal = K.mean((1 - bce_exp) ** gamma * bce)

    # combine with Dice loss
    return focal + dice_loss(y_true, y_pred)


# ── Warmup Learning Rate Scheduler ──
# During the initial stage, gradually increase the learning rate for stable training

class WarmupLRSchedule(tf.keras.callbacks.Callback):

    def __init__(self, warmup_epochs=5, base_lr=3e-4, warmup_start=1e-5):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.base_lr       = base_lr
        self.warmup_start  = warmup_start

    def on_epoch_begin(self, epoch, logs=None):

        if epoch < self.warmup_epochs:

            # increase the learning rate linearly
            lr = self.warmup_start + (
                self.base_lr - self.warmup_start
            ) * (epoch / self.warmup_epochs)

            self.model.optimizer.learning_rate.assign(lr)

            print(f"🔥 Warmup LR - Epoch {epoch+1}: lr = {lr:.2e}")


# ── Compile model ──
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_dice_loss,
    metrics=METRICS
)


# ── Track the duration of each epoch ──
epoch_times = []

class EpochTimer(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_times.append(time.time() - self._t)


# ── Callbacks used during training ──
callbacks = [

    # Warmup learning rate
    WarmupLRSchedule(
        warmup_epochs=5,
        base_lr=3e-4,
        warmup_start=1e-5
    ),

    # Save the best model
    tf.keras.callbacks.ModelCheckpoint(
        "best_unet.keras",
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # Stop early when performance no longer improves
    EarlyStopping(
        monitor='val_dice_coef',
        mode='max',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce the learning rate when the model reaches a plateau
    ReduceLROnPlateau(
        monitor='val_dice_coef',
        mode='max',
        patience=8,
        factor=0.5,
        min_lr=1e-7,
        verbose=1
    ),

    # measure the duration of every training epoch
    EpochTimer(),
]


# ── Display the training configuration ──
print(f"Model compiled | Epochs={EPOCHS} | Batch={BATCH_SIZE} | LR=3e-4 | ReduceLR=0.5")


# ## 12. Train the U-Net Model 



# ── Start measuring training time ──
# t0 is used to calculate the total model-training time
t0 = time.time()


# ── Train the U-Net model ──
# X_train_full, y_train_full: data after augmentation
# validation_data: used to monitor overfitting
# epochs: number of full-dataset training epochs
# batch_size: number of images per gradient update
# callbacks: support mechanisms (early stopping, model saving, learning-rate reduction, etc.)

history = model.fit(
    X_train_full,
    y_train_full,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,     # shuffle the data every epoch for better learning
    verbose=1         # display training logs
)


# ── Finish measuring training time ──
train_time = time.time() - t0


# ── Print the training results ──
print(
    f"\n⏱ Training time: {train_time/60:.2f} minutes"
    f" | Actual completed epochs: {len(history.history['loss'])}"
)


# ## 13. Training-Time Chart



# ── Training Time Analysis ──

epochs_done = len(history.history['loss'])
ep_range    = np.arange(1, len(epoch_times) + 1)

fig, ax = plt.subplots(figsize=(10, 5))

# Per-epoch duration line with circular markers
ax.plot(
    ep_range,
    epoch_times,
    marker='o',      # add circular markers
    linewidth=2,
    markersize=5,
    label='Time per epoch'
)

# Average line
ax.axhline(
    np.mean(epoch_times),
    linestyle='--',
    linewidth=2,
    label=f'Average = {np.mean(epoch_times):.1f}s'
)

ax.set_xlabel('Epoch')
ax.set_ylabel('Time (seconds)')
ax.set_title('Training Time per Epoch',
             fontsize=13,
             fontweight='bold')

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# ── Statistics ──
print(f"Average time per epoch : {np.mean(epoch_times):.2f} s")
print(f"Fastest epoch              : {np.min(epoch_times):.2f} s")
print(f"Slowest epoch               : {np.max(epoch_times):.2f} s")
print(f"Total training time     : {train_time/60:.2f} minutes")


# ## 14. Loss and Metric Charts



import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Metrics to display
# ==========================================================
metric_pairs = [
    ('loss',        'val_loss',        'Loss'),
    ('dice_coef',   'val_dice_coef',   'Dice Coefficient'),
    ('iou_coef',    'val_iou_coef',    'IoU'),
    ('precision_m', 'val_precision_m', 'Precision'),
    ('recall_m',    'val_recall_m',    'Recall'),
    ('accuracy',    'val_accuracy',    'Accuracy'),
]

# ==========================================================
# Evaluate on the test set
# ==========================================================
test_results = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

# ==========================================================
# The final best epoch has the highest validation Dice score
# ==========================================================
val_dice = np.array(history.history['val_dice_coef'])

best_epoch = np.where(
    np.isclose(val_dice, np.max(val_dice))
)[0][-1] + 1

best_idx = best_epoch - 1

print(f"Best Epoch (Last Saved) : {best_epoch}")
print(f"Best Val Dice           : {val_dice[best_idx]:.4f}")

# ==========================================================
# PLOT TRAINING CURVES
# Display only epoch 1 through the best epoch
# ==========================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

for ax, (train_key, val_key, title) in zip(axes, metric_pairs):

    train_vals = history.history[train_key][:best_epoch]
    val_vals   = history.history[val_key][:best_epoch]

    epochs_x = np.arange(1, best_epoch + 1)

    # Train
    ax.plot(
        epochs_x,
        train_vals,
        color='blue',
        linewidth=2.5,
        label='Train'
    )

    # Validation
    ax.plot(
        epochs_x,
        val_vals,
        color='orange',
        linewidth=2.5,
        label='Validation'
    )

    ax.set_title(
        title,
        fontsize=12,
        fontweight='bold'
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)

    ax.set_xlim(1, best_epoch)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.suptitle(
    f'U-Net Training Curves (Epoch 1 → {best_epoch})',
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

# ==========================================================
# TRAINING SUMMARY
# ==========================================================
print("=" * 65)
print("U-NET TRAINING SUMMARY")
print("=" * 65)

print(f"\nBest Epoch        : {best_epoch}")
print(f"Best Val Dice     : {history.history['val_dice_coef'][best_idx]:.4f}")
print(f"Best Val IoU      : {history.history['val_iou_coef'][best_idx]:.4f}")

# ==========================================================
# TRAIN METRICS
# ==========================================================
print("\n" + "=" * 65)
print(f"TRAIN METRICS AT BEST EPOCH ({best_epoch})")
print("=" * 65)

print(f"Train Loss        : {history.history['loss'][best_idx]:.4f}")
print(f"Train Dice        : {history.history['dice_coef'][best_idx]:.4f}")
print(f"Train IoU         : {history.history['iou_coef'][best_idx]:.4f}")
print(f"Train Precision   : {history.history['precision_m'][best_idx]:.4f}")
print(f"Train Recall      : {history.history['recall_m'][best_idx]:.4f}")
print(f"Train Accuracy    : {history.history['accuracy'][best_idx]:.4f}")

# ==========================================================
# VALIDATION METRICS
# ==========================================================
print("\n" + "=" * 65)
print(f"VALIDATION METRICS AT BEST EPOCH ({best_epoch})")
print("=" * 65)

print(f"Val Loss          : {history.history['val_loss'][best_idx]:.4f}")
print(f"Val Dice          : {history.history['val_dice_coef'][best_idx]:.4f}")
print(f"Val IoU           : {history.history['val_iou_coef'][best_idx]:.4f}")
print(f"Val Precision     : {history.history['val_precision_m'][best_idx]:.4f}")
print(f"Val Recall        : {history.history['val_recall_m'][best_idx]:.4f}")
print(f"Val Accuracy      : {history.history['val_accuracy'][best_idx]:.4f}")

# ==========================================================
# TEST METRICS
# ==========================================================
print("\n" + "=" * 65)
print("TEST METRICS")
print("=" * 65)

for name, value in zip(model.metrics_names, test_results):
    print(f"{name:<18}: {value:.4f}")

print("\nTotal Epochs Trained :", len(history.history['loss']))
print("Best Epoch Used      :", best_epoch)
print("=" * 65)


# ## 15. Evaluate on the Test Set



# ─────────────────────────────────────────────
# EVALUATE THE MODEL ON THE TEST SET
# ─────────────────────────────────────────────
# Purpose:
# - Check the model’s generalization ability
# - Evaluate using data that was NEVER seen during training
# ─────────────────────────────────────────────

results = model.evaluate(
    X_test,      # test images
    y_test,      # mask test (ground truth)
    verbose=0    # do not display detailed logs
)

metric_names = [
    'Loss (Focal + Dice)',   # total loss function
    'Dice',                # predicted-region overlap
    'IoU',                 # Intersection over Union
    'Precision',           # precision of lesion-region predictions
    'Recall',              # lesion-region detection ability
    'Accuracy'             # pixel accuracy
]


# ─────────────────────────────────────────────
# PRINT RESULTS AS TEXT 
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("           TEST SET RESULTS")
print("=" * 55)

for name, value in zip(metric_names, results):
    print(f"{name:<20}: {value:.4f}")

print("=" * 55)


# ## 16. Evaluation-Result Charts



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# 1. IDENTIFY THE BEST EPOCH (based on val_dice_coef)
# ─────────────────────────────────────────────
best_epoch_idx = int(np.argmax(history.history['val_dice_coef']))  # index, 0-based
best_epoch = best_epoch_idx + 1  # one-based epoch number for display

# ─────────────────────────────────────────────
# 2. LOAD THE BEST MODEL AND EVALUATE IT ON THE TEST SET
# ─────────────────────────────────────────────
best_model = load_model(
    "best_unet.keras",
    custom_objects={
        'focal_dice_loss': focal_dice_loss,
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'iou_coef': iou_coef,
        'precision_m': precision_m,
        'recall_m': recall_m,
    }
)

results = best_model.evaluate(X_test, y_test, verbose=0)

# ─────────────────────────────────────────────
# 3. LIST OF METRICS TO COMPARE
# ─────────────────────────────────────────────
metric_keys   = ['dice_coef', 'iou_coef', 'precision_m', 'recall_m', 'accuracy']
metric_labels = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy']

# ─────────────────────────────────────────────
# 4. TEST-SET VALUES (results[0] is loss and is excluded)
# ─────────────────────────────────────────────
test_vals = [float(v) for v in results[1:6]]

# ─────────────────────────────────────────────
# 5. TRAINING VALUES AT THE BEST EPOCH
# ─────────────────────────────────────────────
train_vals = [
    float(history.history['dice_coef'][best_epoch_idx]),
    float(history.history['iou_coef'][best_epoch_idx]),
    float(history.history['precision_m'][best_epoch_idx]),
    float(history.history['recall_m'][best_epoch_idx]),
    float(history.history['accuracy'][best_epoch_idx]),
]

# ─────────────────────────────────────────────
# 6. VALIDATION VALUES AT THE BEST EPOCH
# ─────────────────────────────────────────────
val_vals = [
    float(history.history['val_dice_coef'][best_epoch_idx]),
    float(history.history['val_iou_coef'][best_epoch_idx]),
    float(history.history['val_precision_m'][best_epoch_idx]),
    float(history.history['val_recall_m'][best_epoch_idx]),
    float(history.history['val_accuracy'][best_epoch_idx]),
]

# ─────────────────────────────────────────────
# 7. COMBINE THE THREE SPLITS: TRAIN / VALIDATION / TEST
# ─────────────────────────────────────────────
sets = [
    ('Train', train_vals, '#2CA02C'),
    ('Val',   val_vals,   '#DD8452'),
    ('Test',  test_vals,  '#4C72B0'),
]

# ─────────────────────────────────────────────
# 8. PLOT BAR AND RADAR CHARTS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))

for col, (name, vals, color) in enumerate(sets):

    # ── BAR CHART ──
    ax_bar = fig.add_subplot(2, 3, col + 1)

    bars = ax_bar.bar(
        metric_labels,
        vals,
        color=color,
        edgecolor='black',
        width=0.55,
        alpha=0.85
    )

    for bar, v in zip(bars, vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{v:.4f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel('Score')
    ax_bar.set_title(
        f'U-Net — {name} Metrics\n(Best Epoch = {best_epoch})',
        fontsize=12,
        fontweight='bold'
    )
    ax_bar.grid(axis='y', alpha=0.3)

    # ── RADAR CHART ──
    ax_rad = fig.add_subplot(2, 3, col + 4, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    vals_r   = vals + [vals[0]]
    angles_r = angles + [angles[0]]

    ax_rad.plot(angles_r, vals_r, linewidth=2, color=color)
    ax_rad.fill(angles_r, vals_r, alpha=0.25, color=color)

    ax_rad.set_thetagrids(np.degrees(angles), metric_labels, fontsize=10)
    ax_rad.set_ylim(0, 1)

    ax_rad.set_title(
        f'Radar — {name}',
        fontsize=12,
        fontweight='bold',
        pad=15
    )
    ax_rad.grid(alpha=0.3)

# ─────────────────────────────────────────────
# 9. OVERALL TITLE
# ─────────────────────────────────────────────
plt.suptitle(
    f'U-Net — Training / Validation / Test Comparison | Best Epoch = {best_epoch}',
    fontsize=15,
    fontweight='bold',
    y=1.01
)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 10. PRINT THE RESULT SUMMARY
# ─────────────────────────────────────────────
print("=" * 70)
print(f"U-NET BEST RESULTS (Best Epoch = {best_epoch})")
print("=" * 70)

print("\n--- Train Metrics (Best Epoch) ---")
for label, val in zip(metric_labels, train_vals):
    print(f"{label:<12}: {val:.4f}")

print("\n--- Validation Metrics (Best Epoch) ---")
for label, val in zip(metric_labels, val_vals):
    print(f"{label:<12}: {val:.4f}")

print("\n--- Test Metrics (Best Model) ---")
for label, val in zip(metric_labels, test_vals):
    print(f"{label:<12}: {val:.4f}")

print("=" * 70)


# ## 16b. Dice and IoU Charts



import numpy as np
import matplotlib.pyplot as plt

# Test evaluation
test_results = model.evaluate(X_test, y_test, verbose=0)

test_score_map = {
    'loss':        test_results[0],
    'dice_coef':   test_results[1],
    'iou_coef':    test_results[2],
    'precision_m': test_results[3],
    'recall_m':    test_results[4],
    'accuracy':    test_results[5],
}

metric_pairs = [
    ('loss',        'val_loss',        'loss',        'Loss'),
    ('dice_coef',   'val_dice_coef',   'dice_coef',   'Dice'),
    ('iou_coef',    'val_iou_coef',    'iou_coef',    'IoU'),
    ('precision_m', 'val_precision_m', 'precision_m', 'Precision'),
    ('recall_m',    'val_recall_m',    'recall_m',    'Recall'),
    ('accuracy',    'val_accuracy',    'accuracy',    'Accuracy'),
]

epochs_x = np.arange(1, len(history.history['loss']) + 1)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, (train_key, val_key, test_key, title) in zip(axes, metric_pairs):

    train_vals = history.history[train_key]
    val_vals   = history.history[val_key]
    test_val   = test_score_map[test_key]

    # Training set (blue)
    ax.plot(
        epochs_x,
        train_vals,
        linewidth=2.5,
        color='#4C72B0',
        label=f'Train {title}'
    )

    # Validation (cam)
    ax.plot(
        epochs_x,
        val_vals,
        linewidth=2.5,
        color='#DD8452',
        label=f'Val {title}'
    )

    # Test (green)
    ax.axhline(
        test_val,
        linewidth=2,
        linestyle=':',
        color='#2CA02C',
        label=f'Test {title} = {test_val:.4f}'
    )

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if title != 'Loss':
        ax.set_ylim(0, 1)

plt.suptitle(
    'U-Net Training Curves (Train / Validation / Test)',
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout()
plt.show()


# ## 16c. ROC Curve Chart (Train / Validation / Test)



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ─────────────────────────────────────────────
# FUNCTION TO CALCULATE A SEGMENTATION ROC CURVE
# ─────────────────────────────────────────────
# Idea:
# - Flatten mask (2D image → 1D vector)
# - Flatten prediction
# - Calculate FPR and TPR to plot the ROC curve
# - Calculate AUC (area under the curve)
# ─────────────────────────────────────────────

def get_roc(X, y):

    # Predict probabilities with the model
    preds = model.predict(X, verbose=0)

    # Flatten into one-dimensional vectors for pixel-wise ROC calculation
    y_flat = y.flatten()
    pred_flat = preds.flatten()

    # ROC curve: False Positive Rate vs True Positive Rate
    fpr, tpr, _ = roc_curve(y_flat, pred_flat)

    # AUC: area under the ROC curve
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


# ─────────────────────────────────────────────
# CALCULATE ROC FOR THE THREE DATA SPLITS
# ─────────────────────────────────────────────

print("🔵 Calculating ROC for the training set...")
fpr_tr, tpr_tr, auc_tr = get_roc(X_train_full, y_train_full)

print("🟠 Calculating ROC for the validation set...")
fpr_val, tpr_val, auc_val = get_roc(X_val, y_val)

print("🟢 Calculating ROC for the test set...")
fpr_te, tpr_te, auc_te = get_roc(X_test, y_test)


# ─────────────────────────────────────────────
# PLOT THE ROC CURVE
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 7))

# ── ROC Train ──
ax.plot(
    fpr_tr,
    tpr_tr,
    linewidth=2,
    color='#4C72B0',
    label=f'Train (AUC = {auc_tr:.4f})'
)

# ── ROC Validation ──
ax.plot(
    fpr_val,
    tpr_val,
    linewidth=2,
    linestyle='--',
    color='#DD8452',
    label=f'Val (AUC = {auc_val:.4f})'
)

# ── ROC Test ──
ax.plot(
    fpr_te,
    tpr_te,
    linewidth=2,
    linestyle=':',
    color='#2CA02C',
    label=f'Test (AUC = {auc_te:.4f})'
)

# ── Baseline representing random predictions ──
ax.plot(
    [0, 1],
    [0, 1],
    'k--',
    alpha=0.4,
    label='Random (AUC = 0.5)'
)


# ─────────────────────────────────────────────
# CONFIGURE THE CHART
# ─────────────────────────────────────────────

ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_title('ROC Curve — Pixel-Level Image Segmentation', fontweight='bold')

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()


# ## 16d. Confusion Matrix (Pixel-level)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


# ─────────────────────────────────────────────
# FUNCTION TO PLOT A SEGMENTATION CONFUSION MATRIX
# ─────────────────────────────────────────────
# Purpose:
# - Compare actual pixels (y_true) and predicted pixels (y_pred)
# - Convert segmentation into a pixel-wise binary-classification problem
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred_prob, threshold=0.5,
                           title='Confusion Matrix'):

    # Clip predicted values to [0, 1]
    y_pred_prob = np.clip(y_pred_prob, 0, 1)

    # Flatten images into one-dimensional vectors for pixel-level evaluation
    y_true_flat = y_true.squeeze().flatten().astype(int)
    y_pred_flat = (y_pred_prob.squeeze().flatten() > threshold).astype(int)

    # Create the confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    # TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()


    # ─────────────────────────────────────────
    # NORMALIZE THE CONFUSION MATRIX
    # ─────────────────────────────────────────
    # Convert values to percentages
    cm_sum = cm.sum(axis=1, keepdims=True)

    # avoid division by zero
    cm_sum = np.where(cm_sum == 0, 1, cm_sum)

    cm_norm = cm.astype(float) / cm_sum


    # ─────────────────────────────────────────
    # PLOT THE CHART
    # ─────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    classes = ['Background (0)', 'Foreground (1)']


    # =====================================================
    # 1. CONFUSION MATRIX BY COUNT
    # =====================================================
    im = axes[0].imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[0].set_title('Confusion Matrix (Count)', fontweight='bold')


    # =====================================================
    # 2. CONFUSION MATRIX BY PERCENTAGE
    # =====================================================
    im = axes[1].imshow(cm_norm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[1].set_title('Confusion Matrix (Percentage)', fontweight='bold')


    # ─────────────────────────────────────────
    # DISPLAY AXIS LABELS
    # ─────────────────────────────────────────

    for ax, data, fmt in zip(axes, [cm, cm_norm], ['d', '.2%']):

        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))

        ax.set_xticklabels(classes, rotation=25, ha='right')
        ax.set_yticklabels(classes)

        # Threshold for switching text color for readability
        thresh_val = data.max() / 2.0

        # Write values inside each cell
        for i, j in itertools.product(range(2), range(2)):

            value = data[i, j]

            ax.text(
                j, i,
                f'{value:.2f}' if fmt == '.2%' else f'{int(value)}',
                ha='center',
                va='center',
                color='white' if value > thresh_val else 'black',
                fontsize=12
            )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')


    # ─────────────────────────────────────────
    # SUMMARY TITLE FOR TN / FP / FN / TP
    # ─────────────────────────────────────────

    plt.suptitle(
        f'TN = {tn:,} | FP = {fp:,} | FN = {fn:,} | TP = {tp:,}',
        fontsize=11,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# RUN ON THE TEST SET
# ─────────────────────────────────────────────

test_preds = model.predict(X_test, verbose=0)

plot_confusion_matrix(
    y_test,
    test_preds,
    title='Confusion Matrix - Test Set'
)


# ## 17. Test the Trained U-Net and Compare Predicted Masks with Ground Truth



import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score


# =====================================================
# FUNCTION TO PREDICT A MASK FOR ONE IMAGE
# =====================================================
def predict_mask(image):

    # Add the batch dimension
    image = np.expand_dims(image, axis=0)

    # Measure inference time
    t0 = time.time()

    pred = model.predict(image, verbose=0)[0, :, :, 0]

    inference_time = time.time() - t0

    return pred, inference_time


# =====================================================
# CALCULATE METRICS FOR ONE IMAGE
# =====================================================
def calculate_metrics(y_true, y_pred):

    y_true = y_true.astype(np.uint8).flatten()
    y_pred = y_pred.astype(np.uint8).flatten()

    intersection = np.sum(y_true * y_pred)

    # Dice
    dice = (
        2.0 * intersection + 1e-7
    ) / (
        np.sum(y_true) + np.sum(y_pred) + 1e-7
    )

    # IoU
    iou = (
        intersection + 1e-7
    ) / (
        np.sum(y_true) + np.sum(y_pred) - intersection + 1e-7
    )

    # Precision
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Recall
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    return dice, iou, precision, recall, accuracy


# =====================================================
# DISPLAY PREDICTION RESULTS
# =====================================================
def visualize_prediction(idx=None):

    # Select a random sample when no index is provided
    if idx is None:
        idx = np.random.randint(0, len(X_test))

    # Test image
    img = X_test[idx]

    # Ground Truth Mask
    gt_mask = np.squeeze(y_test[idx])

    # Predicted
    pred_mask, inference_time = predict_mask(img)

    # Convert to a binary mask
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # Calculate the metrics
    dice, iou, precision, recall, accuracy = calculate_metrics(
        gt_mask,
        binary_mask
    )

    # =================================================
    # PLOT THE RESULTS
    # =================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Input image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input image", fontsize=12)
    axes[0].axis('off')

    # Ground Truth
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground-truth mask", fontsize=12)
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title(
        f"Predicted\n"
        f"Dice = {dice:.4f}\n"
        f"IoU = {iou:.4f}\n"
        f"P = {precision:.4f} | R = {recall:.4f}\n"
        f"Acc = {accuracy:.4f}\n"
        f"{inference_time*1000:.1f} ms",
        fontsize=10
    )
    axes[2].axis('off')

    # Overall title
    plt.suptitle(
        f"Prediction result for test sample #{idx}",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()

    # =================================================
    # PRINT METRICS TO THE CONSOLE
    # =================================================
    print("=" * 50)
    print(f"Sample Index    : {idx}")
    print(f"Dice Score      : {dice:.4f}")
    print(f"IoU             : {iou:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Inference Time  : {inference_time*1000:.2f} ms")
    print("=" * 50)


# =====================================================
# TEST TEN RANDOM IMAGES
# =====================================================
for _ in range(10):
    visualize_prediction()




import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score

# =====================================================
# FUNCTION TO PREDICT A MASK FOR ONE IMAGE
# =====================================================
def predict_mask(image):

    # Add the batch dimension
    image = np.expand_dims(image, axis=0)

    # Measure inference time
    start_time = time.time()

    # Predicted
    pred = model.predict(image, verbose=0)[0, :, :, 0]

    # Inference time
    inference_time = time.time() - start_time

    return pred, inference_time


# =====================================================
# FUNCTION TO CALCULATE METRICS
# =====================================================
def calculate_metrics(y_true, y_pred):

    y_true = y_true.astype(np.uint8).flatten()
    y_pred = y_pred.astype(np.uint8).flatten()

    intersection = np.sum(y_true * y_pred)

    # Dice Score
    dice = (
        2.0 * intersection + 1e-7
    ) / (
        np.sum(y_true) + np.sum(y_pred) + 1e-7
    )

    # IoU
    iou = (
        intersection + 1e-7
    ) / (
        np.sum(y_true) + np.sum(y_pred) - intersection + 1e-7
    )

    # Precision
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Recall
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    return dice, iou, precision, recall, accuracy


# =====================================================
# DISPLAY THE FIRST TEN IMAGES FROM THE TEST SET
# =====================================================
num_samples = min(10, len(X_test))

fig, axes = plt.subplots(
    num_samples,
    3,
    figsize=(18, 4 * num_samples)
)

# Handle the single-image case
if num_samples == 1:
    axes = np.expand_dims(axes, axis=0)

for idx in range(num_samples):

    # ------------------------------------------
    # Input image
    # ------------------------------------------
    img = X_test[idx]

    # Ground Truth
    gt_mask = np.squeeze(y_test[idx])

    # Predicted
    pred_mask, inference_time = predict_mask(img)

    # Convert to a binary mask mask
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # ------------------------------------------
    # Calculate the metrics
    # ------------------------------------------
    dice, iou, precision, recall, accuracy = calculate_metrics(
        gt_mask,
        binary_mask
    )

    # ------------------------------------------
    # COLUMN 1: INPUT IMAGE
    # ------------------------------------------
    axes[idx, 0].imshow(img, cmap='gray')
    axes[idx, 0].set_title(
        f'Image #{idx}',
        fontsize=11,
        fontweight='bold'
    )
    axes[idx, 0].axis('off')

    # ------------------------------------------
    # COLUMN 2: GROUND-TRUTH MASK
    # ------------------------------------------
    axes[idx, 1].imshow(gt_mask, cmap='gray')
    axes[idx, 1].set_title(
        'Ground Truth',
        fontsize=11,
        fontweight='bold'
    )
    axes[idx, 1].axis('off')

    # ------------------------------------------
    # COLUMN 3: PREDICTED MASK AND METRICS
    # ------------------------------------------
    axes[idx, 2].imshow(binary_mask, cmap='gray')
    axes[idx, 2].set_title(
        f'Dice={dice:.4f}\n'
        f'IoU={iou:.4f}\n'
        f'P={precision:.4f} | R={recall:.4f}\n'
        f'Acc={accuracy:.4f}\n'
        f'{inference_time*1000:.1f} ms',
        fontsize=9
    )
    axes[idx, 2].axis('off')

    # ------------------------------------------
    # In ra Console
    # ------------------------------------------
    print("=" * 60)
    print(f"Sample #{idx}")
    print(f"Dice Score     : {dice:.4f}")
    print(f"IoU            : {iou:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Inference Time : {inference_time*1000:.2f} ms")
    print("=" * 60)

# =====================================================
# OVERALL TITLE
# =====================================================
plt.suptitle(
    'SEGMENTATION RESULTS FOR THE FIRST TEN TEST IMAGES',
    fontsize=18,
    fontweight='bold',
    y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.show()


# ## 18. Reload the Saved Best Model



from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1. LOAD THE BEST MODEL
# ─────────────────────────────────────────────
# Purpose:
# - Reload the best trained model according to validation Dice
# - Custom objects are required because the model uses custom losses and metrics
# ─────────────────────────────────────────────

best_model = load_model(
    'best_unet.keras',
    custom_objects={
        'focal_dice_loss': focal_dice_loss,
        'dice_loss':       dice_loss,
        'dice_coef':       dice_coef,
        'iou_coef':        iou_coef,
        'precision_m':     precision_m,
        'recall_m':        recall_m,
    }
)

print("✅ Best model loaded: best_unet.keras")


# ─────────────────────────────────────────────
# 2. EVALUATE ON THE TEST SET
# ─────────────────────────────────────────────
# Returns: [loss, dice, iou, precision, recall, accuracy]
# ─────────────────────────────────────────────

best_results = best_model.evaluate(X_test, y_test, verbose=1)


# ─────────────────────────────────────────────
# 3. PRINT RESULTS
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("      BEST MODEL RESULTS ON THE TEST SET")
print("="*50)

for name, value in zip(best_model.metrics_names, best_results):
    print(f"{name:20s}: {value:.4f}")


# ─────────────────────────────────────────────
# 4. STORE RESULTS IN A DICTIONARY
# ─────────────────────────────────────────────

test_score_map = dict(zip(best_model.metrics_names, best_results))

print("\nResults as a dictionary:")
print(test_score_map)


# ─────────────────────────────────────────────
# 5. PLOT A BAR CHART
# ─────────────────────────────────────────────
# Purpose:
# - Compare model metrics on the test set
# ─────────────────────────────────────────────

metric_names  = []
metric_values = []

for name, value in zip(best_model.metrics_names, best_results):
    if name != 'loss':   # exclude loss for easier viewing
        metric_names.append(name)
        metric_values.append(value)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    metric_names,
    metric_values,
    edgecolor='black',
    color='#4C72B0'
)

# Write values above each bar
for bar, val in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f"{val:.4f}",
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("U-Net Performance on the Test Set", fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# 6. TEST TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────
# Idea:
# - Create multiple image variants using horizontal and vertical flips
# - Run multiple predictions
# - Average the predictions to improve stability
# ─────────────────────────────────────────────

def predict_tta(model, image):

    # Create four image variants
    imgs = [
        image,
        np.fliplr(image),
        np.flipud(image),
        np.fliplr(np.flipud(image)),
    ]

    preds = []

    for img in imgs:
        p = model.predict(np.expand_dims(img, 0), verbose=0)[0]
        preds.append(p)

    # reverse the transformations to restore the original orientation
    preds[1] = np.fliplr(preds[1])
    preds[2] = np.flipud(preds[2])
    preds[3] = np.fliplr(np.flipud(preds[3]))

    # average the four predictions
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────
# 7. RUN TTA ON THE ENTIRE TEST SET
# ─────────────────────────────────────────────

print("⏳ Running TTA on the test set...")

tta_preds = np.array([
    predict_tta(best_model, x) for x in X_test
])


# ─────────────────────────────────────────────
# 8. CALCULATE DICE AND IOU AFTER TTA
# ─────────────────────────────────────────────

tta_dice = dice_coef(y_test, tta_preds).numpy()
tta_iou  = iou_coef(y_test, tta_preds).numpy()


# ─────────────────────────────────────────────
# 9. COMPARE RESULTS BEFORE AND AFTER TTA
# ─────────────────────────────────────────────

print("=" * 50)
print("      RESULTS AFTER TEST-TIME AUGMENTATION (TTA)")
print("=" * 50)

print(f"TTA Dice : {tta_dice:.4f}  (before: {test_score_map.get('dice_coef', 0):.4f})")
print(f"TTA IoU  : {tta_iou:.4f}  (before: {test_score_map.get('iou_coef', 0):.4f})")

print(f"Dice change : {tta_dice - test_score_map.get('dice_coef', 0):+.4f}")
print(f"IoU change : {tta_iou  - test_score_map.get('iou_coef', 0):+.4f}")

print("=" * 50)




# =====================================================
# SAVE THE U-NET MODEL
# =====================================================
model.save(os.path.join(OUTPUT_DIR, 'unet_without_balancing_full.keras'))
print("✅ The model was successfully saved to the output directory!")

print(f"Saved model path: {os.path.join(OUTPUT_DIR, 'unet_without_balancing_full.keras')}")




# =====================================================
# SAVE THE COMPLETE U-NET MODEL AND RESULTS
# =====================================================

import pickle
import pandas as pd
from tensorflow.keras.models import load_model

# -----------------------------------------------------
# 1. Save the complete model
# -----------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save(os.path.join(OUTPUT_DIR, 'unet_full_model.keras'))

print("✅ Model saved:")
print(os.path.join(OUTPUT_DIR, 'unet_full_model.keras'))


# -----------------------------------------------------
# 2. Save the training history
# -----------------------------------------------------
with open(os.path.join(OUTPUT_DIR, 'history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

print("✅ Training history saved:")
print(os.path.join(OUTPUT_DIR, 'history.pkl'))


# -----------------------------------------------------
# 3. Evaluate on the test set
# -----------------------------------------------------
test_results = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

metric_names = [
    "Loss",
    "Dice",
    "IoU",
    "Precision",
    "Recall",
    "Accuracy"
]

results_df = pd.DataFrame({
    "Metric": metric_names,
    "Value": test_results
})

print("\n📊 TEST RESULTS")
print(results_df)


# -----------------------------------------------------
# 4. Save test results to CSV
# -----------------------------------------------------
results_df.to_csv(
    os.path.join(OUTPUT_DIR, 'test_results.csv'),
    index=False
)

print("✅ Saved:")
print(os.path.join(OUTPUT_DIR, 'test_results.csv'))


# -----------------------------------------------------
# 5. Save the full training history to CSV
# -----------------------------------------------------
history_df = pd.DataFrame(history.history)

history_df.to_csv(
    os.path.join(OUTPUT_DIR, 'training_history.csv'),
    index=False
)

print("✅ Saved:")
print(os.path.join(OUTPUT_DIR, 'training_history.csv'))


# -----------------------------------------------------
# 6. Verify that the model can be loaded again
# -----------------------------------------------------
loaded_model = load_model(
    os.path.join(OUTPUT_DIR, 'unet_full_model.keras'),
    custom_objects={
        'focal_dice_loss': focal_dice_loss,
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'iou_coef': iou_coef,
        'precision_m': precision_m,
        'recall_m': recall_m,
    }
)

print("\n✅ Model loaded successfully!")


# -----------------------------------------------------
# 7. Display the list of saved files
# -----------------------------------------------------
import os

print("\n📁 FILES SAVED:")
for file in os.listdir(OUTPUT_DIR):
    if file.endswith((".keras", ".pkl", ".csv")):
        print(file)

print("\n🎉 Finished saving the complete model and results.")
# -----------------------------------------------------
# 8. Compress all result files
# -----------------------------------------------------
import zipfile
import os

zip_path = os.path.join(OUTPUT_DIR, 'unet_results.zip')

files_to_zip = [
    os.path.join(OUTPUT_DIR, 'unet_full_model.keras'),
    os.path.join(OUTPUT_DIR, 'history.pkl'),
    os.path.join(OUTPUT_DIR, 'test_results.csv'),
    os.path.join(OUTPUT_DIR, 'training_history.csv'),
]

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(
                file,
                arcname=os.path.basename(file)
            )

print("\n✅ ZIP file created:")
print(zip_path)

