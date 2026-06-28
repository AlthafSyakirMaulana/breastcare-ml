---
title: BERSERI API
emoji: 🩺
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# BERSERI ML Backend

BERSERI (Breast Early Risk Screening Intelligence) is a FastAPI-based breast ultrasound screening-support backend. It combines image classification and lesion-area segmentation to support clinical review. It is a prototype and not a medical diagnostic device.

## Core Capabilities

- Classifies breast ultrasound images into Normal, Benign, or Malignant.
- Produces category probabilities/confidence.
- Generates a U-Net segmentation mask and overlay as a visual area-of-attention aid.
- Provides health monitoring through `/health`.
- Supports CORS configuration through `ALLOWED_ORIGINS`.

## Model Architecture

- **ResNet101 classifier** is the primary model for Normal / Benign / Malignant categorization.
- **U-Net segmentation** is supportive only; it highlights the image region receiving model attention and does not independently determine benign or malignant status.

## Evaluation Results

| Component | Primary Metrics |
|---|---|
| ResNet101 classifier | Accuracy: 82.05%; Macro F1-score: 79.70%; Weighted F1-score: 82.27% |
| U-Net segmentation | Dice Score: 69.49%; IoU: 53.33% |

Metrics are internal prototype evaluation results. External clinical validation is still required.

## API Endpoints

### GET /health

Returns backend status and confirms whether both models are loaded.

### POST /predict

- **Request format**: `multipart/form-data`
- **File field name**: `file`
- **Returns**: classification result, confidence/probabilities, lesion segmentation output, and overlay data when available.

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@sample.png"
```

## Model Files

Model files are not stored in GitHub:

- `models/breast_classifier_approach1_best.pt`
- `models/unet_unbalanced_best.keras`

In deployed mode, `api.py` can download model files from [althaf505/berseri-models](https://huggingface.co/althaf505/berseri-models) when local files are unavailable.

## Local Run

1. Create or use a Python 3.11 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

## Deployment

- **Frontend**: Vercel
- **Backend API**: Hugging Face Docker Space
- **Active Space**: [althaf505/berseri-api](https://huggingface.co/spaces/althaf505/berseri-api)
- **Hugging Face app port**: 7860
- `ALLOWED_ORIGINS` must contain the production Vercel origin without a trailing slash.

## Clinical Disclaimer

BERSERI is a screening-support prototype. Its output must be reviewed alongside clinical examination and qualified healthcare-professional assessment. It must not be used as a standalone diagnosis or treatment decision.
