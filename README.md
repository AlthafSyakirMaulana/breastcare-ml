---
title: BERSERI API
emoji: 🩺
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# BERSERI — Breast Early Risk Screening Intelligence

BERSERI is a breast ultrasound screening-support API.

## Endpoints

- **POST /predict** — Upload an ultrasound image (field name: `file`) and receive:
  - Classification: Normal / Jinak / Ganas with confidence scores
  - Segmentation: predicted lesion mask and overlay image
- **GET /health** — Check backend status and model loading state

## Models

- **ResNet101 classifier** — Produces the diagnosis label (Normal / Jinak / Ganas)
- **U-Net segmenter** — Predicts lesion-area mask and overlay only; does not determine diagnosis

## Important Notes

- This is a **prototype screening-support tool**, not a medical diagnosis system.
- Required local model paths:
  - `models/breast_classifier_approach1_best.pt`
  - `models/unet_unbalanced_best.keras`

## Quick Start

```bash
uvicorn api:app --host 0.0.0.0 --port 7860
```

Test:
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/predict -F "file=@sample.png"
```
