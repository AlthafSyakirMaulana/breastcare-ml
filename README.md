# BreastCare ML - Model Deteksi Kanker Payudara

Machine Learning model untuk klasifikasi citra mammografi (Normal, Jinak, Ganas).

## Struktur Folder

```
breastcare-ml/
├── data/               # Dataset (gitignored untuk file besar)
│   ├── raw/            # Data mentah
│   ├── processed/      # Data setelah preprocessing
│   └── splits/         # Train/val/test split
├── notebooks/          # Eksperimen dan analisis
├── models/             # Model yang sudah dilatih
├── src/                # Source code
│   ├── preprocessing/  # Preprocessing pipeline
│   ├── model/          # Arsitektur model
│   ├── train.py        # Training script
│   └── predict.py      # Inference script
├── requirements.txt
└── README.md
```
