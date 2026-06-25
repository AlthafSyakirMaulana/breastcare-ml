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
├── Dockerfile
└── README.md
```

## Deploy ke Railway

1. Push repo ini ke GitHub
2. Buat akun di [Railway](https://railway.app)
3. Klik **New Project** → **Deploy from GitHub repo**
4. Pilih repo `breastcare-ml`
5. Railway akan otomatis mendeteksi `Dockerfile` dan build
6. Setelah deploy, dapatkan URL (contoh: `https://breastcare-ml.up.railway.app`)
7. Test dengan: `curl https://breastcare-ml.up.railway.app/health`

## Deploy ke Render

1. Push repo ke GitHub
2. Buat akun di [Render](https://render.com)
3. Klik **New +** → **Web Service**
4. Hubungkan GitHub repo
5. Set:
   - **Name**: `breastcare-ml`
   - **Runtime**: Docker
   - **Branch**: `main`
   - **Health Check Path**: `/health`
6. Deploy

## Hubungkan ke Frontend

Setelah backend live, set environment variable di Vercel:

```
NEXT_PUBLIC_API_URL=https://breastcare-ml.up.railway.app
```
