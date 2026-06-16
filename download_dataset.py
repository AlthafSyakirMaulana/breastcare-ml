import kagglehub
import shutil
from pathlib import Path

DATA_DIR = Path("/app/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading BUSI dataset...")
path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
print(f"Downloaded to: {path}")

for item in Path(path).iterdir():
    dest = DATA_DIR / item.name
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)

print(f"Dataset saved to: {DATA_DIR}")
print("Files:", list(DATA_DIR.iterdir()))
