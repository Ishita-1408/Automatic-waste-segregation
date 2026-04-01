#!/usr/bin/env python3
"""
Download the Garbage Classification dataset from Kaggle.

Requirements:
  pip install kaggle
  Place kaggle.json at ~/.kaggle/kaggle.json  (API credentials)

Usage:
  python scripts/download_dataset.py
"""

import os
import subprocess
import zipfile
from pathlib import Path

DATASET   = "mostafaabla/garbage-classification"
DATA_DIR  = Path("data")
ZIP_NAME  = "garbage-classification.zip"


def check_kaggle_auth():
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("✗ Kaggle credentials not found.")
        print("  1. Go to: https://www.kaggle.com/settings → API → Create New Token")
        print("  2. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json")
        print("  3. chmod 600 ~/.kaggle/kaggle.json")
        raise SystemExit(1)
    print("✓ Kaggle credentials found.")


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset: {DATASET} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR)],
        check=True,
    )
    print("✓ Download complete.")


def extract():
    zip_path = DATA_DIR / ZIP_NAME
    if not zip_path.exists():
        # Kaggle sometimes names the file differently
        zips = list(DATA_DIR.glob("*.zip"))
        if not zips:
            raise FileNotFoundError("No zip file found in data/")
        zip_path = zips[0]

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()
    print("✓ Extraction complete.")


def verify():
    expected_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    # Find the root folder (dataset might be nested)
    for candidate in [DATA_DIR / "garbage_classification",
                      DATA_DIR / "Garbage classification",
                      DATA_DIR]:
        if candidate.is_dir():
            found = [d.name.lower() for d in candidate.iterdir() if d.is_dir()]
            if all(c in found for c in expected_classes):
                total = sum(len(list((candidate / c).glob("*.*"))) for c in expected_classes)
                print(f"✓ Dataset verified at: {candidate}")
                print(f"  Classes : {expected_classes}")
                print(f"  Images  : {total:,}")
                return str(candidate)

    print("⚠ Could not verify dataset structure. Check data/ directory manually.")
    return str(DATA_DIR)


if __name__ == "__main__":
    check_kaggle_auth()
    download()
    extract()
    path = verify()
    print(f"\nNext step: python src/train.py --data {path}")
