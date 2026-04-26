"""
add_training_data.py
--------------------
Copies a set number of images from the new GRAVEX dataset into your
existing aiImagesLarge and real_imagesLarge training folders.

Place at:  imageMetadata/add_training_data.py
Run with:  python add_training_data.py
"""

import shutil
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Source folders from the new dataset
NEW_AI_DIR   = PROJECT_ROOT / "images" / "newData" / "my_real_vs_ai_dataset" / "my_real_vs_ai_dataset" / "ai_images"
NEW_REAL_DIR = PROJECT_ROOT / "images" / "newData" / "my_real_vs_ai_dataset" / "my_real_vs_ai_dataset" / "real"

# Destination — your existing training folders
DEST_AI_DIR   = PROJECT_ROOT / "images" / "aiImagesLarge"
DEST_REAL_DIR = PROJECT_ROOT / "images" / "real_imagesLarge"

# How many images to copy from each class
N_IMAGES = 5000

SEED = 42
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def copy_images(src: Path, dst: Path, n: int, label: str):
    dst.mkdir(parents=True, exist_ok=True)

    all_files = [
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not all_files:
        print(f"[ERROR] No images found in {src}")
        return

    random.shuffle(all_files)
    selected = all_files[:n]

    print(f"\nCopying {len(selected)} {label} images → {dst}")

    copied  = 0
    skipped = 0

    for src_file in selected:
        dst_file = dst / src_file.name

        # If a file with that name already exists, add a suffix to avoid overwriting
        if dst_file.exists():
            dst_file = dst / f"new_{src_file.name}"

        if dst_file.exists():
            skipped += 1
            continue

        shutil.copy2(src_file, dst_file)
        copied += 1

    print(f"  Done — copied {copied}, skipped {skipped} duplicates")


if __name__ == "__main__":
    random.seed(SEED)

    print("=" * 50)
    print(f"Adding {N_IMAGES} AI images and {N_IMAGES} real images")
    print("=" * 50)

    # Check source folders exist
    if not NEW_AI_DIR.exists():
        print(f"[ERROR] AI source folder not found: {NEW_AI_DIR}")
        exit(1)
    if not NEW_REAL_DIR.exists():
        print(f"[ERROR] Real source folder not found: {NEW_REAL_DIR}")
        exit(1)

    copy_images(NEW_AI_DIR,   DEST_AI_DIR,   N_IMAGES, "AI")
    copy_images(NEW_REAL_DIR, DEST_REAL_DIR, N_IMAGES, "Real")

    # Print final counts
    ai_total   = len([p for p in DEST_AI_DIR.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    real_total = len([p for p in DEST_REAL_DIR.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

    print(f"\n{'=' * 50}")
    print(f"Training folder totals:")
    print(f"  aiImagesLarge   : {ai_total} images")
    print(f"  real_imagesLarge: {real_total} images")
    print(f"{'=' * 50}")