"""
add_face_data.py
----------------
Adds 10k fake and 10k real face images from the 140k real-vs-fake
dataset into the existing aiImagesLarge and real_imagesLarge folders.

Combined with the existing 10k CIFAKE images this gives
20k per class total for retraining.

Place at:  imageMetadata/add_face_data.py
Run with:  python add_face_data.py
"""

import shutil
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

FACE_ROOT = PROJECT_ROOT / "images" / "faceData" / "real_vs_fake" / "real-vs-fake"

# Pull from all three splits for maximum diversity
SOURCE_DIRS = {
    "ai":   [
        FACE_ROOT / "train" / "fake",
        FACE_ROOT / "test"  / "fake",
        FACE_ROOT / "valid" / "fake",
    ],
    "real": [
        FACE_ROOT / "train" / "real",
        FACE_ROOT / "test"  / "real",
        FACE_ROOT / "valid" / "real",
    ],
}

DEST_AI_DIR   = PROJECT_ROOT / "images" / "aiImagesLarge"
DEST_REAL_DIR = PROJECT_ROOT / "images" / "real_imagesLarge"

N_IMAGES = 10000
SEED     = 42
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(folders: list) -> list:
    files = []
    for folder in folders:
        if not folder.exists():
            print(f"  [WARN] Not found: {folder}")
            continue
        found = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        print(f"  {len(found)} images in {folder.relative_to(PROJECT_ROOT)}")
        files.extend(found)
    return files


def copy_images(files: list, dst: Path, n: int, label: str):
    dst.mkdir(parents=True, exist_ok=True)
    random.shuffle(files)
    selected = files[:n]

    print(f"\nCopying {len(selected)} {label} images → {dst.name}/")
    copied = skipped = 0

    for src_file in selected:
        dst_file = dst / f"face_{src_file.name}"
        if dst_file.exists():
            skipped += 1
            continue
        shutil.copy2(src_file, dst_file)
        copied += 1

    print(f"  Done — copied {copied}, skipped {skipped} duplicates")


if __name__ == "__main__":
    random.seed(SEED)

    print("=" * 55)
    print(f"Adding {N_IMAGES} face images per class to training folders")
    print("=" * 55)

    print("\nScanning fake face sources:")
    ai_files   = collect_images(SOURCE_DIRS["ai"])

    print("\nScanning real face sources:")
    real_files = collect_images(SOURCE_DIRS["real"])

    print(f"\nTotal available — AI: {len(ai_files)}, Real: {len(real_files)}")

    if not ai_files or not real_files:
        print("[ERROR] No images found. Check folder paths.")
        exit(1)

    copy_images(ai_files,   DEST_AI_DIR,   N_IMAGES, "AI faces")
    copy_images(real_files, DEST_REAL_DIR, N_IMAGES, "Real faces")

    # Final counts
    ai_total   = len([p for p in DEST_AI_DIR.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    real_total = len([p for p in DEST_REAL_DIR.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

    print(f"\n{'=' * 55}")
    print(f"Training folder totals:")
    print(f"  aiImagesLarge    : {ai_total} images")
    print(f"  real_imagesLarge : {real_total} images")
    print(f"{'=' * 55}")