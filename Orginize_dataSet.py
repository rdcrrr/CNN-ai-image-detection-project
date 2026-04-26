"""
organize_dataset.py
-------------------
Copies images from the CIFAKE dataset (newData2) into the
aiImagesLarge and real_imagesLarge training folders.

Pulls from both train/ and test/ splits so we get maximum diversity.
Caps each class at N_IMAGES to keep training balanced.

Place at:  imageMetadata/organize_dataset.py
Run with:  python organize_dataset.py
"""

import shutil
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Source folders from CIFAKE
CIFAKE_ROOT  = PROJECT_ROOT / "images" / "newData2"
SOURCE_DIRS  = {
    "ai":   [
        CIFAKE_ROOT / "train" / "FAKE",
        CIFAKE_ROOT / "test"  / "FAKE",
    ],
    "real": [
        CIFAKE_ROOT / "train" / "REAL",
        CIFAKE_ROOT / "test"  / "REAL",
    ],
}

# Destination folders
DEST_AI_DIR   = PROJECT_ROOT / "images" / "aiImagesLarge"
DEST_REAL_DIR = PROJECT_ROOT / "images" / "real_imagesLarge"

# Max images per class
N_IMAGES = 10000

SEED = 42
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(folders: list) -> list:
    """Collects all image paths from a list of folders."""
    files = []
    for folder in folders:
        if not folder.exists():
            print(f"  [WARN] Folder not found: {folder}")
            continue
        found = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        print(f"  Found {len(found)} images in {folder.relative_to(PROJECT_ROOT)}")
        files.extend(found)
    return files


def copy_images(files: list, dst: Path, n: int, label: str):
    dst.mkdir(parents=True, exist_ok=True)

    random.shuffle(files)
    selected = files[:n]

    print(f"\nCopying {len(selected)} {label} images → {dst.name}/")

    copied  = 0
    skipped = 0

    for src_file in selected:
        dst_file = dst / src_file.name

        # Avoid overwriting — add prefix if name collision
        if dst_file.exists():
            dst_file = dst / f"c_{src_file.name}"
        if dst_file.exists():
            skipped += 1
            continue

        shutil.copy2(src_file, dst_file)
        copied += 1

    print(f"  Done — copied {copied}, skipped {skipped} duplicates")


if __name__ == "__main__":
    random.seed(SEED)

    print("=" * 55)
    print(f"CIFAKE Dataset Organizer")
    print(f"Copying up to {N_IMAGES} images per class")
    print("=" * 55)

    # Collect all available images
    print("\nScanning AI (FAKE) sources:")
    ai_files   = collect_images(SOURCE_DIRS["ai"])

    print("\nScanning Real (REAL) sources:")
    real_files = collect_images(SOURCE_DIRS["real"])

    print(f"\nTotal available — AI: {len(ai_files)}, Real: {len(real_files)}")

    if not ai_files or not real_files:
        print("[ERROR] No images found. Check your folder paths.")
        exit(1)

    # Copy
    copy_images(ai_files,   DEST_AI_DIR,   N_IMAGES, "AI")
    copy_images(real_files, DEST_REAL_DIR, N_IMAGES, "Real")

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