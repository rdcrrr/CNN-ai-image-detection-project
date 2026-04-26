"""
add_diverse_data.py
-------------------
Adds images from the labeled diverse dataset into the correct
general and face training folders.

General AI  <- DALL-E, Midjourney, Stable Diffusion, NeuralTextures
Face AI     <- DeepFaceLab, Face2Face, FaceShifter, FaceSwap, StyleGAN
Real        <- added to both general and face real folders

Place at:  imageMetadata/add_diverse_data.py
Run with:  python add_diverse_data.py
"""

import shutil
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DIVERSE_ROOT = PROJECT_ROOT / "images" / "diverseData"

# Which folders go where
GENERAL_AI_SOURCES = [
    DIVERSE_ROOT / "DALL-E",
    DIVERSE_ROOT / "Midjourney",
    DIVERSE_ROOT / "Stable Diffusion",
    DIVERSE_ROOT / "NeuralTextures",
]

FACE_AI_SOURCES = [
    DIVERSE_ROOT / "DeepFaceLab",
    DIVERSE_ROOT / "Face2Face",
    DIVERSE_ROOT / "FaceShifter",
    DIVERSE_ROOT / "FaceSwap",
    DIVERSE_ROOT / "StyleGAN",
]

REAL_SOURCE = DIVERSE_ROOT / "Real"

# Destinations
DEST_GENERAL_AI   = PROJECT_ROOT / "images" / "training" / "general" / "ai"
DEST_GENERAL_REAL = PROJECT_ROOT / "images" / "training" / "general" / "real"
DEST_FACE_AI      = PROJECT_ROOT / "images" / "training" / "faces" / "ai"
DEST_FACE_REAL    = PROJECT_ROOT / "images" / "training" / "faces" / "real"

# How many images to add per class
N_GENERAL_AI = 3000
N_FACE_AI    = 3000
N_REAL_EACH  = 2000   # added to both general and face real folders

SEED = 42
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(folders: list) -> list:
    files = []
    for folder in folders:
        if not folder.exists():
            print(f"  [WARN] Not found: {folder.name}")
            continue
        found = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        print(f"  {len(found):>5} images  ←  {folder.name}")
        files.extend(found)
    return files


def copy_images(files: list, dst: Path, n: int, label: str):
    dst.mkdir(parents=True, exist_ok=True)
    random.shuffle(files)
    selected = files[:n]

    print(f"\nCopying {len(selected)} {label} → {dst.relative_to(PROJECT_ROOT)}")
    copied = skipped = 0

    for src in selected:
        dst_file = dst / f"div_{src.name}"
        if dst_file.exists():
            skipped += 1
            continue
        shutil.copy2(src, dst_file)
        copied += 1

    print(f"  Done — copied {copied}, skipped {skipped}")


def count_folder(folder: Path) -> int:
    return len([p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


if __name__ == "__main__":
    random.seed(SEED)

    print("=" * 55)
    print("DIVERSE DATASET ORGANIZER")
    print("=" * 55)

    # ── Collect ───────────────────────────────────────────────────────────
    print("\nGeneral AI sources:")
    general_ai_files = collect_images(GENERAL_AI_SOURCES)

    print("\nFace AI sources:")
    face_ai_files = collect_images(FACE_AI_SOURCES)

    print("\nReal source:")
    real_files = collect_images([REAL_SOURCE])

    print(f"\nTotal available:")
    print(f"  General AI : {len(general_ai_files)}")
    print(f"  Face AI    : {len(face_ai_files)}")
    print(f"  Real       : {len(real_files)}")

    if not general_ai_files and not face_ai_files:
        print("[ERROR] No images found. Check folder paths.")
        exit(1)

    # ── Copy ──────────────────────────────────────────────────────────────
    copy_images(general_ai_files, DEST_GENERAL_AI,   N_GENERAL_AI, "General AI images")
    copy_images(face_ai_files,    DEST_FACE_AI,      N_FACE_AI,    "Face AI images")
    copy_images(real_files,       DEST_GENERAL_REAL, N_REAL_EACH,  "Real → General")
    copy_images(real_files,       DEST_FACE_REAL,    N_REAL_EACH,  "Real → Face")

    # ── Final counts ──────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Training folder totals:")
    print(f"  general/ai   : {count_folder(DEST_GENERAL_AI)}")
    print(f"  general/real : {count_folder(DEST_GENERAL_REAL)}")
    print(f"  faces/ai     : {count_folder(DEST_FACE_AI)}")
    print(f"  faces/real   : {count_folder(DEST_FACE_REAL)}")
    print(f"{'=' * 55}")