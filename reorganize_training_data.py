"""
reorganize_training_data.py
---------------------------
Step 1: Audits aiImagesLarge and real_imagesLarge, counting
        how many face images (prefix "face_") vs general images exist.

Step 2: Asks for confirmation, then reorganizes into:
  images/training/general/ai/
  images/training/general/real/
  images/training/faces/ai/
  images/training/faces/real/

Face images are identified by the "face_" prefix added by add_face_data.py.
All other images are treated as general.

Place at:  imageMetadata/reorganize_training_data.py
Run with:  python reorganize_training_data.py
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_AI_DIR   = PROJECT_ROOT / "images" / "aiImagesLarge"
SRC_REAL_DIR = PROJECT_ROOT / "images" / "real_imagesLarge"

DEST_GENERAL_AI   = PROJECT_ROOT / "images" / "training" / "general" / "ai"
DEST_GENERAL_REAL = PROJECT_ROOT / "images" / "training" / "general" / "real"
DEST_FACE_AI      = PROJECT_ROOT / "images" / "training" / "faces" / "ai"
DEST_FACE_REAL    = PROJECT_ROOT / "images" / "training" / "faces" / "real"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def audit_folder(folder: Path) -> tuple[list, list]:
    """Returns (face_files, general_files) from a folder."""
    all_files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    face    = [f for f in all_files if f.name.startswith("face_")]
    general = [f for f in all_files if not f.name.startswith("face_")]
    return face, general


def move_images(files: list, dst: Path, label: str):
    dst.mkdir(parents=True, exist_ok=True)
    moved = skipped = 0
    for src in files:
        dst_file = dst / src.name
        if dst_file.exists():
            skipped += 1
            continue
        shutil.move(str(src), dst_file)
        moved += 1
    print(f"  {label}: moved {moved}, skipped {skipped}")


if __name__ == "__main__":
    print("=" * 55)
    print("TRAINING DATA AUDIT")
    print("=" * 55)

    # ── Audit ────────────────────────────────────────────────
    print(f"\nScanning {SRC_AI_DIR.name}...")
    ai_faces, ai_general = audit_folder(SRC_AI_DIR)

    print(f"  Face images    (face_ prefix) : {len(ai_faces)}")
    print(f"  General images               : {len(ai_general)}")

    print(f"\nScanning {SRC_REAL_DIR.name}...")
    real_faces, real_general = audit_folder(SRC_REAL_DIR)

    print(f"  Face images    (face_ prefix) : {len(real_faces)}")
    print(f"  General images               : {len(real_general)}")

    print(f"\n{'─' * 55}")
    print(f"Summary:")
    print(f"  General training — AI: {len(ai_general)}, Real: {len(real_general)}")
    print(f"  Face training    — AI: {len(ai_faces)},   Real: {len(real_faces)}")
    print(f"{'─' * 55}")

    if not any([ai_faces, ai_general, real_faces, real_general]):
        print("[ERROR] No images found. Check folder paths.")
        exit(1)

    # ── Confirm ───────────────────────────────────────────────
    print(f"""
Destination folders:
  images/training/general/ai/   ← {len(ai_general)} images
  images/training/general/real/ ← {len(real_general)} images
  images/training/faces/ai/     ← {len(ai_faces)} images
  images/training/faces/real/   ← {len(real_faces)} images
""")
    confirm = input("Type YES to reorganize: ").strip()
    if confirm != "YES":
        print("Aborted.")
        exit(0)

    # ── Move ──────────────────────────────────────────────────
    print("\nMoving files...")
    move_images(ai_general,   DEST_GENERAL_AI,   "General AI")
    move_images(real_general, DEST_GENERAL_REAL, "General Real")
    move_images(ai_faces,     DEST_FACE_AI,      "Face AI")
    move_images(real_faces,   DEST_FACE_REAL,    "Face Real")

    print(f"\n{'=' * 55}")
    print("Done! Training data reorganized.")
    print(f"{'=' * 55}")