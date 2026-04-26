"""
sanity_check.py
---------------
Tests each CNN on a small sample of known real and AI images.
Helps diagnose if a model has collapsed (always predicting one class).

Place at:  imageMetadata/sanity_check.py
Run with:  python sanity_check.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CNN.feature_extract.inference import (
    _safe_cnn_score, PRNUNet, ELANet, FreqNet,
    PRNU_MODEL_PATH, ELA_MODEL_PATH, FREQ_MODEL_PATH,
    get_device
)
from CNN.feature_extract.PRNU_and_ELA_preparing import build_prnu_tensor, build_ela_tensor
from CNN.feature_extract.frequency_extractor import build_frequency_tensor

# ---------------------------------------------------------------------------
# CONFIGURE — point these at folders with known real and AI images
# ---------------------------------------------------------------------------
REAL_FOLDER = PROJECT_ROOT / "images" / "testReal" / "real"
AI_FOLDER   = PROJECT_ROOT / "images" / "testReal" / "fake"
N_SAMPLES   = 10   # how many images to test from each folder
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sample_images(folder: Path, n: int) -> list:
    files = [p for p in folder.rglob("*")
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return files[:n]


def test_module(name, model_class, path, tensor_fn, real_files, ai_files):
    device = get_device()
    print(f"\n── {name.upper()} ──────────────────────────────")

    real_scores = []
    for f in real_files:
        s = _safe_cnn_score(name, model_class, path, tensor_fn, str(f), device)
        real_scores.append(s)
        print(f"  REAL  {f.name:<40} {s:.4f}" if s is not None else f"  REAL  {f.name:<40} FAILED")

    ai_scores = []
    for f in ai_files:
        s = _safe_cnn_score(name, model_class, path, tensor_fn, str(f), device)
        ai_scores.append(s)
        print(f"  AI    {f.name:<40} {s:.4f}" if s is not None else f"  AI    {f.name:<40} FAILED")

    valid_real = [s for s in real_scores if s is not None]
    valid_ai   = [s for s in ai_scores   if s is not None]

    if valid_real and valid_ai:
        avg_real = sum(valid_real) / len(valid_real)
        avg_ai   = sum(valid_ai)   / len(valid_ai)
        print(f"\n  Avg real score : {avg_real:.4f}  (should be close to 0)")
        print(f"  Avg AI score   : {avg_ai:.4f}  (should be close to 1)")
        if abs(avg_real - avg_ai) < 0.05:
            print("  ⚠ WARNING: model is not separating real from AI — likely collapsed")
        else:
            print("  ✓ Model is producing different scores for real vs AI")


if __name__ == "__main__":
    real_files = sample_images(REAL_FOLDER, N_SAMPLES)
    ai_files   = sample_images(AI_FOLDER,   N_SAMPLES)

    print(f"Testing with {len(real_files)} real and {len(ai_files)} AI images\n")

    test_module("prnu", PRNUNet, PRNU_MODEL_PATH, build_prnu_tensor,      real_files, ai_files)
    test_module("ela",  ELANet,  ELA_MODEL_PATH,  build_ela_tensor,       real_files, ai_files)
    test_module("freq", FreqNet, FREQ_MODEL_PATH, build_frequency_tensor, real_files, ai_files)