"""
test_pipeline.py
----------------
End-to-end test. Takes an image, runs all 4 modules, prints the verdict.

Place at:  imageMetadata/test_pipeline.py

Run from project root:
  python test_pipeline.py                                     # default image, general mode
  python test_pipeline.py path/to/image.jpg                  # general mode
  python test_pipeline.py path/to/image.jpg face             # face mode
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CNN.feature_extract.inference    import run_inference
from CNN.feature_extract.score_fusion import fuse_scores

# ---------------------------------------------------------------------------
# TEST IMAGE — change this to test a different image
# ---------------------------------------------------------------------------
TEST_IMAGE = PROJECT_ROOT / "images" / "testReal" / "fake" / "fake_1.jpg"
# ---------------------------------------------------------------------------

image_path = sys.argv[1] if len(sys.argv) > 1 else str(TEST_IMAGE)
image_type = sys.argv[2] if len(sys.argv) > 2 else "general"

print(f"\nImage      : {image_path}")
print(f"Image type : {image_type}")
print("-" * 50)

# ── Run inference ─────────────────────────────────────────────────────────
raw = run_inference(image_path, image_type=image_type)

print(f"PRNU score     : {raw['prnu_score']}")
print(f"ELA  score     : {raw['ela_score']}")
print(f"Freq score     : {raw['freq_score']}")
print(f"Metadata score : {raw['metadata_score']}  ({raw['metadata_reason']})")

# ── Fuse scores ───────────────────────────────────────────────────────────
fusion = fuse_scores(
    prnu_score      = raw["prnu_score"],
    ela_score       = raw["ela_score"],
    freq_score      = raw["freq_score"],
    metadata_score  = raw["metadata_score"],
    metadata_format = raw["metadata_format"],
    metadata_reason = raw["metadata_reason"],
    image_type      = image_type,
)

print("-" * 50)
print(f"Verdict     : {fusion['verdict']}")
print(f"Confidence  : {fusion['confidence']}%")
print(f"Final score : {fusion['final_score']}  (0=real, 1=AI)")
print(f"Weights     : {fusion['weights_used']}")
print()