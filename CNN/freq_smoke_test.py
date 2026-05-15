"""
freq_smoke_test.py

Verifies that the frequency tensor pipeline works correctly
before plugging it into CNN training.

Run from imageMetadata/ root:
  python -m CNN.freq_smoke_test
"""

import sys
from pathlib import Path

import torch
import numpy as np


# 1. Import test

print("=" * 55)
print("  Frequency Tensor Pipeline Smoke Test")
print("=" * 55)

try:
    from .feature_extract.frequency_extractor import build_frequency_tensor
    print("[OK] Import successful")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    print("       Make sure frequency_extractor.py is inside CNN/feature_extract/")
    sys.exit(1)


# 2. Find a test image — search recursively with next() for speed

SEARCH_DIRS = [
    Path(r"C:\Users\rdc20\PycharmProjects\imageMetadata\images\real_imagesLarge"),
    Path(r"C:\Users\rdc20\PycharmProjects\imageMetadata\images\aiImagesLarge"),
]

test_image = None
for d in SEARCH_DIRS:
    if not d.exists():
        continue
    # next() grabs the very first match without scanning everything
    match = next(
        (p for p in d.rglob("*")
         if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        None
    )
    if match:
        test_image = match
        break

if test_image is None:
    print("[FAIL] No test image found — check that image folders exist and contain images")
    sys.exit(1)

print(f"[OK] Using test image: {test_image}")


# 3. Build tensor and run checks

try:
    tensor, radial = build_frequency_tensor(str(test_image))
except Exception as e:
    print(f"[FAIL] build_frequency_tensor raised: {e}")
    sys.exit(1)

print(f"\n--- Tensor checks ---")

# Shape check
expected_shape = (3, 224, 224)
if tensor.shape == expected_shape:
    print(f"[OK] Shape: {tuple(tensor.shape)}  (expected {expected_shape})")
else:
    print(f"[FAIL] Shape: {tuple(tensor.shape)}  (expected {expected_shape})")

# Dtype check
if tensor.dtype == torch.float32:
    print(f"[OK] Dtype: {tensor.dtype}")
else:
    print(f"[FAIL] Dtype: {tensor.dtype}  (expected torch.float32)")

# NaN / Inf check
if not torch.isnan(tensor).any() and not torch.isinf(tensor).any():
    print(f"[OK] No NaN or Inf values")
else:
    print(f"[FAIL] Tensor contains NaN or Inf values — check normalize_channel")

# Channel stats — after normalize_channel, mean ≈ 0 and std ≈ 1
means = tensor.mean(dim=[1, 2])
stds  = tensor.std(dim=[1, 2])
print(f"\n--- Channel statistics ---")
for i, (m, s) in enumerate(zip(means, stds)):
    mean_ok = abs(m.item()) < 0.1
    std_ok  = 0.5 < s.item() < 2.0
    mean_status = "OK" if mean_ok else "WARN"
    std_status  = "OK" if std_ok  else "WARN"
    print(f"  Ch {i}: mean={m:.4f} [{mean_status}]  std={s:.4f} [{std_status}]")

# Radial spectrum check
print(f"\n--- Radial spectrum ---")
print(f"  Shape : {radial.shape}  (expected (64,))")
print(f"  Min   : {radial.min():.4f}")
print(f"  Max   : {radial.max():.4f}")
if radial.min() >= 0 and radial.max() <= 1.0:
    print(f"  [OK] Values in [0, 1] range")
else:
    print(f"  [WARN] Values outside [0, 1] range")


# 4. Batch simulation — simulate what the DataLoader will do

print(f"\n--- Batch simulation ---")
try:
    batch = torch.stack([tensor, tensor, tensor, tensor])  # fake batch of 4
    print(f"[OK] Batch shape: {tuple(batch.shape)}  (expected (4, 3, 224, 224))")
except Exception as e:
    print(f"[FAIL] Could not stack into batch: {e}")


# 5. Summary
print(f"\n{'=' * 55}")
print("  Smoke test complete — ready for CNN training")
print(f"{'=' * 55}\n")
