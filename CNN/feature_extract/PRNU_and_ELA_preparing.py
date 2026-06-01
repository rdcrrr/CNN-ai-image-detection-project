"""
PRNU_and_ELA_preparing.py

Bridge between the feature extractors and the CNN pipeline.
Prepares PRNU and ELA feature maps as tensors ready for training or inference.

1 shapes everything to what the CNN expects
2 converts into pytorch arrays
3 normalizes the value ranges

pretty much returns a much cleaner output ready for the CNN
"""

import cv2
import numpy as np
import torch

from ..noise_residuals.noise_residual_extraction import extract_prnu_residual, normalize_channel
from ..noise_residuals.ELA_extraction import run_ela

# Input image dimensions (must match the CNN pipeline)
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
TARGET_SIZE  = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Default JPEG quality used for ELA re-compression
ELA_DEFAULT_QUALITY = 75


def build_prnu_tensor(image_path: str,
                      target_size: tuple = TARGET_SIZE) -> torch.Tensor:
    """
    Loads an image, extracts its PRNU residual, and returns a (2, H, W) tensor
    containing the normalized grayscale image and the PRNU map as two channels.
    Both channels are fed to the CNN so it can learn the relationship between them.
    """
    H, W = target_size

    # Load in BGR (OpenCV default) and convert to grayscale
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (W, H)).astype(np.float32)

    prnu = extract_prnu_residual(img_gray)

    channels = [
        normalize_channel(img_gray),
        normalize_channel(prnu),
    ]
    return torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)


def build_ela_tensor(image_path: str,
                     target_size: tuple = TARGET_SIZE,
                     ela_quality: int = ELA_DEFAULT_QUALITY) -> torch.Tensor:
    """
    Runs ELA on an image and returns a (1, H, W) tensor of the normalized ELA map.
    """
    ela = run_ela(image_path, quality=ela_quality)

    H, W = target_size
    if ela.shape != (H, W):
        import cv2 as _cv2
        ela = _cv2.resize(ela, (W, H))

    channel = normalize_channel(ela)[np.newaxis, ...]  # shape: (1, H, W)
    return torch.tensor(channel, dtype=torch.float32)


if __name__ == "__main__":
    path   = "../../images/ai_imagesRandom/ai_000.jpg"
    prnu_t = build_prnu_tensor(path)
    ela_t  = build_ela_tensor(path)

    print("PRNU tensor shape :", prnu_t.shape)             # (2, 224, 224)
    print("PRNU channel means:", prnu_t.mean(dim=[1, 2]))  # approx 0
    print("PRNU channel stds :", prnu_t.std(dim=[1, 2]))   # approx 1
    print()
    print("ELA tensor shape  :", ela_t.shape)              # (1, 224, 224)
    print("ELA channel mean  :", ela_t.mean(dim=[1, 2]))   # approx 0
    print("ELA channel std   :", ela_t.std(dim=[1, 2]))    # approx 1
