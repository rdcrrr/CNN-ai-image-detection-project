import cv2
import numpy as np
import torch

#import our extractions from before
from ..noise_residuals.noise_residual_extraction import extract_prnu_residual, normalize_channel
from ..noise_residuals.ELA_extraction import run_ela


"""
A birdge to the CNN program, used to prepare PRNU and ELA extractions for the CNN training

"""
TARGET_SIZE = (224, 224)



def build_prnu_tensor(image_path: str,
                      target_size: tuple = TARGET_SIZE) -> torch.Tensor:
    H, W = target_size

    # Loads images in blue green red order (instead of rgb) we do this because open CV
    # then convert to grayscale
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (W, H)).astype(np.float32)

    # Call prnu extraction
    prnu = extract_prnu_residual(img_gray)  # reuses the function from noise_residual_extraction.py

    # Create a two channel tensor
    # feed both the prnu map and the grayscale to the CNN
    # this is to allow the CNN to learn the relationship between them
    channels = [
        normalize_channel(img_gray),
        normalize_channel(prnu),
    ]
    return torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)



# ELA pipeline  →  returns (1, H, W) tensor

def build_ela_tensor(image_path: str,
                     target_size: tuple = TARGET_SIZE,
                     ela_quality: int = 75) -> torch.Tensor:
    ela = run_ela(image_path, quality=ela_quality)  # reuses the function from ELA_extraction.py

    # run_ela returns (H, W) at whatever size the image is — resize to target
    H, W = target_size
    if ela.shape != (H, W):
        import cv2 as _cv2
        ela = _cv2.resize(ela, (W, H))


    channel = normalize_channel(ela)[np.newaxis, ...]  # (1, H, W)
    return torch.tensor(channel, dtype=torch.float32)



# Smoke test

if __name__ == "__main__":
    path = "../../images/ai_imagesRandom/ai_000.jpg"

    prnu_t = build_prnu_tensor(path)
    ela_t  = build_ela_tensor(path)

    print("PRNU tensor shape :", prnu_t.shape)             # → (2, 224, 224)
    print("PRNU channel means:", prnu_t.mean(dim=[1, 2]))  # ≈ 0
    print("PRNU channel stds :", prnu_t.std(dim=[1, 2]))   # ≈ 1
    print()
    print("ELA tensor shape  :", ela_t.shape)              # → (1, 224, 224)
    print("ELA channel mean  :", ela_t.mean(dim=[1, 2]))   # ≈ 0
    print("ELA channel std   :", ela_t.std(dim=[1, 2]))    # ≈ 1