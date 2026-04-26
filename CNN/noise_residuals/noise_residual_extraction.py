"""
PRNU  (Photo Response Non-Uniformity) extraction




PRNU patterns are small imprefections that are caused by camera sensors. ai image do not have these imperfections thus we can detect them
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import wiener

TARGET_SIZE = (224, 224)


#separates signal from noise by looking at local variance
def _stable_wiener(arr: np.ndarray, size: tuple) -> np.ndarray:
    """
    Wrapper around scipy wiener that prevents divide-by-zero.
    Adds a tiny noise floor to ensure local variance is never exactly zero.
    Also clips extreme outliers before and after filtering.
    """
    # Clip extreme values before filtering
    arr = np.clip(arr, -1e4, 1e4)

    # Add tiny noise floor so local variance is never zero
    noise_floor = np.random.default_rng(seed=0).normal(
        0, 1e-6, arr.shape
    ).astype(np.float32)
    arr_stable = arr + noise_floor

    filtered = wiener(arr_stable, size)

    # Clip again after filtering to remove any remaining outliers
    filtered = np.clip(filtered, -1e4, 1e4)
    return filtered.astype(np.float32)


def extract_prnu_residual(img_gray: np.ndarray) -> np.ndarray:
    """
    Extracts the PRNU noise residual from a grayscale image.

    img_gray - float32 grayscale array, already resized to TARGET_SIZE
    returns  : PRNU map, same spatial size, float32
    """

    #Applies a 2D wavelet decomposition using 6 db
    #this transformation decomposes the image to low frequency components and high
    #using 3 levels (doing the decomposition 3 times)
    wavelet, levels = 'db8', 3
    coeffs = pywt.wavedec2(img_gray, wavelet, level=levels)


    # Applies wiener filter to high frequency bands
    # we do this to denise the detail components while preserving the structure
    denoised_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        denoised_detail = []
        for band in detail_level:
            denoised_detail.append(_stable_wiener(band, (3, 3)))
        denoised_coeffs.append(tuple(denoised_detail))

    # Reconstructs the denoised image
    # (crop needed because of unstable output)
    denoised = pywt.waverec2(denoised_coeffs, wavelet)
    denoised = denoised[:img_gray.shape[0], :img_gray.shape[1]]

    # Compute residual and normalize by local image intensity
    # we do this by subtracting the denoised image from the orignal and then divided by the image intesnity(im_gray + esp to remove brightness on the noise pattern)
    #(denoised simpyly means the image without unwanted noise)
    residual = img_gray - denoised
    eps      = 1e-8

    # Clean up
    # 1 remove the mean (subtracting the avrage from every pixle)
    prnu     = residual / (img_gray + eps)
    prnu    -= prnu.mean()
    # 2 apply another winer filter to the full prnu map
    prnu = _stable_wiener(prnu, (5, 5))
    # 3 subtract a gaussian blur of itself to remove and remaining low frequency content that leaked
    #gaussian blur avrages every pixle with its neighbors 9used to remove high frequency information)
    prnu -= cv2.GaussianBlur(prnu, (9, 9), 0)

    return prnu.astype(np.float32)


def normalize_channel(ch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalization for a single channel. step to prepare for CNN"""
    return (ch - ch.mean()) / (ch.std() + 1e-8)
    # 1e - 8 used to prevent divistion by zero




# Standalone visualization (only use for visual not the actual CNN)

if __name__ == "__main__":
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    default_path = PROJECT_ROOT / "images" / "testReal" / "real" / "real_1.jpg"

    img = cv2.imread(str(default_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {default_path}")

    img = img.astype(np.float32)
    H, W = TARGET_SIZE
    img = cv2.resize(img, (W, H))

    prnu     = extract_prnu_residual(img)
    residual = img - cv2.GaussianBlur(img, (9, 9), 0)

    fft       = np.fft.fftshift(np.fft.fft2(prnu))
    magnitude = np.log(np.abs(fft) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("PRNU Estimate")
    plt.imshow(prnu, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("FFT of PRNU")
    plt.imshow(magnitude, cmap='inferno')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Residual")
    plt.imshow(residual, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()