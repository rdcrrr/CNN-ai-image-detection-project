"""
noise_residual_extraction.py

PRNU (Photo Response Non-Uniformity) extraction.

Concept:
    Camera sensors have tiny manufacturing imperfections that create a unique
    noise pattern in every photo. This pattern is called the PRNU fingerprint.
    AI-generated images do not go through a real camera sensor, so they lack
    this fingerprint. Detecting its absence is a strong signal of AI generation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import wiener

# Input image dimensions (must match the CNN pipeline)
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
TARGET_SIZE  = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Wavelet decomposition settings
WAVELET_TYPE   = 'db8'   # Daubechies 8 wavelet
WAVELET_LEVELS = 3       # number of decomposition levels

# Wiener filter window sizes
DETAIL_WIENER_SIZE = (3, 3)   # applied to each high-frequency band
PRNU_WIENER_SIZE   = (5, 5)   # applied to the final PRNU map

# Gaussian blur kernel for removing residual low-frequency content from PRNU
GAUSSIAN_KERNEL_SIZE = (9, 9)

# Clipping range to remove extreme outlier values before/after Wiener filtering
WIENER_CLIP_MIN = -1e4
WIENER_CLIP_MAX =  1e4

# Tiny noise floor added to prevent Wiener filter's local variance from being exactly zero
WIENER_NOISE_STD = 1e-6

# Small constant to prevent division by zero during PRNU normalization
PRNU_NORM_EPSILON = 1e-8

# Small constant to prevent division by zero during channel normalization
CHANNEL_NORM_EPSILON = 1e-8


def _stable_wiener(arr: np.ndarray, size: tuple) -> np.ndarray:
    """
    Wrapper around scipys Wiener filter that prevents divide by zero errors.
    Adds a tiny noise floor so local variance is never exactly zero,
    and clips extreme values before and after filtering.

    This filter works by first, looking for a local variance (how much do neighboring pixels vary from each other?)
    then applies smoothing to the area
    if the area is flat it could result in a zero
    """
    arr  = np.clip(arr, WIENER_CLIP_MIN, WIENER_CLIP_MAX) # Limit the values to a certain threshold
    noise_floor = np.random.default_rng(seed=0).normal(0, WIENER_NOISE_STD, arr.shape,).astype(np.float32) # Adds a tiny random amount of noise
    # this is the wrapper that prevents divide by zero
    arr_stable = arr + noise_floor
    filtered   = wiener(arr_stable, size)
    filtered   = np.clip(filtered, WIENER_CLIP_MIN, WIENER_CLIP_MAX) # Limit extreme values again
    return filtered.astype(np.float32)


def extract_prnu_residual(img_gray: np.ndarray) -> np.ndarray:
    """
    Extracts the PRNU noise residual from a grayscale image.

    img_gray: float32 grayscale array already resized to TARGET_SIZE
    returns: PRNU map of the same spatial size, float32

    Steps:
        1 Wavelet decomposition (WAVELET_LEVELS levels of WAVELET_TYPE)
        2 Wiener filter on high frequency bands to denoise while preserving structure (to smooth out random noise)
        3 Reconstruct the denoised image
        4 Compute residual = original - denoised, normalized by local intensity
        5 Post-process: remove mean, apply full-map Wiener filter, remove
           residual low-frequency content with a Gaussian blur subtraction
    """
    # Wavelet decomposition splits the image into low-frequency (approximation)
    # and high-frequency (detail) components across WAVELET_LEVELS levels
    # Wavelet works by applying filters to the image, that make the noise "pop"
    # Uses db8, a smooth shape that represents gradual intensity, clearly separates image content from high frequency noise
    coeffs = pywt.wavedec2(img_gray, WAVELET_TYPE, level=WAVELET_LEVELS) # do this 3 times (levels = the amount of times you peel image layers away)

    # coeffs[0] is the low-frequency base (sort of a blurry version of the image), coeffs[1], [2], [3] are the high frequency detail layers.

    # Apply Wiener filter to each high frequency band to remove image content
    # while preserving the faint sensor noise pattern
    denoised_coeffs = [coeffs[0]]   # keep the low-frequency approximation unchanged
    for detail_level in coeffs[1:]:
        denoised_detail = [_stable_wiener(band, DETAIL_WIENER_SIZE) for band in detail_level]
        denoised_coeffs.append(tuple(denoised_detail))


    # Reconstruct the denoised image (crop removes boundary artifacts from wavelet reconstruction) for later subtraction

    # prevents the output from being one pixel too large
    denoised = pywt.waverec2(denoised_coeffs, WAVELET_TYPE)
    denoised = denoised[:img_gray.shape[0], :img_gray.shape[1]]

    # Residual normalized by local image intensity to remove brightness dependent noise
    # subtracting the diagnosed image from the original leaving us mostly with PRNU
    residual = img_gray - denoised
    prnu     = residual / (img_gray + PRNU_NORM_EPSILON) # divide to normalize the noise

    # Post-processing: remove mean, apply Wiener filter, remove residual low frequencies
    prnu -= prnu.mean() # Center the pattern to around zero, this removes positive / negative biases from diagnoses
    # preventing positive or negative biases accurately shows in what region contains what amount of noise
    prnu  = _stable_wiener(prnu, PRNU_WIENER_SIZE) # Reduces any remaining data other then the PRNU residuals
    prnu -= cv2.GaussianBlur(prnu, GAUSSIAN_KERNEL_SIZE, 0) # Remove any low frequency content that leaked into the PRNU map
    # Gaussian blur averages each pixel with its neighbours, helps capture the low frequency content
    return prnu.astype(np.float32)


def normalize_channel(ch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalization for a single channel Prepares data for the CNN."""
    return (ch - ch.mean()) / (ch.std() + CHANNEL_NORM_EPSILON) # CHANNEL_NORM_EPSILON prevents divide by zero


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    default_path = PROJECT_ROOT / "images" / "testReal" / "real" / "real_1.jpg"

    img = cv2.imread(str(default_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {default_path}")

    img = img.astype(np.float32)
    H, W = TARGET_SIZE
    img  = cv2.resize(img, (W, H))

    prnu      = extract_prnu_residual(img)
    residual  = img - cv2.GaussianBlur(img, GAUSSIAN_KERNEL_SIZE, 0)
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
