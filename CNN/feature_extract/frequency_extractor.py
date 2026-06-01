"""
frequency_extractor.py

Every image can be decomposed into waves of different frequencies.
Low frequency = slow changes across the image, high frequency = rapid changes.
Each "signal" represents a pattern, any pattern even non obvious ones
AI-generated images tend to produce characteristic frequency signatures
either too regular or with missing sections that real photos do not exhibit.

FFT is better at capturing periodic patterns
FFT uses eulers formula on each pixel to extract the frequency from it
DCT is better at capturing compression history
"""

import cv2
import numpy as np
from scipy.fft import dctn
import torch
from typing import Tuple

# Input image dimensions (must match the CNN pipeline)
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
TARGET_SIZE  = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Number of radial frequency bins for the azimuthal spectrum summary
NUM_RADIAL_BINS   = 64

# Number of angular sectors for the azimuthal power map
NUM_AZIMUTH_SECTORS = 36

# Small constant to avoid log(0) and division by zero
LOG_EPSILON = 1.0
NORM_EPSILON = 1e-8


def extract_fft_map(img_gray: np.ndarray) -> np.ndarray:
    """
    FFT (Fast Fourier Transform) converts the entire image from pixel values into frequency components

    Applies a 2D FFT to the image and returns a log magnitude frequency map.
    Low frequencies are shifted to the center using fftshift.
    Log scale is used because frequency magnitudes span a very large range.


    """
    f       = np.fft.fft2(img_gray.astype(np.float32)) # produces the fft image
    fshift  = np.fft.fftshift(f)   # move low frequencies to center and high ones to the edge (this just works better for CNN)
    magnitude = np.log(np.abs(fshift) + LOG_EPSILON) # applies a logarithm because frequency magnitudes span a large range + extracts magnitudes from complex numbers
    return magnitude.astype(np.float32) # returns the 2d map


def extract_dct_map(img_gray: np.ndarray) -> np.ndarray:
    """
    Applies a 2D DCT (Discrete Cosine Transform) to the image.
    Similar to FFT but uses cosines instead of complex exponentials.
    Used to probe compression history and reveal frequency characteristics.
    AI images often show distinctive DCT patterns.

    DCT produces consine waves, keeping strong low frequency ways and discarding weak high frequencies.
    """
    dct     = dctn(img_gray.astype(np.float32), norm='ortho') # Applies dctn to the entire image
    dct_log = np.log(np.abs(dct) + LOG_EPSILON) # same reason in FFT
    return dct_log.astype(np.float32)


def extract_radial_spectrum(img_gray: np.ndarray,num_bins: int = NUM_RADIAL_BINS) -> np.ndarray:
    """
    Summarizes the FFT by averaging power at each distance from the center,
    producing a 1D radial power profile.
    Returns an array of length num_bins (bin 0 = center, bin N-1 = edge).

    This gives an evergy vs frequency summery allowing the cnn to check if the energy drops off naturally
    """

    H, W = img_gray.shape
    f        = np.fft.fft2(img_gray.astype(np.float32))
    fshift   = np.fft.fftshift(f)
    power    = np.abs(fshift) ** 2

    cy, cx   = H // 2, W // 2
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    dist     = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    max_dist = min(cy, cx)

    # Assign each frequency point to one of num_bins radial rings
    bins        = np.linspace(0, max_dist, num_bins + 1)
    bin_indices = np.digitize(dist.ravel(), bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    power_flat  = power.ravel()

    radial_power = np.zeros(num_bins, dtype=np.float32)
    np.add.at(radial_power, bin_indices, power_flat)
    counts       = np.bincount(bin_indices, minlength=num_bins).astype(np.float32)
    radial_power /= np.maximum(counts, 1)

    # Log compress and normalize to [0, 1]
    radial_power  = np.log(radial_power + LOG_EPSILON)
    radial_power /= (radial_power.max() + NORM_EPSILON)

    return radial_power


def extract_azimuthal_map(img_gray: np.ndarray,
                          num_angles: int = NUM_AZIMUTH_SECTORS) -> np.ndarray:
    """
    Checks whether frequency energy is evenly distributed in all directions.
    Instead of looking at raw frequency content, it asks is the frequency energy evenly distributed in all directions.
    Divides the full 360 degrees into num_angles sectors and averages the power
    in each sector, then maps each pixel back to its sector's average power.

    Real photos have their energy spread out, AI images often have a directional bias. this catches this bias
    """
    H, W = img_gray.shape # gets image dimensions
    f        = np.fft.fft2(img_gray.astype(np.float32)) # same thing in fft
    fshift   = np.fft.fftshift(f) # does the same shift as fft
    power    = np.abs(fshift) ** 2

    cy, cx       = H // 2, W // 2 #extracts center pixel
    y_idx, x_idx = np.mgrid[0:H, 0:W] # gives the cordinates of each pixel without a loop
    angle        = np.arctan2(y_idx - cy, x_idx - cx)  # calculates the angles relative to the center of every FFT output.  angle in [-pi, pi]

    angle_bins  = np.linspace(-np.pi, np.pi, num_angles + 1) # divides the 360 dagrees into 36 sectors of 10 dagress each
    bin_indices = np.digitize(angle.ravel(), angle_bins) - 1 # flatten into a 1d rig and find witch sector each angle falls into
    bin_indices = np.clip(bin_indices, 0, num_angles - 1) # prevents overshoots
    power_flat  = power.ravel()

    az_power = np.zeros(num_angles, dtype=np.float32) # creates teh array
    np.add.at(az_power, bin_indices, power_flat)
    counts   = np.bincount(bin_indices, minlength=num_angles).astype(np.float32) # count up to average
    az_power /= np.maximum(counts, 1) # averages out each one of the 10 degrees, ending up with 10 numbers summering energy distribution

    # Map each pixel back to its sector's average power to produce a 2D spatial map
    azimuthal_map  = az_power[bin_indices].reshape(H, W) # turns the summery back into a 2D map
    azimuthal_map  = np.log(azimuthal_map + LOG_EPSILON)
    azimuthal_map /= (azimuthal_map.max() + NORM_EPSILON)

    return azimuthal_map.astype(np.float32)


def normalize_channel(ch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalization for a single channel."""
    return (ch - ch.mean()) / (ch.std() + NORM_EPSILON)


def build_frequency_tensor(image_path: str,target_size: tuple = TARGET_SIZE) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Loads an image, computes FFT, DCT, and azimuthal frequency maps,
    stacks them into a (3, H, W) tensor, and returns the tensor along
    with a 1D radial spectrum for visualization purposes.
    """
    H, W = target_size

    # Load in BGR (OpenCV default) and convert to grayscale
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (W, H)).astype(np.float32)

    fft_map   = extract_fft_map(img_gray)
    dct_map   = extract_dct_map(img_gray)
    az_map    = extract_azimuthal_map(img_gray)
    radial_1d = extract_radial_spectrum(img_gray)  # only used for visualization

    # Stack the three maps into a single (3, H, W) tensor, similar to RGB channels
    channels = [
        normalize_channel(fft_map),
        normalize_channel(dct_map),
        normalize_channel(az_map),
    ] # All three maps get normalized
    freq_tensor = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)

    return freq_tensor, radial_1d


if __name__ == "__main__":
    path   = "../../images/ai_imagesRandom/ai_000.jpg"
    tensor, radial = build_frequency_tensor(path)
    print("Frequency tensor shape:", tensor.shape)    # (3, 224, 224)
    print("Radial spectrum shape: ", radial.shape)    # (64,)
    print("Channel means:", tensor.mean(dim=[1, 2]))  # approx 0
    print("Channel stds: ", tensor.std(dim=[1, 2]))   # approx 1
