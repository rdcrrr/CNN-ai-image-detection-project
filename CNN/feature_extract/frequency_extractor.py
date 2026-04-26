import cv2
import numpy as np
from scipy.fft import dctn
import torch
from typing import Tuple

"""
Frequency extraction

the concept - every image can be decomposed into waves of different frequencies. low frequency = slow changes, high freq = rapid changes.
ai teds to produce characteristic frequency signatures. either to regual or with missing sections

"""


TARGET_SIZE = (224, 224)

# Fast fourier transformation
# converts the image from pixel values into frequency
def extract_fft_map(img_gray: np.ndarray) -> np.ndarray:

    # Applies a 2D FFT to the entire image
    f = np.fft.fft2(img_gray.astype(np.float32))

    # By defualt fft puts low frequencies in corners and high in center. this flips this default
    fshift = np.fft.fftshift(f)
    # Takes the absolute values, then applies a logarithum because frequency magnitudes spam a large range
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude.astype(np.float32)


# DCT - discrete cosine transform
# FFT but using cosine stead of exponential.
# we use this to probe the compression history and revel frequency characteristics of the image
def extract_dct_map(img_gray: np.ndarray) -> np.ndarray:

    # Applies 2D DCT to the whole image and takes the log magnitude
    dct = dctn(img_gray.astype(np.float32), norm='ortho')
    dct_log = np.log(np.abs(dct) + 1)
    return dct_log.astype(np.float32)
    # Ai images often show distinctive DCT patters

# Summerizes FFT. averaging power at each distance from the center to map into a 1D profile
def extract_radial_spectrum(img_gray: np.ndarray,
                             num_bins: int = 64) -> np.ndarray:
    H, W = img_gray.shape
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2

    # Calculates distance from center
    cy, cx = H // 2, W // 2
    # Creates two arrows one index of every pxile the other column index
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    max_dist = min(cy, cx)

    # Divides the frequency space into 64 rings and assigns each frequency point to its ring
    bins = np.linspace(0, max_dist, num_bins + 1)
    bin_indices = np.digitize(dist.ravel(), bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    power_flat  = power.ravel()

    # Log compression and normalization
    # gives us a 64 element array describing image energy distribution (center ring 0 edges 53)
    radial_power = np.zeros(num_bins, dtype=np.float32)
    np.add.at(radial_power, bin_indices, power_flat)
    counts = np.bincount(bin_indices, minlength=num_bins).astype(np.float32)
    radial_power /= np.maximum(counts, 1)

    radial_power = np.log(radial_power + 1)
    radial_power /= (radial_power.max() + 1e-8)
    return radial_power


# Checks whether frequency is event distributed in all directions
def extract_azimuthal_map(img_gray: np.ndarray,
                           num_angles: int = 36) -> np.ndarray:
    H, W = img_gray.shape
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2

    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.mgrid[0:H, 0:W]

    # Calculates angle of each point relative to center (from -pie to pie)
    angle = np.arctan2(y_idx - cy, x_idx - cx)

    # Divides the full 360 into 36 sectors and avrages the power of each
    angle_bins  = np.linspace(-np.pi, np.pi, num_angles + 1)
    bin_indices = np.digitize(angle.ravel(), angle_bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_angles - 1)
    power_flat  = power.ravel()

    az_power = np.zeros(num_angles, dtype=np.float32)
    np.add.at(az_power, bin_indices, power_flat)
    counts = np.bincount(bin_indices, minlength=num_angles).astype(np.float32)
    az_power /= np.maximum(counts, 1)

    # Converts the 36 bit summary back into a 2D map
    # (each pixles value is the average power of its direction, creating a spatial map the CNN can process)
    azimuthal_map = az_power[bin_indices].reshape(H, W)
    azimuthal_map = np.log(azimuthal_map + 1)
    azimuthal_map /= (azimuthal_map.max() + 1e-8)
    return azimuthal_map.astype(np.float32)


def normalize_channel(ch: np.ndarray) -> np.ndarray:
    return (ch - ch.mean()) / (ch.std() + 1e-8)

# This is only used for visuals
def build_frequency_tensor(image_path: str,
                            target_size: tuple = TARGET_SIZE
                            ) -> Tuple[torch.Tensor, np.ndarray]:
    H, W = target_size
    # Converts to grayscale
    # Color information is not needed
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (W, H)).astype(np.float32)

    fft_map   = extract_fft_map(img_gray)
    dct_map   = extract_dct_map(img_gray)
    az_map    = extract_azimuthal_map(img_gray)
    # only for visual
    radial_1d = extract_radial_spectrum(img_gray)

    # Stacks the three channels into a single tensor
    # (simular to RGB values)
    channels = [
        normalize_channel(fft_map),
        normalize_channel(dct_map),
        normalize_channel(az_map),
    ]

    freq_tensor = torch.tensor(
        np.stack(channels, axis=0), dtype=torch.float32
    )
    return freq_tensor, radial_1d


if __name__ == "__main__":
    path = "../../images/ai_imagesRandom/ai_000.jpg"
    tensor, radial = build_frequency_tensor(path)
    print("Frequency tensor shape:", tensor.shape)   # (3, 224, 224)
    print("Radial spectrum shape: ", radial.shape)   # (64,)
    print("Channel means:", tensor.mean(dim=[1, 2])) # ≈ 0
    print("Channel stds: ", tensor.std(dim=[1, 2]))  # ≈ 1