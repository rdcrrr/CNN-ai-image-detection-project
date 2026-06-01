"""
ELA_extraction.py

Error Level Analysis (ELA) for AI image detection.

Concept:
    When a JPEG image is saved, small compression errors occur.
    If you re-compress the image and compare the two versions, the differences
    (error levels) will be small and uniform for an untouched photo.
    Edited regions or AI-generated content show higher, less uniform error levels
    because they have a different compression history.

Limitation:
    This method is less effective on images that have already been heavily compressed.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO

# Input image dimensions (must match the CNN pipeline)
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
TARGET_SIZE  = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Default JPEG re-compression quality.
# quality=75 gives a more informative ELA map than the previously used quality=95,
# which produced near-zero differences when the source was already high quality.
DEFAULT_ELA_QUALITY = 75

# Fallback max_diff value to avoid division by zero
MIN_MAX_DIFF = 1


def jpeg_recompress_in_memory(img_pil: Image.Image, quality: int = DEFAULT_ELA_QUALITY) -> Image.Image:
    """
    Re-saves a PIL image as JPEG at the given quality level and returns the
    re-compressed version without writing to disk.

    (PIL is just the way pillows python library saves images)
    """
    buffer = BytesIO() # Creates a "fake" memory file that writes the compressed JPEG bytes into RAM instead of disk to improve running speed
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0) # rewinds buffer back to the start so it can be read again (otherwise reading will start of the end of the file)
    return Image.open(buffer)


def run_ela(image_path: str, quality: int = DEFAULT_ELA_QUALITY) -> np.ndarray:
    """
    Runs ELA on an image and returns a (H, W) float32 grayscale error map
    resized to TARGET_SIZE.

    Steps:
        1 Load and normalize to RGB
        2 Re-compress at the given quality level
        3 Compute pixel-wise difference between original and re-compressed
        4 Normalize differences and convert to grayscale (Normalize = bring every number to a consistent scaling of 0 - 1)
    """
    original   = Image.open(image_path).convert('RGB') # Some images are rgba, make sure its in the right type
    compressed = jpeg_recompress_in_memory(original, quality) # Activate re-compression function

    # Subtract pixel by pixel to reveal compression differences (also applies abs value)
    ela_im = ImageChops.difference(original, compressed)

    # Find the brightest pixel and scale it to 255 (max white value)
    # takes the amount the pixel was enhanced by and applies it to the rest of the image
    extrema  = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = MIN_MAX_DIFF

    scale  = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    # Convert to grayscale and resize
    # Converting to grayscale to get rid of useless data (such as colors) for faster running time
    ela_gray = ela_im.convert('L')
    H, W     = TARGET_SIZE
    ela_gray = ela_gray.resize((W, H), Image.BILINEAR)

    return np.array(ela_gray).astype(np.float32)


if __name__ == "__main__":
    ela_map = run_ela(r"../../images/real_imagesLarge/flickr30k-images/36979.jpg")

    plt.figure(figsize=(6, 6))
    plt.title(f"ELA Map (quality={DEFAULT_ELA_QUALITY})")
    plt.imshow(ela_map, cmap='gray')
    plt.axis("off")
    plt.show()
