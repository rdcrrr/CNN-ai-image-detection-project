"""
ela extraction
ai images have no real compression history, to re-compressing them will have unatural patters compared to real photos.
however this is less effective on photos that have been heavily compressed


ela (error level analysis)- when saving a jpeg image small error occur, recompressing the image and compare the two versions there will only be small diffrences
however if the image was edited there will be alot more diffrences
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO

TARGET_SIZE = (224, 224)  # consistent with the CNN pipeline


#takes the image PIL and re-saves it as a jpeg  with 75% quality. (75% being how aggressive the compression is)
#retunrs the re-compressed version
#(PIL is the way the pillow extension saves the image)
def jpeg_recompress_in_memory(img_pil, quality=75):
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def run_ela(image_path, quality=75):
    # Load image
    # forces RGB onto an image (some images can be RGBA)
    original = Image.open(image_path).convert('RGB')

    # Recompress at lower quality to stress-test compression differences.
    # Previously quality=95 was used, which produces near-zero difference
    # maps when the source image was already saved at high quality.
    # quality=75 gives a much more informative ELA map.
    compressed = jpeg_recompress_in_memory(original, quality)

    # finds the diffrences by deducting pixel by pixel
    ela_im = ImageChops.difference(original, compressed)

    # Normalizes the diffrences
    # (pretty much converting the image to black and adding the differences of each pixel to the color value)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    # Convert to grayscale and resize to TARGET_SIZE
    ela_gray = ela_im.convert('L')
    H, W = TARGET_SIZE
    ela_gray = ela_gray.resize((W, H), Image.BILINEAR)
    ela_np = np.array(ela_gray).astype(np.float32)

    return ela_np

if __name__ == "__main__":

    # Run ELA

    ela_map = run_ela(r"../../images/real_imagesLarge/flickr30k-images/36979.jpg")


    # Display

    plt.figure(figsize=(6, 6))
    plt.title("ELA Map (quality=75)")
    plt.imshow(ela_map, cmap='gray')
    plt.axis("off")
    plt.show()