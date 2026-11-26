"""Load an image from a file for later AstroVision processing."""

import sys

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load an image file and return its pixels as a NumPy array.

    Raises:
        FileNotFoundError: If OpenCV cannot read the image file.
    """
    image = cv2.imread(image_path)

    # OpenCV returns None when the file is missing or cannot be decoded.
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def main() -> None:
    """Load the image path supplied on the command line."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python image_loader.py IMAGE_PATH")

    image = load_image(sys.argv[1])
    print(f"Loaded image: {image.shape[1]} x {image.shape[0]} pixels")
    print(f"NumPy array data type: {image.dtype}")


if __name__ == "__main__":
    main()
