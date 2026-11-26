"""Convert a loaded BGR color image to grayscale."""

import sys

import cv2
import numpy as np

if __package__:
    from .image_loader import load_image
else:
    from image_loader import load_image


def convert_to_grayscale(color_image: np.ndarray) -> np.ndarray:
    """Return a grayscale version of a loaded BGR color image.

    Raises:
        ValueError: If the input is not a valid three-channel color image.
    """
    if (
        not isinstance(color_image, np.ndarray)
        or color_image.size == 0
        or color_image.ndim != 3
        or color_image.shape[2] != 3
    ):
        raise ValueError("Input must be a non-empty three-channel BGR image.")

    return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


def main() -> None:
    """Load an image path from the command line and convert it to grayscale."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python grayscale_converter.py IMAGE_PATH")

    color_image = load_image(sys.argv[1])
    grayscale_image = convert_to_grayscale(color_image)
    print(
        f"Converted image: {grayscale_image.shape[1]} x "
        f"{grayscale_image.shape[0]} pixels"
    )
    print(f"NumPy array data type: {grayscale_image.dtype}")


if __name__ == "__main__":
    main()
