"""Reduce image noise with a Gaussian blur."""

import sys

import cv2
import numpy as np

if __package__:
    from .grayscale_converter import convert_to_grayscale
    from .image_loader import load_image
else:
    from grayscale_converter import convert_to_grayscale
    from image_loader import load_image


def reduce_noise(
    grayscale_image: np.ndarray, kernel_size: int = 5
) -> np.ndarray:
    """Apply Gaussian blur to a valid grayscale image.

    Args:
        grayscale_image: A non-empty, two-dimensional grayscale image.
        kernel_size: Positive odd width and height of the Gaussian kernel.

    Raises:
        ValueError: If the image or kernel size is invalid.
    """
    if (
        not isinstance(grayscale_image, np.ndarray)
        or grayscale_image.size == 0
        or grayscale_image.ndim != 2
    ):
        raise ValueError("Input must be a non-empty two-dimensional grayscale image.")

    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be a positive odd integer.")

    # OpenCV requires odd kernel dimensions for Gaussian blur.
    return cv2.GaussianBlur(
        grayscale_image, (kernel_size, kernel_size), sigmaX=0
    )


def main() -> None:
    """Load a color image, convert it, then apply Gaussian blur."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python noise_reducer.py IMAGE_PATH")

    color_image = load_image(sys.argv[1])
    grayscale_image = convert_to_grayscale(color_image)
    filtered_image = reduce_noise(grayscale_image)
    print(
        f"Filtered image: {filtered_image.shape[1]} x "
        f"{filtered_image.shape[0]} pixels"
    )
    print(f"NumPy array data type: {filtered_image.dtype}")


if __name__ == "__main__":
    main()
