"""Enhance local contrast in a grayscale image with CLAHE."""

import sys

import cv2
import numpy as np

if __package__:
    from .grayscale_converter import convert_to_grayscale
    from .image_loader import load_image
else:
    from grayscale_converter import convert_to_grayscale
    from image_loader import load_image


def enhance_contrast(
    grayscale_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local contrast in a non-empty 8-bit grayscale image.

    Args:
        grayscale_image: A two-dimensional grayscale NumPy array.
        clip_limit: Positive limit for contrast amplification.
        tile_grid_size: Positive (columns, rows) tile grid dimensions.

    Raises:
        ValueError: If the image or CLAHE parameters are invalid.
    """
    if (
        not isinstance(grayscale_image, np.ndarray)
        or grayscale_image.size == 0
        or grayscale_image.ndim != 2
        or grayscale_image.dtype != np.uint8
    ):
        raise ValueError(
            "Input must be a non-empty, two-dimensional uint8 grayscale image."
        )

    if (
        isinstance(clip_limit, bool)
        or not isinstance(clip_limit, (int, float))
        or not np.isfinite(clip_limit)
        or clip_limit <= 0
    ):
        raise ValueError("clip_limit must be a finite positive number.")

    if (
        not isinstance(tile_grid_size, tuple)
        or len(tile_grid_size) != 2
        or any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            for size in tile_grid_size
        )
    ):
        raise ValueError(
            "tile_grid_size must be a tuple of two positive integers."
        )

    # CLAHE enhances local contrast while limiting excessive amplification.
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(grayscale_image)


def main() -> None:
    """Load a color image, convert it, then enhance its contrast."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python contrast_enhancer.py IMAGE_PATH")

    color_image = load_image(sys.argv[1])
    grayscale_image = convert_to_grayscale(color_image)
    enhanced_image = enhance_contrast(grayscale_image)
    print(
        f"Enhanced image: {enhanced_image.shape[1]} x "
        f"{enhanced_image.shape[0]} pixels"
    )
    print(f"NumPy array data type: {enhanced_image.dtype}")


if __name__ == "__main__":
    main()
