"""Find bright, compact source candidates in a grayscale image."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StarCandidate:
    """Pixel position and approximate size of one bright-source candidate."""

    x: float
    y: float
    radius: float
    area: int


def detect_stars(
    grayscale_image: np.ndarray,
    min_area: int = 2,
    max_area: int = 150,
    background_sigma: float = 8.0,
) -> list[StarCandidate]:
    """Find compact bright sources and estimate their pixel centroids.

    This simple detector finds image features, not scientifically confirmed
    stars. Galaxies, hot pixels, and image artifacts may also be detected.
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
        isinstance(min_area, bool)
        or not isinstance(min_area, int)
        or min_area < 1
    ):
        raise ValueError("min_area must be a positive integer.")
    if (
        isinstance(max_area, bool)
        or not isinstance(max_area, int)
        or max_area < min_area
    ):
        raise ValueError("max_area must be an integer greater than or equal to min_area.")
    if (
        isinstance(background_sigma, bool)
        or not isinstance(background_sigma, (int, float))
        or not np.isfinite(background_sigma)
        or background_sigma <= 0
    ):
        raise ValueError("background_sigma must be a finite positive number.")

    # Subtract a smoothed background to make compact bright sources stand out.
    background = cv2.GaussianBlur(
        grayscale_image, (0, 0), sigmaX=float(background_sigma)
    )
    foreground = cv2.subtract(grayscale_image, background)
    _, binary_image = cv2.threshold(
        foreground, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    count, _, statistics, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    candidates: list[StarCandidate] = []

    for component_index in range(1, count):
        area = int(statistics[component_index, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            x, y = centroids[component_index]
            radius = float(np.sqrt(area / np.pi))
            candidates.append(
                StarCandidate(x=float(x), y=float(y), radius=radius, area=area)
            )

    return candidates


def draw_star_markers(
    color_image: np.ndarray,
    candidates: list[StarCandidate],
) -> np.ndarray:
    """Return a copy of a BGR image with circles around source candidates."""
    if (
        not isinstance(color_image, np.ndarray)
        or color_image.size == 0
        or color_image.ndim != 3
        or color_image.shape[2] != 3
        or color_image.dtype != np.uint8
    ):
        raise ValueError("Input must be a non-empty three-channel uint8 BGR image.")

    marked_image = color_image.copy()
    for candidate in candidates:
        center = (round(candidate.x), round(candidate.y))
        radius = max(3, round(candidate.radius * 2))
        cv2.circle(marked_image, center, radius, (0, 255, 0), 1)

    return marked_image
