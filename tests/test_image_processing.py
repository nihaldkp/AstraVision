"""Tests for the image-processing pipeline using a public NASA image."""

import unittest
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.astrovision.image_processing.contrast_enhancer import enhance_contrast
from src.astrovision.image_processing.grayscale_converter import convert_to_grayscale
from src.astrovision.image_processing.image_loader import load_image
from src.astrovision.image_processing.noise_reducer import reduce_noise
from src.astrovision.image_processing.star_detector import (
    detect_stars,
    draw_star_markers,
)
from src.astrovision.visualization.plots import plot_image_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NASA_IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "hubble_deep_field_pia12110.jpg"
NASA_IMAGE_URL = (
    "https://images-assets.nasa.gov/image/PIA12110/PIA12110~small.jpg"
)


class ImageProcessingTests(unittest.TestCase):
    """Exercise all image operations with NASA's Hubble Deep Field image."""

    @classmethod
    def setUpClass(cls) -> None:
        NASA_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not NASA_IMAGE_PATH.exists():
            urlretrieve(NASA_IMAGE_URL, NASA_IMAGE_PATH)
        cls.color_image = load_image(str(NASA_IMAGE_PATH))
        cls.grayscale_image = convert_to_grayscale(cls.color_image)

    def test_pipeline_preserves_dimensions_and_expected_types(self) -> None:
        blurred = reduce_noise(self.grayscale_image)
        enhanced = enhance_contrast(blurred)
        candidates = detect_stars(self.grayscale_image)

        self.assertEqual(blurred.shape, self.grayscale_image.shape)
        self.assertEqual(enhanced.shape, self.grayscale_image.shape)
        self.assertEqual(blurred.dtype, np.uint8)
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            self.assertGreaterEqual(candidate.x, 0)
            self.assertGreaterEqual(candidate.y, 0)
            self.assertLess(candidate.x, self.grayscale_image.shape[1])
            self.assertLess(candidate.y, self.grayscale_image.shape[0])

    def test_alternative_processing_parameters(self) -> None:
        blurred = reduce_noise(self.grayscale_image, kernel_size=7)
        enhanced = enhance_contrast(
            blurred, clip_limit=3.0, tile_grid_size=(10, 10)
        )
        self.assertEqual(enhanced.shape, self.grayscale_image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_marker_and_plot_creation(self) -> None:
        candidates = detect_stars(self.grayscale_image)
        marked = draw_star_markers(self.color_image, candidates)
        figure = plot_image_results(
            self.color_image, self.grayscale_image, candidates
        )
        self.assertEqual(marked.shape, self.color_image.shape)
        self.assertEqual(marked.dtype, self.color_image.dtype)
        self.assertEqual(len(figure.axes), 2)
        import matplotlib.pyplot as plt

        plt.close(figure)

    def test_invalid_images_are_rejected(self) -> None:
        invalid_color_inputs = (
            None,
            np.empty((0, 0), dtype=np.uint8),
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 4), dtype=np.uint8),
        )
        for invalid_image in invalid_color_inputs:
            with self.subTest(image=type(invalid_image).__name__):
                with self.assertRaises(ValueError):
                    convert_to_grayscale(invalid_image)

        invalid_grayscale_inputs = (
            None,
            np.empty((0, 0), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        for invalid_image in invalid_grayscale_inputs:
            with self.subTest(image=type(invalid_image).__name__):
                with self.assertRaises(ValueError):
                    reduce_noise(invalid_image)
                with self.assertRaises(ValueError):
                    enhance_contrast(invalid_image)
                with self.assertRaises(ValueError):
                    detect_stars(invalid_image)

        invalid_8_bit_inputs = (
            np.zeros((10, 10), dtype=np.float32),
        )
        for invalid_image in invalid_8_bit_inputs:
            with self.subTest(dtype=invalid_image.dtype):
                with self.assertRaises(ValueError):
                    enhance_contrast(invalid_image)
                with self.assertRaises(ValueError):
                    detect_stars(invalid_image)

        with self.assertRaises(FileNotFoundError):
            load_image(str(NASA_IMAGE_PATH.with_name("missing-image.jpg")))
        with self.assertRaises(ValueError):
            enhance_contrast(self.grayscale_image, clip_limit=0)
        with self.assertRaises(ValueError):
            enhance_contrast(self.grayscale_image, tile_grid_size=(8, 0))

    def test_invalid_kernel_and_detection_parameters_are_rejected(self) -> None:
        for invalid_kernel in (0, -1, 2, 3.0, True):
            with self.subTest(kernel=invalid_kernel):
                with self.assertRaises(ValueError):
                    reduce_noise(self.grayscale_image, invalid_kernel)
        with self.assertRaises(ValueError):
            detect_stars(self.grayscale_image, min_area=0)
        with self.assertRaises(ValueError):
            detect_stars(self.grayscale_image, min_area=5, max_area=2)


if __name__ == "__main__":
    unittest.main()
