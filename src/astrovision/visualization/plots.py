"""Plot astronomical images and light curves."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lightkurve import LightCurve

from ..image_processing.star_detector import StarCandidate, draw_star_markers


def plot_image_results(
    original_bgr: np.ndarray,
    processed_image: np.ndarray,
    candidates: list[StarCandidate] | None = None,
) -> plt.Figure:
    """Show the original image beside its processed grayscale image."""
    marked_image = (
        draw_star_markers(original_bgr, candidates)
        if candidates is not None
        else original_bgr
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original image and source candidates")
    axes[0].axis("off")
    axes[1].imshow(processed_image, cmap="gray")
    axes[1].set_title("Processed grayscale image")
    axes[1].axis("off")
    figure.tight_layout()
    return figure


def plot_light_curve(light_curve: LightCurve) -> plt.Figure:
    """Plot normalized stellar brightness against observation time."""
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(light_curve.time.value, light_curve.flux.value, ".", markersize=2)
    axis.set_xlabel(f"Time ({light_curve.time.format})")
    axis.set_ylabel("Normalized flux")
    axis.set_title("TESS light curve")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_folded_light_curve(
    light_curve: LightCurve,
    period_days: float,
    transit_time_days: float,
) -> plt.Figure:
    """Plot normalized brightness folded on a candidate orbital period."""
    if not np.isfinite(period_days) or period_days <= 0:
        raise ValueError("period_days must be a finite positive number.")

    time_days = light_curve.time.value
    flux = light_curve.flux.value
    phase = ((time_days - transit_time_days + period_days / 2) % period_days)
    phase = phase / period_days - 0.5
    order = np.argsort(phase)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(phase[order], flux[order], ".", markersize=2)
    axis.set_xlabel("Orbital phase")
    axis.set_ylabel("Normalized flux")
    axis.set_title(f"Folded TESS light curve (period {period_days:.5f} days)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure
