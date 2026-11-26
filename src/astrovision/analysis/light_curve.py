"""Clean and normalize TESS light-curve measurements."""

import numpy as np
from lightkurve import LightCurve


def prepare_light_curve(light_curve: LightCurve) -> LightCurve:
    """Remove invalid measurements, sort by time, and normalize flux to median 1."""
    if not isinstance(light_curve, LightCurve):
        raise TypeError("light_curve must be a Lightkurve LightCurve object.")

    time_values = np.asarray(light_curve.time.value, dtype=float)
    flux_values = np.asarray(light_curve.flux.value, dtype=float)
    valid_measurements = np.isfinite(time_values) & np.isfinite(flux_values)
    valid_measurements &= flux_values > 0

    if np.count_nonzero(valid_measurements) < 3:
        raise ValueError("At least three valid positive-flux measurements are required.")

    cleaned_curve = light_curve[valid_measurements]
    sort_order = np.argsort(cleaned_curve.time.value)
    cleaned_curve = cleaned_curve[sort_order]
    median_flux = float(np.nanmedian(cleaned_curve.flux.value))
    if not np.isfinite(median_flux) or median_flux <= 0:
        raise ValueError("The light curve must have a finite positive median flux.")

    return cleaned_curve.normalize()
