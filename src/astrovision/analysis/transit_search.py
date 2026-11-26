"""Search a TESS light curve for periodic dips using Box Least Squares."""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.timeseries import BoxLeastSquares
from lightkurve import LightCurve


@dataclass(frozen=True)
class TransitCandidate:
    """Best-fitting periodic dip found by the BLS search."""

    period_days: float
    duration_hours: float
    depth_fraction: float
    transit_time_days: float
    power: float


def search_transits(
    light_curve: LightCurve,
    minimum_period_days: float = 0.5,
    maximum_period_days: float = 20.0,
) -> TransitCandidate:
    """Return the highest-power BLS periodic-dip candidate.

    A BLS result is a candidate signal, not confirmation of an exoplanet.
    """
    if not isinstance(light_curve, LightCurve):
        raise TypeError("light_curve must be a Lightkurve LightCurve object.")
    if len(light_curve) < 20:
        raise ValueError("At least 20 measurements are required for a BLS search.")
    if any(
        isinstance(period, bool) or not isinstance(period, (int, float, np.number))
        for period in (minimum_period_days, maximum_period_days)
    ):
        raise ValueError("Period limits must be numbers.")
    if (
        not np.isfinite(minimum_period_days)
        or not np.isfinite(maximum_period_days)
        or minimum_period_days <= 0
        or maximum_period_days <= minimum_period_days
    ):
        raise ValueError(
            "Period limits must be finite, positive, and maximum must exceed minimum."
        )

    time_values = np.asarray(light_curve.time.value, dtype=float)
    flux_values = np.asarray(light_curve.flux.value, dtype=float)
    valid_measurements = np.isfinite(time_values) & np.isfinite(flux_values)
    if np.count_nonzero(valid_measurements) < 20:
        raise ValueError("At least 20 finite time and flux measurements are required.")

    usable_curve = light_curve[valid_measurements]
    median_flux = float(np.nanmedian(usable_curve.flux.value))
    if not np.isfinite(median_flux) or median_flux <= 0:
        raise ValueError("The light curve must have a finite positive median flux.")
    usable_curve = usable_curve.normalize()

    duration_hours = np.array([1, 2, 3, 4, 6, 8], dtype=float)
    duration_hours = duration_hours[
        duration_hours < minimum_period_days * 24
    ]
    if duration_hours.size == 0:
        duration_hours = np.array([minimum_period_days * 6], dtype=float)
    durations = duration_hours * u.hour
    bls = BoxLeastSquares(usable_curve.time, usable_curve.flux)
    periodogram = bls.autopower(
        duration=durations,
        minimum_period=minimum_period_days * u.day,
        maximum_period=maximum_period_days * u.day,
        minimum_n_transit=2,
    )

    best_index = int(np.nanargmax(periodogram.power.value))
    return TransitCandidate(
        period_days=float(periodogram.period[best_index].to_value(u.day)),
        duration_hours=float(periodogram.duration[best_index].to_value(u.hour)),
        depth_fraction=float(periodogram.depth[best_index].value),
        transit_time_days=float(periodogram.transit_time[best_index].value),
        power=float(periodogram.power[best_index].value),
    )
