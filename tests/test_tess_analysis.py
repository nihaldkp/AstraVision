"""Tests for real TESS retrieval, preprocessing, BLS, and plots."""

import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")

import lightkurve as lk

from src.astrovision.analysis.light_curve import prepare_light_curve
from src.astrovision.analysis.transit_search import search_transits
from src.astrovision.tess.data_access import (
    normalize_target_identifier,
    retrieve_tess_light_curve,
)
from src.astrovision.visualization.plots import (
    plot_folded_light_curve,
    plot_light_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESS_DATA_PATH = PROJECT_ROOT / "data" / "raw"


class TessAnalysisTests(unittest.TestCase):
    """Verify the analysis using a downloaded public TESS observation."""

    @classmethod
    def setUpClass(cls) -> None:
        known_target_files = list(
            TESS_DATA_PATH.glob("mastDownload/**/*0000000420814525*.fits")
        )
        if known_target_files:
            cls.raw_curve = lk.read(known_target_files[0])
        else:
            cls.raw_curve = retrieve_tess_light_curve("HD 209458")

    def test_target_identifier_validation(self) -> None:
        self.assertEqual(normalize_target_identifier("150428135"), "TIC 150428135")
        self.assertEqual(normalize_target_identifier("tic 150428135"), "TIC 150428135")
        self.assertEqual(normalize_target_identifier("TOI 700"), "TOI 700")
        with self.assertRaises(ValueError):
            normalize_target_identifier("")
        with self.assertRaises(ValueError):
            normalize_target_identifier("TIC not-a-number")

    def test_clean_normalize_bls_and_plots(self) -> None:
        prepared_curve = prepare_light_curve(self.raw_curve)
        self.assertGreater(len(prepared_curve), 100)
        self.assertTrue(np.all(np.isfinite(prepared_curve.flux.value)))
        self.assertAlmostEqual(float(np.median(prepared_curve.flux.value)), 1.0, places=5)

        # Use the actual observation, with one deliberately invalid measurement
        # in a copy, to exercise the missing-value removal path.
        curve_with_missing_value = self.raw_curve.copy()
        curve_with_missing_value.flux.value[0] = np.nan
        cleaned_copy = prepare_light_curve(curve_with_missing_value)
        self.assertEqual(len(cleaned_copy), len(self.raw_curve) - 1)

        candidate = search_transits(
            prepared_curve,
            minimum_period_days=0.5,
            maximum_period_days=20.0,
        )
        self.assertAlmostEqual(candidate.period_days, 3.5243, delta=0.1)
        self.assertGreater(candidate.duration_hours, 0)
        self.assertGreater(candidate.depth_fraction, 0)
        self.assertTrue(np.isfinite(candidate.depth_fraction))
        self.assertTrue(np.isfinite(candidate.power))

        light_curve_figure = plot_light_curve(prepared_curve)
        folded_figure = plot_folded_light_curve(
            prepared_curve,
            candidate.period_days,
            candidate.transit_time_days,
        )
        self.assertEqual(len(light_curve_figure.axes), 1)
        self.assertEqual(len(folded_figure.axes), 1)
        import matplotlib.pyplot as plt

        plt.close(light_curve_figure)
        plt.close(folded_figure)

    def test_invalid_light_curve_inputs_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            prepare_light_curve(None)
        with self.assertRaises(TypeError):
            search_transits(None)
        with self.assertRaises(ValueError):
            search_transits(self.raw_curve, minimum_period_days=0)

    def test_no_mast_product_reports_lookup_error(self) -> None:
        with patch(
            "src.astrovision.tess.data_access.lk.search_lightcurve",
            side_effect=[[], []],
        ):
            with self.assertRaises(LookupError):
                retrieve_tess_light_curve(
                    "TIC 999999999",
                    download_directory=PROJECT_ROOT / "data" / "processed",
                )


if __name__ == "__main__":
    unittest.main()
