"""Tkinter interface for AstroVision's image and TESS analysis tools."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from lightkurve import LightCurve

from ..analysis.light_curve import prepare_light_curve
from ..analysis.transit_search import TransitCandidate, search_transits
from ..image_processing.contrast_enhancer import enhance_contrast
from ..image_processing.grayscale_converter import convert_to_grayscale
from ..image_processing.image_loader import load_image
from ..image_processing.noise_reducer import reduce_noise
from ..image_processing.star_detector import StarCandidate, detect_stars
from ..tess.data_access import retrieve_tess_light_curve
from ..visualization.plots import (
    plot_folded_light_curve,
    plot_image_results,
    plot_light_curve,
)


class AstroVisionApp:
    """Create the two simple AstroVision application tabs."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AstroVision")
        self.root.geometry("760x520")
        self.original_image: np.ndarray | None = None
        self.processed_image: np.ndarray | None = None
        self.star_candidates: list[StarCandidate] = []
        self.light_curve: LightCurve | None = None
        self.transit_candidate: TransitCandidate | None = None

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_tab = ttk.Frame(notebook, padding=12)
        self.tess_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.image_tab, text="Image Processing")
        notebook.add(self.tess_tab, text="TESS Transit Analysis")

        self._create_image_tab()
        self._create_tess_tab()

    def _create_image_tab(self) -> None:
        ttk.Label(
            self.image_tab,
            text="Load a public astronomical image, then choose a processing step.",
            wraplength=680,
        ).pack(anchor="w", pady=(0, 10))

        buttons = ttk.Frame(self.image_tab)
        buttons.pack(anchor="w", fill="x")
        actions = [
            ("Load image", self.load_astronomical_image),
            ("Grayscale", self.convert_image),
            ("Reduce noise", self.reduce_image_noise),
            ("Enhance contrast", self.enhance_image_contrast),
            ("Detect stars", self.detect_image_stars),
            ("Show results", self.show_image_results),
        ]
        for label, action in actions:
            ttk.Button(buttons, text=label, command=action).pack(
                side="left", padx=(0, 5), pady=4
            )

        self.image_status = tk.StringVar(value="No image loaded.")
        ttk.Label(
            self.image_tab, textvariable=self.image_status, wraplength=680
        ).pack(anchor="w", pady=12)
        ttk.Label(
            self.image_tab,
            text=(
                "Detected points are bright-source candidates, not verified "
                "astronomical star identifications."
            ),
            wraplength=680,
        ).pack(anchor="w")

    def _create_tess_tab(self) -> None:
        ttk.Label(
            self.tess_tab,
            text="Enter a TIC number or another identifier supported by MAST.",
        ).pack(anchor="w", pady=(0, 6))

        target_row = ttk.Frame(self.tess_tab)
        target_row.pack(anchor="w", fill="x")
        self.target_entry = ttk.Entry(target_row, width=28)
        self.target_entry.pack(side="left", padx=(0, 8))
        ttk.Button(
            target_row,
            text="Retrieve and analyse",
            command=self.analyse_tess_target,
        ).pack(side="left")

        plot_buttons = ttk.Frame(self.tess_tab)
        plot_buttons.pack(anchor="w", pady=10)
        ttk.Button(
            plot_buttons,
            text="Show light curve",
            command=self.show_tess_light_curve,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            plot_buttons,
            text="Show folded curve",
            command=self.show_folded_light_curve,
        ).pack(side="left")

        self.tess_status = tk.StringVar(
            value="No TESS observation has been retrieved."
        )
        ttk.Label(
            self.tess_tab, textvariable=self.tess_status, wraplength=680
        ).pack(anchor="w", pady=10)
        ttk.Label(
            self.tess_tab,
            text=(
                "A BLS dip is a candidate signal, not an exoplanet confirmation. "
                "Noise, eclipsing binaries, and other false positives require "
                "additional checks."
            ),
            wraplength=680,
        ).pack(anchor="w")

    def load_astronomical_image(self) -> None:
        """Load a user-selected image without altering the source file."""
        image_path = filedialog.askopenfilename(
            title="Select an astronomical image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not image_path:
            return

        try:
            self.original_image = load_image(image_path)
            self.processed_image = convert_to_grayscale(self.original_image)
            self.star_candidates = []
            self.image_status.set(
                f"Loaded {image_path}; image is "
                f"{self.original_image.shape[1]} x {self.original_image.shape[0]} pixels."
            )
        except Exception as error:
            messagebox.showerror("Image loading failed", str(error))

    def _get_grayscale_image(self) -> np.ndarray:
        """Return the current grayscale image or report a clear UI error."""
        if self.original_image is None:
            raise ValueError("Load an image before choosing a processing step.")
        if self.processed_image is None or self.processed_image.ndim != 2:
            self.processed_image = convert_to_grayscale(self.original_image)
        return self.processed_image

    def convert_image(self) -> None:
        """Convert the loaded image to grayscale."""
        try:
            self.processed_image = convert_to_grayscale(self._require_original())
            self.image_status.set("Converted the loaded image to grayscale.")
        except Exception as error:
            messagebox.showerror("Grayscale conversion failed", str(error))

    def reduce_image_noise(self) -> None:
        """Apply the default Gaussian blur to the current grayscale image."""
        try:
            self.processed_image = reduce_noise(self._get_grayscale_image())
            self.image_status.set("Applied Gaussian noise reduction.")
        except Exception as error:
            messagebox.showerror("Noise reduction failed", str(error))

    def enhance_image_contrast(self) -> None:
        """Apply default CLAHE contrast enhancement."""
        try:
            self.processed_image = enhance_contrast(self._get_grayscale_image())
            self.image_status.set("Enhanced local image contrast with CLAHE.")
        except Exception as error:
            messagebox.showerror("Contrast enhancement failed", str(error))

    def detect_image_stars(self) -> None:
        """Detect compact bright-source candidates in the current image."""
        try:
            detection_image = convert_to_grayscale(self._require_original())
            self.star_candidates = detect_stars(detection_image)
            self.image_status.set(
                f"Found {len(self.star_candidates)} bright-source candidates. "
                "These are not confirmed star identifications."
            )
        except Exception as error:
            messagebox.showerror("Source detection failed", str(error))

    def show_image_results(self) -> None:
        """Display the original image with markers beside the processed image."""
        if self.original_image is None:
            messagebox.showerror("No image", "Load an image before showing results.")
            return
        try:
            figure = plot_image_results(
                self.original_image,
                self._get_grayscale_image(),
                self.star_candidates,
            )
            plt.show()
            plt.close(figure)
        except Exception as error:
            messagebox.showerror("Image display failed", str(error))

    def _require_original(self) -> np.ndarray:
        if self.original_image is None:
            raise ValueError("Load an image before choosing a processing step.")
        return self.original_image

    def analyse_tess_target(self) -> None:
        """Retrieve a real TESS light curve, prepare it, and search with BLS."""
        target = self.target_entry.get().strip()
        if not target:
            messagebox.showerror("Missing target", "Enter a TIC ID or target name.")
            return
        self.tess_status.set("Searching MAST and analysing the returned observation…")
        self.root.update_idletasks()

        try:
            downloaded_curve = retrieve_tess_light_curve(target)
            self.light_curve = prepare_light_curve(downloaded_curve)
            self.transit_candidate = search_transits(self.light_curve)
            candidate = self.transit_candidate
            self.tess_status.set(
                f"Retrieved and normalized {len(self.light_curve)} real TESS "
                f"measurements. BLS candidate: period {candidate.period_days:.5f} "
                f"days, duration {candidate.duration_hours:.2f} hours, depth "
                f"{candidate.depth_fraction * 100:.3f}%. This is not a confirmed "
                "exoplanet."
            )
        except Exception as error:
            self.tess_status.set("TESS retrieval or analysis did not complete.")
            messagebox.showerror("TESS analysis failed", str(error))

    def show_tess_light_curve(self) -> None:
        """Display the normalized TESS brightness measurements."""
        if self.light_curve is None:
            messagebox.showerror(
                "No light curve", "Retrieve TESS data before plotting the light curve."
            )
            return
        figure = plot_light_curve(self.light_curve)
        plt.show()
        plt.close(figure)

    def show_folded_light_curve(self) -> None:
        """Display the light curve folded at the best BLS candidate period."""
        if self.light_curve is None or self.transit_candidate is None:
            messagebox.showerror(
                "No BLS result", "Retrieve and analyse TESS data first."
            )
            return
        candidate = self.transit_candidate
        figure = plot_folded_light_curve(
            self.light_curve,
            candidate.period_days,
            candidate.transit_time_days,
        )
        plt.show()
        plt.close(figure)


def main() -> None:
    """Start the AstroVision desktop application."""
    root = tk.Tk()
    AstroVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
