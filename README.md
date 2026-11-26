# AstroVision

AstroVision is a beginner-friendly Python application for exploring
astronomical images and searching real TESS observations for possible
exoplanet-transit signals. It connects a simple OpenCV image workflow with
public light curves from NASA's Mikulski Archive for Space Telescopes (MAST).

**A periodic dip found by the software is only a candidate signal. It is not
proof that a planet exists.** Instrumental effects, stellar variability,
background sources, noise, and eclipsing binary stars can all produce
transit-like dips. Confirmation needs additional observations and scientific
validation.

## Objectives and features

### Astronomical image processing

- Load color image files with OpenCV.
- Convert BGR images to grayscale.
- Reduce noise with Gaussian blur.
- Enhance local contrast with CLAHE.
- Find compact bright-source candidates and estimate their pixel centroids.
- Display the original image with candidate circles next to the processed image.

The detector is an educational brightness/shape detector. It can mark stars,
galaxies, hot pixels, and artifacts; it does not identify or classify objects
against an astronomical catalog.

### TESS transit analysis

- Search MAST for a TESS light curve using a TIC ID or another supported target
  name.
- Download the first available SPOC product into `data/raw/`.
- Remove rows with invalid times or fluxes and non-positive flux.
- Normalize flux so its median is 1.
- Plot brightness against time and make a phase-folded plot.
- Search for a periodic box-shaped dip with Astropy's Box Least Squares (BLS).
- Report the best-fit candidate period, duration, depth, transit time, and
  search power.

## Technologies

- Python 3.12
- NumPy for pixel arrays and numerical calculations
- OpenCV (`opencv-python`) for image reading and processing
- Matplotlib for result plots
- Lightkurve for TESS light-curve discovery and downloads
- Astropy for astronomical time-series and BLS
- Pandas and SciPy for scientific Python ecosystem support
- Tkinter for the desktop GUI (included with standard Python on typical Windows
  installations)

## Project structure

```text
src/astrovision/
    project_info.py
    image_processing/
        image_loader.py
        grayscale_converter.py
        noise_reducer.py
        contrast_enhancer.py
        star_detector.py
    tess/
        data_access.py
    analysis/
        light_curve.py
        transit_search.py
    visualization/
        plots.py
    gui/
        app.py
data/
    raw/          # downloaded NASA image and TESS observations (git-ignored)
    processed/    # optional derived output (git-ignored)
tests/
    test_image_processing.py
    test_tess_analysis.py
requirements.txt
README.md
```

## Installation

Open PowerShell in the project folder:

```powershell
cd D:\Astra
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The `.venv` is a project-specific environment. The requirements install the
packages used by image processing, plots, and astronomical analysis. No
separate GUI framework is needed; the application uses Tkinter.

## Run the application

From `D:\Astra`:

```powershell
.\.venv\Scripts\python.exe -m src.astrovision.gui.app
```

The window has two tabs: **Image Processing** and **TESS Transit Analysis**.
TESS data retrieval requires an internet connection and a matching public MAST
product.

## Image processing workflow

1. Select **Image Processing** and click **Load image**.
2. Choose a color astronomical image such as a NASA image in JPEG or PNG form.
3. Use **Grayscale**, **Reduce noise**, and **Enhance contrast** as needed.
4. Click **Detect stars** to mark compact bright-source candidates.
5. Click **Show results** to display the original and processed images.

The reusable functions can also be imported from their modules. They accept
NumPy arrays and raise clear errors for invalid input. The project tests use
NASA's Hubble Deep Field image PIA12110. If it is not already in
`data/raw/hubble_deep_field_pia12110.jpg`, the test downloads the public NASA
image to that ignored data folder.

## TESS light-curve workflow

1. Select **TESS Transit Analysis**.
2. Enter a TIC number (for example `150428135` or `TIC 150428135`) or a target
   name supported by MAST.
3. Click **Retrieve and analyse**. AstroVision prefers a TESS SPOC product and
   falls back to another available TESS author if no SPOC product exists. It
   downloads one returned observation, removes invalid measurements, and
   normalizes the flux.
4. Review the reported BLS candidate parameters.
5. Use **Show light curve** and **Show folded curve** to inspect the data.

Downloaded TESS observations are written under `data/raw/`. The project ignores
downloaded data files so large observations are not accidentally committed.
If MAST is unavailable, the GUI reports the retrieval or analysis error rather
than substituting simulated data.

## Light curves and normalization

A light curve records a target's measured brightness over observation time.
TESS produces a series of flux measurements. AstroVision removes points whose
time or flux is not finite and removes non-positive flux values. It then
divides the remaining flux by its median, giving a relative brightness with a
median near 1. This makes changes easier to compare but does not remove every
instrumental trend or stellar variation.

## Box Least Squares (BLS)

BLS searches for a repeating, approximately box-shaped decrease in brightness.
It tests trial orbital periods and transit durations. For each trial, it
compares a baseline brightness model with a model that includes a lower-flux
interval. AstroVision reports the highest-power trial in the configured
period range.

- **Period:** time between repeating candidate dips, in days.
- **Duration:** approximate length of one dip, in hours.
- **Depth:** fractional flux decrease; the GUI also displays it as a percentage.
- **Transit time:** fitted center time, in the TESS BTJD time system.
- **Power:** relative BLS fit statistic for comparing the searched trials; it
  is not a probability that the signal is a planet.

BLS does not determine the object causing a dip. Eclipsing binaries,
contaminating neighboring stars, instrumental systematics, and random noise
can mimic transits. A candidate needs vetting (for example, checking odd/even
events, centroid motion, known catalogs, and independent observations) before
any claim of an exoplanet. AstroVision intentionally labels results as
candidates and does not claim confirmation.

## Run tests

From the project folder:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Image tests use the NASA Hubble image. TESS analysis tests use an actual
downloaded public FITS light curve when available; otherwise they retrieve a
real observation for HD 209458. Some tests therefore need network access
the first time they run. Invalid-input tests use small arrays solely to verify
input validation, not as observational data. One cleaning test inserts a NaN
into a copy of a real TESS measurement only to verify that the invalid row is
removed; the altered copy is not used for BLS analysis or presented as an
observation.

## Verified real-data example

During development, Lightkurve found and downloaded a public SPOC TESS Sector
56 light curve for HD 209458 (TIC 420814525). The observation had 117,488
cadences; preprocessing retained them and normalized median flux to 1. The
tested BLS search returned a candidate period of 3.52429 days, duration 3
hours, and depth about 1.279%. The period is consistent with the well-studied
transiting system, but AstroVision reports only its fitted candidate and does
not confirm planets. Results can differ with the selected product, data
quality, and search settings.

## Limitations and future improvements

- The image detector finds simple bright connected regions; it is not a
  calibrated source-extraction or star-identification package.
- The GUI currently uses the first returned product (preferring SPOC) and does
  not let the user choose a sector or cadence.
- Only one selected light-curve product is analysed at a time; multiple-sector
  stitching and systematic detrending are not implemented.
- The BLS period range is currently 0.5–20 days, with a small set of tested
  transit durations. A signal outside this range will not be found.
- No false-positive vetting, radial-velocity follow-up, or catalog-based
  confirmation is performed.

Possible future work includes sector selection, calibrated FITS image support,
improved aperture photometry, multi-sector light-curve stitching, trend
removal, and transit-candidate vetting tools.
