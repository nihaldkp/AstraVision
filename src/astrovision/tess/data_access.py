"""Retrieve public TESS light curves through Lightkurve and MAST."""

from pathlib import Path

import lightkurve as lk
from lightkurve import LightCurve


def normalize_target_identifier(target: str) -> str:
    """Format a TIC number or pass through another supported target name."""
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Enter a non-empty TIC ID or supported target identifier.")

    cleaned_target = " ".join(target.strip().split())
    if cleaned_target.isdecimal():
        return f"TIC {cleaned_target}"
    if cleaned_target[:3].lower() == "tic":
        parts = cleaned_target.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].isdecimal():
            raise ValueError("A TIC identifier must contain a numeric ID, e.g. TIC 150428135.")
        return f"TIC {parts[1]}"
    return cleaned_target


def retrieve_tess_light_curve(
    target: str,
    download_directory: str | Path | None = None,
) -> LightCurve:
    """Prefer a SPOC product, then download another available TESS product.

    Network, archive, and download errors from Lightkurve are allowed to
    propagate so callers can display or log the original failure.
    """
    normalized_target = normalize_target_identifier(target)
    if download_directory is None:
        project_root = Path(__file__).resolve().parents[3]
        download_path = project_root / "data" / "raw"
    else:
        download_path = Path(download_directory)
    download_path.mkdir(parents=True, exist_ok=True)

    search_result = lk.search_lightcurve(
        normalized_target,
        mission="TESS",
        author="SPOC",
        limit=1,
    )
    if len(search_result) == 0:
        search_result = lk.search_lightcurve(
            normalized_target,
            mission="TESS",
            limit=1,
        )
    if len(search_result) == 0:
        raise LookupError(f"No public TESS light curve found for {normalized_target}.")

    light_curve = search_result.download(
        quality_bitmask="default",
        download_dir=str(download_path),
    )
    if light_curve is None:
        raise RuntimeError(f"MAST returned no downloadable light curve for {normalized_target}.")
    return light_curve
