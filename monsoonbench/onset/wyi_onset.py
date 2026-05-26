"""Webster-Yang Index (WYI) monsoon onset detection.

Ported from get_wyi_onset.m. Loads daily WYI NetCDF files, computes a
365-day climatology (Feb 29 excluded), and detects the onset date for a
target year as the first day where the 7-day centred moving average of
the WYI falls below the climatological June-2 threshold.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Calendar year range matching the MATLAB script (1979-2024)
_WYI_YEAR_RANGE = np.arange(1979, 2025)
_WYI_VARIABLE = "webster_yang_index"
_IDX_JUN2 = 152  # 0-indexed day-of-year for June 2 in a 365-day calendar
_EARLY_ONSET_WARNING_MONTH = 5
_EARLY_ONSET_WARNING_DAY = 20


def _load_wyi_year(year: int, data_dir: Path) -> np.ndarray:
    """Load the WYI time series for one year, stripping Feb 29 from leap years.

    Returns a 1-D array of length 365.
    """
    fname = data_dir / f"wyi_daily_{year}.nc"
    ds = xr.open_dataset(str(fname))
    wyi = ds[_WYI_VARIABLE].values.astype(float)
    ds.close()

    if len(wyi) == 366:
        # Remove Feb 29 (0-indexed day 59)
        wyi = np.delete(wyi, 59)

    if len(wyi) != 365:
        raise ValueError(
            f"Unexpected WYI length {len(wyi)} for year {year} "
            f"(expected 365 after removing Feb 29)."
        )
    return wyi


def _build_365day_dates(year: int) -> pd.DatetimeIndex:
    """Return a DatetimeIndex for *year* with Feb 29 removed."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    if len(dates) == 366:
        dates = dates[dates != pd.Timestamp(year, 2, 29)]
    return dates


def get_wyi_onset(
    year: int,
    data_dir: str,
) -> tuple[pd.Timestamp | None, np.ndarray, np.ndarray]:
    """Detect the WYI-based monsoon onset date for *year*.

    Loads daily WYI files for all years in the climatology period
    (1979-2024), builds a 365-day mean climatology, then finds the first
    day in *year* where the 7-day centred moving average of the WYI is
    below the climatological value on June 2.

    Parameters
    ----------
    year:
        Target year for onset detection.
    data_dir:
        Directory containing files named ``wyi_daily_{year}.nc``, each
        with a variable ``webster_yang_index`` of length 365 or 366.

    Returns:
    -------
    wyi_onset:
        Onset date as a ``pd.Timestamp``, or ``None`` if onset is not
        detected within the year.
    wyi_smoothed:
        7-day centred moving-mean of WYI for *year* (length 365).
        Edge values where the window is incomplete are ``NaN``.
    wyi_clim:
        Climatological mean WYI (length 365), computed over 1979-2024
        with ``NaN`` values omitted.

    Warns:
    -----
    UserWarning
        If the detected onset date is before May 20.
    """
    data_path = Path(data_dir)

    # --- Build climatology matrix (365 x n_years) ---
    wyi_yr = np.full((365, len(_WYI_YEAR_RANGE)), np.nan)
    for i, yr in enumerate(_WYI_YEAR_RANGE):
        wyi_yr[:, i] = _load_wyi_year(yr, data_path)

    wyi_clim = np.nanmean(wyi_yr, axis=1)  # shape (365,)

    # --- Threshold: climatological WYI on June 2 ---
    wyi_thres = wyi_clim[_IDX_JUN2]

    # --- Load and smooth target year ---
    wyi_idvyr = _load_wyi_year(year, data_path)
    wyi_smoothed = (
        pd.Series(wyi_idvyr)
        .rolling(window=7, center=True, min_periods=7)
        .mean()
        .to_numpy()
    )

    # --- Find first onset date ---
    t_yr = _build_365day_dates(year)
    onset_mask = wyi_smoothed < wyi_thres
    onset_dates = t_yr[onset_mask]

    if len(onset_dates) == 0:
        wyi_onset = None
    else:
        wyi_onset = onset_dates[0]
        early_onset_threshold = pd.Timestamp(year, _EARLY_ONSET_WARNING_MONTH, _EARLY_ONSET_WARNING_DAY)
        if wyi_onset < early_onset_threshold:
            warnings.warn(
                f"The WYI onset date ({wyi_onset.strftime('%Y-%m-%d')}) "
                f"is earlier than May 20.",
                stacklevel=1
            )

    return wyi_onset, wyi_smoothed, wyi_clim
