"""Monsoon onset detection algorithms.

This module implements the detection of monsoon onset dates using the
5-day wet spell criterion based on Moron & Robertson (2013).
"""

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


def detect_observed_onset(
    rainfall_ds: xr.DataArray,
    thresh_slice: xr.DataArray,
    year: int,
    mok: bool = True,
    mok_month: int = 6,
    mok_day: int = 2,
    onset_window: int = 5,
) -> xr.DataArray:
    """Detect observed monsoon onset dates for a given year.

    Uses the wet spell criterion: onset occurs on the first day where
    (1) rainfall > 1mm and (2) cumulative rainfall over onset_window days
    exceeds the threshold.

    Args:
        rainfall_ds: Daily rainfall DataArray with time, lat, lon dimensions
        thresh_slice: Threshold DataArray with lat, lon dimensions
        year: Year for which to detect onset
        mok: If True, start detection from MOK date (June 2 by default)
        mok_month: Month for MOK filter (default: 6 for June)
        mok_day: Day for MOK filter (default: 2)
        onset_window: Number of consecutive days for wet spell (default: 5)

    Returns:
        DataArray of onset dates with lat, lon dimensions
        Values are datetime64[ns] or NaT where no onset detected

    Example:
        >>> rainfall = xr.open_dataarray("rainfall_2020.nc")
        >>> thresholds = xr.open_dataarray("thresholds.nc")
        >>> onset_dates = detect_observed_onset(rainfall, thresholds, 2020)
        >>> print(onset_dates.sel(lat=20, lon=75, method='nearest'))
    """
    rain_slice = rainfall_ds
    window = onset_window

    if mok:
        start_date = datetime(year, mok_month, mok_day)
        date_label = (
            f"MOK date ({datetime(year, mok_month, mok_day).strftime('%B %d')})"
        )
    else:
        start_date = datetime(year, 5, 1)  # May 1st
        date_label = "May 1st"

    time_dates = pd.to_datetime(rain_slice.time.values)
    start_idx_candidates = np.where(time_dates > start_date)[0]

    if len(start_idx_candidates) == 0:
        print(
            f"Warning: {date_label} ({start_date.strftime('%Y-%m-%d')}) "
            f"not found in data for year {year}"
        )
        fallback_date = datetime(year, 4, 1)
        start_idx = np.where(time_dates >= fallback_date)[0][0]
        print("Using fallback date: April 1st")
    else:
        start_idx = start_idx_candidates[0]
        print(
            f"Using {date_label} ({start_date.strftime('%Y-%m-%d')}) "
            f"as start date for onset detection"
        )

    rain_subset = rain_slice.isel(time=slice(start_idx, None))
    rolling_sum = rain_subset.rolling(
        time=window, min_periods=window, center=False
    ).sum()
    rolling_sum_aligned = rolling_sum.shift(time=-(window - 1))

    # Wet spell criteria
    first_day_condition = rain_subset > 1
    sum_condition = rolling_sum_aligned > thresh_slice
    onset_condition = first_day_condition & sum_condition

    def find_first_true(arr: np.ndarray) -> int:
        """Find index of first True value, or -1 if none."""
        if arr.any():
            return int(np.argmax(arr))
        return -1

    onset_indices = xr.apply_ufunc(
        find_first_true,
        onset_condition,
        input_core_dims=[["time"]],
        output_dtypes=[int],
        vectorize=True,
    )

    valid_mask = onset_indices.to_numpy() >= 0
    time_coords = rain_subset.time.to_numpy()
    onset_dates_array = np.full(
        onset_indices.shape, np.datetime64("NaT"), dtype="datetime64[ns]"
    )

    for i in range(onset_indices.shape[0]):
        for j in range(onset_indices.shape[1]):
            if valid_mask[i, j]:
                idx = int(onset_indices[i, j].to_numpy())
                if 0 <= idx < len(time_coords):
                    onset_dates_array[i, j] = time_coords[idx]

    onset_da = xr.DataArray(
        onset_dates_array,
        coords=[("lat", rain_slice.lat.to_numpy()), ("lon", rain_slice.lon.to_numpy())],
        name="onset_date",
    )

    return onset_da
