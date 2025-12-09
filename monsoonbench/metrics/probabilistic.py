"""Probabilistic model onset metrics computation.

This module provides the ProbabilisticOnsetMetrics class for computing
onset metrics from probabilistic ensemble model forecasts.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .base import OnsetMetricsBase


class ProbabilisticOnsetMetrics(OnsetMetricsBase):
    """Probabilistic model specific onset metrics calculations."""

    @staticmethod
    def get_forecast_probabilistic_twice_weekly(yr, model_forecast_dir):
        """Load model precip data for twice-weekly initializations from May to July.

        Filters for Mondays and Thursdays in the specified year.
        The forecast file is expected to be named as '{year}.nc' in the model_forecast_dir with
        variable "tp" being daily accumulated rainfall with dimensions (init_time, lat, lon, step, member).

        Parameters:
        yr: int, year to load data for

        Returns:
        p_model: ndarray, precipitation data
        """
        fname = f"{yr}.nc"
        file_path = Path(model_forecast_dir) / fname

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Filter for twice weekly data from daily for the specified year
        start_date = datetime(2024, 5, 1)
        end_date = datetime(2024, 7, 31)
        date_range = pd.date_range(start_date, end_date, freq="D")

        # Find Mondays and Thursdays
        is_monday = date_range.weekday == 0
        is_thursday = date_range.weekday == 3
        filtered_dates = date_range[is_monday | is_thursday]
        filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f"{yr}-%m-%d"))

        # Load data using xarray
        ds = xr.open_dataset(file_path)
        if "time" in ds.dims:
            ds = ds.rename({"time": "init_time"})
        if "number" in ds.dims:
            ds = ds.rename({"number": "member"})
        # Find common dates between desired dates and available dates
        available_init_times = pd.to_datetime(ds.init_time.values)
        matching_times = available_init_times[
            available_init_times.isin(filtered_dates_yr)
        ]

        if len(matching_times) == 0:
            raise ValueError(f"No matching initialization times found for year {yr}")

        # Select only the matching initialization times
        ds = ds.sel(init_time=matching_times)
        if "day" in ds.dims:
            # Check if the first value of 'day' is 0, then slice to exclude it
            if ds["day"][0].values == 0:
                ds = ds.sel(day=slice(1, None))
        # Check if 'step' dimension exists and conditionally slice
        if "step" in ds.dims:
            # Check if the first value of 'step' is 0, then slice to exclude it
            if ds["step"][0].values == 0:
                ds = ds.sel(step=slice(1, None))
        if "day" in ds.dims:
            ds = ds.rename({"day": "step"})

        p_model = ds["tp"]  # in mm
        ds.close()
        return p_model

    @staticmethod
    def compute_mean_onset_for_all_members(
        p_model,
        thresh_slice,
        onset_da,
        max_forecast_day=15,
        mok=True,
        onset_window=5,
        mok_month=6,
        mok_day=2,
    ):
        """Compute onset dates for each ensemble member, init time, and grid point.

        Only processes forecasts initialized before the observed onset date.
        For each initialization, requires at least 50% of members to have onset.
        If threshold met, uses ceiling of mean onset day as the ensemble onset.

        Parameters:
        p_model: xarray DataArray with dims [init_time, step, lat, lon, member]
        thresh_slice: xarray DataArray with threshold values for each grid point
        onset_da: xarray DataArray with observed onset dates for filtering
        max_forecast_day: int, maximum forecast day to consider for onset (default 15)
        mok: bool, if True only count onset after June 2nd (MOK date), if False use all forecasts

        Returns:
        pandas DataFrame with columns: init_time, lat, lon, onset_day, member_onset_count, total_members
        """
        window = onset_window
        results_list = []

        # Get dimensions
        init_times = p_model.init_time.values
        lats = p_model.lat.values
        lons = p_model.lon.values
        members = p_model.member.values

        date_method = f"MOK ({mok_month}/{mok_day} filter)" if mok else "no date filter"
        print(
            f"Processing {len(init_times)} init times x {len(lats)} lats x {len(lons)} lons..."
        )
        print(f"Using {date_method} for onset detection")
        print("Only processing forecasts initialized before observed onset dates")
        print(
            f"Requiring ≥50% of {len(members)} members to have onset for ensemble onset"
        )

        # We need first 19 days to check onset up to day 15 (because of 5-day window)
        max_steps_needed = max_forecast_day + window - 1

        # Track statistics
        total_potential_inits = 0
        valid_inits = 0
        skipped_no_obs = 0
        skipped_late_init = 0
        ensemble_onsets_found = 0

        # Loop over all combinations
        for t_idx, init_time in enumerate(init_times):
            if t_idx % 5 == 0:  # Print progress every 5 init times
                print(
                    f"Processing init time {t_idx + 1}/{len(init_times)}: {pd.to_datetime(init_time).strftime('%Y-%m-%d')}"
                )

            # Get init date for MOK filtering and onset comparison
            init_date = pd.to_datetime(init_time)
            year = init_date.year
            mok_date = datetime(year, mok_month, mok_day)  # June 2nd of the same year

            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    total_potential_inits += 1

                    # Get observed onset date for this grid point
                    try:
                        obs_onset = onset_da.isel(lat=i, lon=j).values
                    except (IndexError, KeyError):
                        skipped_no_obs += 1
                        continue

                    # Skip if no observed onset
                    if pd.isna(obs_onset):
                        skipped_no_obs += 1
                        continue

                    # Convert observed onset to datetime
                    obs_onset_dt = pd.to_datetime(obs_onset)

                    # Only process if forecast was initialized before observed onset
                    if init_date >= obs_onset_dt:
                        skipped_late_init += 1
                        continue

                    valid_inits += 1

                    # Get threshold for this grid point
                    thresh = thresh_slice.isel(lat=i, lon=j).values

                    # Collect onset days for all members at this init/location
                    member_onset_days = []

                    for m_idx, _member in enumerate(members):
                        try:
                            # Extract forecast time series for this member
                            forecast_series = (
                                p_model.isel(
                                    init_time=t_idx,
                                    lat=i,
                                    lon=j,
                                    member=m_idx,
                                )
                                .sel(step=slice(1, max_steps_needed))
                                .values
                            )

                            if len(forecast_series) < max_steps_needed:
                                member_onset_days.append(None)
                                continue

                            # Check for onset on each possible day
                            member_onset_day = None

                            for day in range(1, max_forecast_day + 1):
                                start_idx = day - 1
                                end_idx = start_idx + window

                                if end_idx <= len(forecast_series):
                                    window_series = forecast_series[start_idx:end_idx]

                                    # Check basic onset condition: first day > 1mm AND 5-day sum > threshold
                                    if (
                                        window_series[0] > 1
                                        and np.nansum(window_series) > thresh
                                    ):
                                        # Calculate the actual date this forecast day represents
                                        forecast_date = init_date + pd.Timedelta(
                                            days=day
                                        )

                                        # If MOK flag is True, only count onset if it's on or after June 2nd
                                        if mok:
                                            if forecast_date.date() > mok_date.date():
                                                member_onset_day = day
                                                break  # Found valid onset after MOK date
                                            # else: continue checking later days
                                        else:
                                            # No MOK filtering, count this onset
                                            member_onset_day = day
                                            break

                            member_onset_days.append(member_onset_day)

                        except Exception as e:
                            print(
                                f"Error at init_time {t_idx}, lat {i}, lon {j}, member {m_idx}: {e}"
                            )
                            member_onset_days.append(None)

                    # Now check if at least 50% of members have onset
                    valid_onsets = [day for day in member_onset_days if day is not None]
                    onset_count = len(valid_onsets)
                    total_members = len(member_onset_days)
                    onset_percentage = (
                        onset_count / total_members if total_members > 0 else 0
                    )

                    # Determine ensemble onset day
                    ensemble_onset_day = None
                    ensemble_onset_date = None
                    if onset_percentage >= 0.5:  # At least 50% of members have onset
                        # Use rounding of mean onset day
                        mean_onset = np.mean(valid_onsets)
                        ensemble_onset_day = int(round(mean_onset))
                        ensemble_onsets_found += 1
                        ensemble_onset_date = init_date + pd.Timedelta(
                            days=ensemble_onset_day
                        )

                    # Store result
                    result = {
                        "init_time": init_time,
                        "lat": lat,
                        "lon": lon,
                        "onset_day": ensemble_onset_day,  # None if <50% members have onset
                        "onset_date": ensemble_onset_date.strftime("%Y-%m-%d")
                        if ensemble_onset_date is not None
                        else None,
                        "member_onset_count": onset_count,
                        "total_members": total_members,
                        "onset_percentage": onset_percentage,
                        "obs_onset_date": obs_onset_dt.strftime(
                            "%Y-%m-%d"
                        ),  # Store observed onset for reference
                    }
                    results_list.append(result)

        # Convert to DataFrame
        onset_df = pd.DataFrame(results_list)

        print("\nProcessing Summary:")
        print(f"Total potential initializations: {total_potential_inits}")
        print(f"Skipped (no observed onset): {skipped_no_obs}")
        print(f"Skipped (initialized after observed onset): {skipped_late_init}")
        print(f"Valid initializations processed: {valid_inits}")
        print(f"Ensemble onsets found (≥50% members): {ensemble_onsets_found}")
        print(
            f"Ensemble onset rate: {ensemble_onsets_found / valid_inits:.3f}"
            if valid_inits > 0
            else "Ensemble onset rate: 0.000"
        )

        if mok:
            print(
                f"Note: Only onsets on or after {mok_month}/{mok_day} were counted due to MOK flag"
            )

        return onset_df

    @staticmethod
    def compute_metrics_multiple_years(
        years,
        model_forecast_dir,
        imd_folder,
        thres_file,
        tolerance_days=3,
        verification_window=1,
        forecast_days=15,
        max_forecast_day=15,
        mok=True,
        onset_window=5,
        mok_month=6,
        mok_day=2,
    ):
        """Compute onset metrics for multiple years."""
        metrics_df_dict = {}
        onset_da_dict = {}

        thresh_ds = xr.open_dataset(thres_file)
        thres_da = thresh_ds["MWmean"]

        for year in years:
            print(f"\n{'=' * 50}")
            print(f"Processing year {year}")
            print(f"{'=' * 50}")

            p_model = ProbabilisticOnsetMetrics.get_forecast_probabilistic_twice_weekly(
                year, model_forecast_dir
            )
            imd = OnsetMetricsBase.load_imd_rainfall(year, imd_folder)
            onset_da = OnsetMetricsBase.detect_observed_onset(
                imd, thres_da, year, mok=mok
            )

            onset_df = ProbabilisticOnsetMetrics.compute_mean_onset_for_all_members(
                p_model,
                thres_da,
                onset_da,
                max_forecast_day=max_forecast_day,
                mok=mok,
                onset_window=onset_window,
                mok_month=mok_month,
                mok_day=mok_day,
            )

            metrics_df, summary_stats = (
                OnsetMetricsBase.compute_onset_metrics_with_windows(
                    onset_df,
                    tolerance_days=tolerance_days,
                    verification_window=verification_window,
                    forecast_days=forecast_days,
                )
            )

            metrics_df_dict[year] = metrics_df
            onset_da_dict[year] = onset_da

            print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")

        return metrics_df_dict, onset_da_dict

    @staticmethod
    def compute_metrics_multiple_years_from_loaders(
        tp_forecast: xr.DataArray,  # (day, time, lat, lon, member)
        tp_imd: xr.DataArray,  # (time, lat, lon)
        thres_da: xr.DataArray,  # (lat, lon)
        years=None,
        tolerance_days: int = 3,
        verification_window: int = 1,
        forecast_days: int = 15,
        max_forecast_day: int = 15,
        mok: bool = True,
        onset_window: int = 5,
        mok_month: int = 6,
        mok_day: int = 2,
    ):
        """Loader-based version of "compute_onset_metrics_for_multiple_years" using

        three *loaded* DataArrays:

            - tp_forecast: Probabilistic model precip, dims ('day', 'time', 'lat', 'lon', 'member').
            - tp_imd: Observed precip, dims ('time', 'lat', 'lon').
            - thres_da: Threshold field, dims ('lat', 'lon').

        Years are inferred from tp_forecast['time'] if not provided.
        """
        if "number" in tp_forecast.dims:
            tp_forecast = tp_forecast.rename({"number": "member"})
        elif "sample" in tp_forecast.dims:
            tp_forecast = tp_forecast.rename({"sample": "member"})

        metrics_df_dict = {}
        onset_da_dict = {}

        # Infer which years to process from forecast init_time ---
        init_times_all = pd.to_datetime(tp_forecast["time"].values)
        if years is None:
            years = sorted(np.unique(init_times_all.year))
        else:
            years = sorted(int(y) for y in years)

        # Loop over years and reuse existing logic
        for year in years:
            print("\n" + "=" * 50)
            print(f"Processing year {year}")
            print("=" * 50)

            # Slice forecasts for this year based on init_time
            year_mask = init_times_all.year == year
            year_init_times = init_times_all[year_mask]

            if len(year_init_times) == 0:
                print(f"No init times for year {year} in forecast data, skipping.")
                continue

            tp_fc_year = tp_forecast.sel(time=year_init_times)

            p_model = tp_fc_year.rename({"time": "init_time", "day": "step"}).transpose(
                "init_time", "step", "lat", "lon", "member"
            )

            # Drop step=0 if present (lead 0), same as in get_forecast_probabilistic_twice_weekly
            if int(p_model.step[0]) == 0:
                p_model = p_model.sel(step=slice(1, None))

            # Slice IMD rainfall for this year and load into memory
            tp_imd_year = tp_imd.sel(
                time=slice(f"{year}-01-01", f"{year}-12-31")
            ).load()

            # Detect_observed_onset expects rainfall with dims (time, lat, lon)
            onset_da = OnsetMetricsBase.detect_observed_onset(
                tp_imd_year, thres_da, year, mok=mok
            )

            # Existing probabilistic onset logic (ensemble -> ensemble_onset_day per init/lat/lon)
            onset_df = ProbabilisticOnsetMetrics.compute_mean_onset_for_all_members(
                p_model,
                thres_da,
                onset_da,
                max_forecast_day=max_forecast_day,
                mok=mok,
                onset_window=onset_window,
                mok_month=mok_month,
                mok_day=mok_day,
            )

            # Existing contingency-table metrics (MAE/FAR/MR)
            metrics_df, summary_stats = (
                OnsetMetricsBase.compute_onset_metrics_with_windows(
                    onset_df,
                    tolerance_days=tolerance_days,
                    verification_window=verification_window,
                    forecast_days=forecast_days,
                )
            )

            metrics_df_dict[year] = metrics_df
            onset_da_dict[year] = onset_da

            print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")

        return metrics_df_dict, onset_da_dict
