"""Probabilistic model onset metrics computation.

This module provides the ProbabilisticOnsetMetrics class for computing
onset metrics from probabilistic ensemble model forecasts.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from monsoonbench.spatial.regions import points_inside_polygon

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
        # Failsafe for no-ensemble number specified data
        else:
            ds = ds.expand_dims(member=np.arange(4))
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
    def get_forecast_probabilistic_twice_weekly_2(
        yr,
        model_forecast_dir,
        mem_num,
        date_filter_year=2024,
        file_pattern="tp_4p0_{}.nc",
    ):
        """Loads model precip data for twice-weekly initializations from May to July."""
        fname = file_pattern.format(yr)
        file_path = os.path.join(model_forecast_dir, fname)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Filter for twice weekly data from daily for the specified year
        start_date = datetime(date_filter_year, 5, 1)
        end_date = datetime(date_filter_year, 7, 31)
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
        if "sample" in ds.dims:
            ds = ds.rename({"sample": "member"})
        else:
            ds = ds.expand_dims(member=np.arange(4))

        # Find common dates between desired dates and available dates
        available_init_times = pd.to_datetime(ds.init_time.values)
        matching_times = available_init_times[
            available_init_times.isin(filtered_dates_yr)
        ]

        if len(matching_times) == 0:
            raise ValueError(f"No matching initialization times found for year {yr}")

        # Select only the matching initialization times
        ds = ds.sel(init_time=matching_times)
        if "total_precipitation_24hr" in ds.data_vars:
            ds = ds.rename(
                {"total_precipitation_24hr": "tp"}
            )  # For the quantile-mapped variable change the var name from total_precipitation_24hr to tp
            ds = ds[["tp"]] * 1000  # Convert from m to mm
        if "day" in ds.dims:
            if ds["day"][0].values == 0:
                ds = ds.sel(day=slice(1, None))

        if "step" in ds.dims:
            if ds["step"][0].values == 0:
                ds = ds.sel(step=slice(1, None))

        if "day" in ds.dims:
            ds = ds.rename({"day": "step"})

        ds = ds.isel(
            member=slice(0, mem_num)
        )  # limit to first mem_num members (0-mem_num)
        p_model = ds["tp"]  # in mm
        init_times = p_model.init_time.values
        ds.close()
        return p_model, init_times

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

    # Function to compute onset dates for all ensemble members and save it as DataFrame
    @staticmethod
    def compute_onset_for_all_members(
        p_model, thresh_slice, onset_da, max_forecast_day=15, mok=True
    ):
        """Compute onset dates for each ensemble member, initialization time, and grid point."""
        window = 5
        results_list = []

        # Get dimensions
        init_times = p_model.init_time.values
        members = p_model.member.values

        # Get the actual lat/lon coordinates from the data
        lats = p_model.lat.values
        lons = p_model.lon.values

        # Create unique lat-lon pairs (no repetition)
        unique_pairs = list(zip(lons, lats))

        date_method = "MOK (June 2nd filter)" if mok else "no date filter"
        print(
            f"Processing {len(init_times)} init times x {len(unique_pairs)} unique locations x {len(members)} members..."
        )
        print(f"Unique lat-lon pairs: {unique_pairs}")
        print(f"Using {date_method} for onset detection")

        max_steps_needed = max_forecast_day + window - 1

        # Track statistics
        total_potential_forecasts = 0
        valid_forecasts = 0
        skipped_no_obs = 0
        skipped_late_init = 0

        # Loop over all combinations
        for t_idx, init_time in enumerate(init_times):
            if t_idx % 5 == 0:
                print(
                    f"Processing init time {t_idx+1}/{len(init_times)}: {pd.to_datetime(init_time).strftime('%Y-%m-%d')}"
                )

            init_date = pd.to_datetime(init_time)
            year = init_date.year
            mok_date = datetime(year, 6, 2)

            # Loop over unique lat-lon pairs only
            for loc_idx, (lon, lat) in enumerate(unique_pairs):
                total_potential_forecasts += len(members)

                # Get observed onset date for this location
                try:
                    obs_onset = onset_da.isel(lat=loc_idx, lon=loc_idx).values
                except:
                    skipped_no_obs += len(members)
                    continue

                # Skip if no observed onset
                if pd.isna(obs_onset):
                    skipped_no_obs += len(members)
                    continue

                # Convert observed onset to datetime
                obs_onset_dt = pd.to_datetime(obs_onset)

                # Only process if forecast was initialized before observed onset
                if init_date >= obs_onset_dt:
                    skipped_late_init += len(members)
                    continue

                # Get threshold for this location
                thresh = thresh_slice.isel(lat=loc_idx, lon=loc_idx).values

                for m_idx, member in enumerate(members):
                    valid_forecasts += 1

                    try:
                        # Extract forecast time series for this member and location
                        forecast_series = p_model.isel(
                            init_time=t_idx,
                            lat=loc_idx,
                            lon=loc_idx,
                            member=m_idx,
                            step=slice(0, max_steps_needed),
                        ).values

                        if len(forecast_series) < max_steps_needed:
                            continue

                        # Check for onset on each possible day
                        onset_day = None

                        for day in range(1, max_forecast_day + 1):
                            start_idx = day - 1
                            end_idx = start_idx + window

                            if end_idx <= len(forecast_series):
                                window_series = forecast_series[start_idx:end_idx]

                                # Check basic onset condition
                                if (
                                    window_series[0] > 1
                                    and np.nansum(window_series) > thresh
                                ):
                                    # Calculate the actual date this forecast day represents
                                    forecast_date = init_date + pd.Timedelta(days=day)

                                    # If MOK flag is True, only count onset if it's on or after June 2nd
                                    if mok:
                                        if forecast_date.date() > mok_date.date():
                                            onset_day = day
                                            break
                                    else:
                                        onset_day = day
                                        break

                        # Store result
                        result = {
                            "init_time": init_time,
                            "lat": lat,
                            "lon": lon,
                            "member": member,
                            "onset_day": onset_day,
                            "obs_onset_date": obs_onset_dt.strftime("%Y-%m-%d"),
                        }
                        results_list.append(result)

                    except Exception as e:
                        print(
                            f"Error at init_time {t_idx}, location ({lon}, {lat}), member {m_idx}: {e}"
                        )
                        continue

        # Convert to DataFrame
        onset_df = pd.DataFrame(results_list)

        print("\nProcessing Summary:")
        print(f"Total potential forecasts: {total_potential_forecasts}")
        print(f"Skipped (no observed onset): {skipped_no_obs}")
        print(f"Skipped (initialized after observed onset): {skipped_late_init}")
        print(f"Valid forecasts processed: {valid_forecasts}")
        print(f"Generated {len(onset_df)} member-forecast combinations")
        print(f"Found onset in {onset_df['onset_day'].notna().sum()} cases")
        print(f"Onset rate: {onset_df['onset_day'].notna().mean():.3f}")

        # Check for uniqueness
        unique_combinations = onset_df.groupby(
            ["init_time", "lat", "lon", "member"]
        ).size()
        if (unique_combinations > 1).any():
            print(
                f"Warning: Found {(unique_combinations > 1).sum()} duplicate combinations!"
            )
        else:
            print("✓ All init_time-lat-lon-member combinations are unique")
        0
        return onset_df

    # ADD TO REPO
    # Function to create forecast-observation pairs with specified day bins for probabilistic verification
    @staticmethod
    def create_forecast_observation_pairs_with_bins(
        onset_all_members, onset_da, day_bins, max_forecast_day=15
    ):
        """Create forecast-observation pairs using specified day bins, including a final bin for "after max_forecast_day".

        Parameters:
        -----------
        onset_all_members : DataFrame
            DataFrame with ensemble member onset predictions
        onset_da : xarray.DataArray
            Observed onset dates
        day_bins : list of tuples
            List of (start_day, end_day) tuples for bins within forecast window
            e.g., [(1, 5), (6, 10), (11, 15)]
        max_forecast_day : int, default=15
            Maximum forecast day. Members without onset get assigned to "after day X" bin
        """
        results_list = []

        # Get unique combinations of init_time, lat, lon from the filtered forecast data
        forecast_groups = onset_all_members.groupby(["init_time", "lat", "lon"])

        # Add the "after max_forecast_day" bin
        extended_bins = day_bins + [(max_forecast_day + 1, float("inf"))]

        print(
            f"Processing {len(forecast_groups)} forecast cases with day bins: {day_bins}"
        )
        print(
            f"Including 'after day {max_forecast_day}' bin for members without onset in forecast window"
        )

        for (init_time, lat, lon), group in forecast_groups:
            # Get observed onset for this location
            try:
                lat_idx = np.where(np.abs(onset_da.lat.values - lat) < 0.01)[0][0]
                lon_idx = np.where(np.abs(onset_da.lon.values - lon) < 0.01)[0][0]
                obs_date = onset_da.isel(lat=lat_idx, lon=lon_idx).values
            except:
                continue

            # Skip if no observed onset
            if pd.isna(obs_date):
                continue

            # Convert dates for comparison
            init_date = pd.to_datetime(init_time)
            obs_date_dt = pd.to_datetime(obs_date)

            # Double-check: Only use forecasts initialized before the observed onset
            if init_date >= obs_date_dt:
                continue

            # For each day bin (including the "after max_forecast_day" bin)
            for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
                # Handle the "after max_forecast_day" bin differently
                if bin_start > max_forecast_day:
                    bin_label = f"After day {max_forecast_day}"

                    # Check if observed onset occurs after max_forecast_day
                    forecast_end_date = init_date + pd.Timedelta(days=max_forecast_day)
                    observed_onset = int(obs_date_dt.date() > forecast_end_date.date())

                    # Count members that didn't predict onset within forecast window
                    members_with_onset_in_bin = 0
                    total_members = len(group)

                    for member_idx, member_row in group.iterrows():
                        member_onset_day = member_row["onset_day"]

                        # Member predicts "after day X" if onset_day is NaN or > max_forecast_day
                        if (
                            pd.isna(member_onset_day)
                            or member_onset_day > max_forecast_day
                        ):
                            members_with_onset_in_bin += 1

                else:
                    # Regular bin within forecast window
                    bin_label = f"Days {bin_start}-{bin_end}"

                    # Calculate the date range for this bin
                    bin_start_date = init_date + pd.Timedelta(days=bin_start)
                    bin_end_date = init_date + pd.Timedelta(days=bin_end)

                    # Check if observed onset falls within this day bin
                    observed_onset = int(
                        bin_start_date.date()
                        <= obs_date_dt.date()
                        <= bin_end_date.date()
                    )

                    # Calculate ensemble probability for this day bin
                    members_with_onset_in_bin = 0
                    total_members = len(group)

                    for member_idx, member_row in group.iterrows():
                        member_onset_day = member_row["onset_day"]

                        if (
                            pd.notna(member_onset_day)
                            and bin_start <= member_onset_day <= bin_end
                        ):
                            members_with_onset_in_bin += 1

                # Calculate probability
                predicted_prob = members_with_onset_in_bin / total_members

                # Store result
                result = {
                    "init_time": init_time,
                    "lat": lat,
                    "lon": lon,
                    "bin_start": bin_start
                    if bin_start <= max_forecast_day
                    else max_forecast_day + 1,
                    "bin_end": bin_end if bin_end <= max_forecast_day else float("inf"),
                    "bin_label": bin_label,
                    "predicted_prob": predicted_prob,
                    "observed_onset": observed_onset,
                    "members_with_onset": members_with_onset_in_bin,
                    "total_members": total_members,
                    "year": pd.to_datetime(init_time).year,
                    "obs_onset_date": obs_date_dt.strftime("%Y-%m-%d"),
                    "bin_index": bin_idx,
                }
                results_list.append(result)

        # Convert to DataFrame
        forecast_obs_df = pd.DataFrame(results_list)

        print(f"Generated {len(forecast_obs_df)} forecast-observation pairs")
        print(f"Total bins per forecast: {len(extended_bins)}")
        print(
            f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}"
        )
        print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
        print(
            f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}"
        )

        # Show distribution across bins
        print("\nDistribution across bins:")
        bin_stats = (
            forecast_obs_df.groupby("bin_label")
            .agg({"predicted_prob": ["count", "mean"], "observed_onset": "mean"})
            .round(3)
        )
        print(bin_stats)

        return forecast_obs_df

    # New integrations of Rajat's code
    # ADD TO REPO (forecast_obs_df)
    # This function creates the observed forecast pairs for multiple years (core monsoon zone grids) and combines them
    @staticmethod
    def multi_year_forecast_obs_pairs(
        years: list[int],
        model_forecast_dir: str | Path,
        imd_folder: str | Path,
        thres_file: str | Path,
        mem_num: int,
        max_forecast_day: int,
        day_bins: list[int],
        mok: bool = True,
        date_filter_year: int = 2024,
        file_pattern: str = "{}.nc",
    ):
        """Main function to perform multi-year reliability analysis.

        Args:
            years: Iterable of years to process (e.g., [2019, 2020, 2021]).
            model_forecast_dir: Directory containing model forecast NetCDF files.
            imd_folder: Directory containing IMD observation files.
            thres_file: Path to threshold file used for onset calculations.
            mem_num: Number of ensemble members.
            max_forecast_day: Maximum lead time (days) to consider.
            day_bins: Forecast day bins used for aggregation.
            mok: Whether to apply MOK-specific logic.
            date_filter_year: Year used to filter dates (default: 2024).
            file_pattern: Filename pattern for NetCDF files (default: "{}.nc").
        """
        print(f"Processing years: {years}")

        # Load threshold data (same for all years)
        thresh_ds = xr.open_dataset(thres_file)
        thresh_da = thresh_ds["MWmean"]
        orig_lat = thresh_da.lat.values
        orig_lon = thresh_da.lon.values

        lat_diff = abs(orig_lat[1] - orig_lat[0])
        if abs(lat_diff - 2.0) < 0.1:  # 2-degree resolution
            polygon1_lon = np.array(
                [83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83]
            )
            polygon1_lat = np.array(
                [17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17]
            )
            print("Using 2-degree CMZ polygon coordinates")
        elif abs(lat_diff - 4.0) < 0.1:  # 4-degree resolution
            polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
            polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
            print("Using 4-degree CMZ polygon coordinates")
        elif abs(lat_diff - 1.0) < 0.1:  # 1-degree resolution
            polygon1_lon = np.array(
                [
                    74,
                    85,
                    85,
                    86,
                    86,
                    87,
                    87,
                    88,
                    88,
                    88,
                    85,
                    85,
                    82,
                    82,
                    79,
                    79,
                    78,
                    78,
                    69,
                    69,
                    74,
                    74,
                ]
            )
            polygon1_lat = np.array(
                [
                    18,
                    18,
                    19,
                    19,
                    20,
                    20,
                    21,
                    21,
                    21,
                    24,
                    24,
                    25,
                    25,
                    26,
                    26,
                    27,
                    27,
                    28,
                    28,
                    21,
                    21,
                    18,
                ]
            )
            print("Using 1-degree CMZ polygon coordinates")

        inside_mask, inside_lons, inside_lats = points_inside_polygon(
            polygon1_lon, polygon1_lat, orig_lon, orig_lat
        )
        thresh_slice = thresh_da.sel(lat=inside_lats, lon=inside_lons)

        # Initialize list to store all forecast-observation pairs
        all_forecast_obs_pairs = []

        # Process each year
        for year in years:
            print(f"\n{'='*50}")
            print(f"Processing year {year}")
            print(f"{'='*50}")

            try:
                # Load model and observation data
                print("Loading S2S model data...")
                p_model, _ = (
                    ProbabilisticOnsetMetrics.get_forecast_probabilistic_twice_weekly_2(
                        year,
                        model_forecast_dir,
                        mem_num,
                        date_filter_year,
                        file_pattern,
                    )
                )
                p_model_slice = p_model.sel(lat=inside_lats, lon=inside_lons)

                print("Loading IMD rainfall data...")
                rainfall_ds = OnsetMetricsBase.load_imd_rainfall(year, imd_folder)
                rainfall_ds_slice = rainfall_ds.sel(lat=inside_lats, lon=inside_lons)
                print("Detecting observed onset...")
                onset_da = OnsetMetricsBase.detect_observed_onset(
                    rainfall_ds_slice, thresh_slice, year, mok
                )
                print(
                    f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points"
                )

                print("Computing onset for all ensemble members...")
                onset_all_members = (
                    ProbabilisticOnsetMetrics.compute_onset_for_all_members(
                        p_model_slice,
                        thresh_slice,
                        onset_da,
                        max_forecast_day=max_forecast_day,
                        mok=True,
                    )
                )
                print(
                    f"Found onset in {onset_all_members['onset_day'].notna().sum()} member cases"
                )

                print("Creating forecast-observation pairs...")
                forecast_obs_pairs = ProbabilisticOnsetMetrics.create_forecast_observation_pairs_with_bins(
                    onset_all_members,
                    onset_da,
                    day_bins,
                    max_forecast_day=max_forecast_day,
                )

                # Add to master list
                all_forecast_obs_pairs.append(forecast_obs_pairs)

                print(
                    f"Year {year} completed: {len(forecast_obs_pairs)} forecast-observation pairs"
                )

            except Exception as e:
                print(f"Error processing year {year}: {e}")
                continue

        # Combine all years
        print(f"\n{'='*50}")
        print("Combining all years")
        print(f"{'='*50}")

        if not all_forecast_obs_pairs:
            raise ValueError("No data was successfully processed for any year")

        combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

        # Print final summary statistics
        print("\nFinal Summary Statistics:")
        print(f"Years processed: {years}")
        return combined_forecast_obs

    # Function to calculate Brier Score and Fair Brier Score for the model forecasts (both overall and bin-wise)
    @staticmethod
    def calculate_brier_score(forecast_obs_df):
        """Calculate Brier Score and Fair Brier Score for probabilistic forecasts.

        Brier Score = (1/n*m) * Σ(Y_ij - p_ij)²
        Fair Brier Score = (1/n*m) * Σ[(Y_ij - p_ij)² - p_ij(1-p_ij)/(ens-1)]

        where:
        - n = number of forecasts
        - m = number of bins per forecast
        - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
        - p_ij = predicted probability for bin j in forecast i
        - ens = number of ensemble members

        Parameters:
        -----------
        forecast_obs_df : DataFrame
            Output from create_forecast_observation_pairs_with_bins()
            Must contain columns: 'predicted_prob', 'observed_onset', 'total_members'

        Returns:
        --------
        dict with Brier score metrics
        """
        # Calculate squared differences
        squared_diffs = (
            forecast_obs_df["observed_onset"] - forecast_obs_df["predicted_prob"]
        ) ** 2

        # Calculate overall Brier Score
        brier_score = squared_diffs.mean()

        # Calculate Fair Brier Score correction term
        # ens-1 where ens is the number of ensemble members
        correction_term = (
            forecast_obs_df["predicted_prob"] * (1 - forecast_obs_df["predicted_prob"])
        ) / (forecast_obs_df["total_members"] - 1)

        # Fair Brier Score
        fair_brier_components = squared_diffs - correction_term
        fair_brier_score = fair_brier_components.mean()
        # Calculate squared differences for bin-wise analysis
        forecast_obs_df["squared_diff"] = squared_diffs
        forecast_obs_df["fair_brier_component"] = fair_brier_components

        # Bin-wise Brier scores
        bin_brier_scores = forecast_obs_df.groupby("bin_label")["squared_diff"].mean()
        bin_fair_brier_scores = forecast_obs_df.groupby("bin_label")[
            "fair_brier_component"
        ].mean()

        brier_results = {
            "brier_score": brier_score,
            "fair_brier_score": fair_brier_score,
            "bin_brier_scores": bin_brier_scores.to_dict(),
            "bin_fair_brier_scores": bin_fair_brier_scores.to_dict(),
        }

        print(f"Brier Score: {brier_results['brier_score']:.4f}")
        print(f"Fair Brier Score: {brier_results['fair_brier_score']:.4f}")

        return brier_results

    # Function to calculate Area Under the Curve (AUC) for the model forecasts (both overall and bin-wise)
    @staticmethod
    def calculate_auc(forecast_obs_df):
        """Calculate Area Under the Curve (AUC) for probabilistic forecasts.

        AUC = Σ_{i,j,i',j'} Y_{ij}(1-Y_{i'j'}) · 1[p_{ij} > p_{i'j'}] /
            [(Σ_{i,j} Y_{ij})(Σ_{i,j} (1-Y_{ij}))]

        where:
        - Y_{ij} = 1 if onset occurred in bin j for forecast i, 0 otherwise
        - p_{ij} = predicted probability for bin j in forecast i
        - 1[p_{ij} > p_{i'j'}] = indicator function (1 if true, 0 if false)

        Parameters:
        -----------
        forecast_obs_df : DataFrame
            Output from create_forecast_observation_pairs_with_bins()
            Must contain columns: 'predicted_prob', 'observed_onset'

        Returns:
        --------
        dict with AUC metrics
        """
        # Extract probabilities and observations
        p_ij = forecast_obs_df["predicted_prob"].values
        y_ij = forecast_obs_df["observed_onset"].values

        # Count total positive and negative cases
        n_positive = np.sum(y_ij)  # Σ Y_{ij}
        n_negative = np.sum(1 - y_ij)  # Σ (1-Y_{ij})

        if n_positive == 0 or n_negative == 0:
            print(
                "Warning: Cannot calculate AUC - all cases are either positive or negative"
            )
            return {
                "auc": np.nan,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "bin_auc_scores": {},
                "forecast_obs_df_with_ranks": forecast_obs_df,
            }

        # Calculate AUC using the Mann-Whitney U statistic approach
        # This is equivalent to the formula but more computationally efficient

        # Separate positive and negative cases
        positive_probs = p_ij[y_ij == 1]
        negative_probs = p_ij[y_ij == 0]

        # Calculate Mann-Whitney U statistic
        u_statistic, _ = stats.mannwhitneyu(
            positive_probs, negative_probs, alternative="greater"
        )

        # AUC is U statistic divided by (n_positive * n_negative)
        auc = u_statistic / (n_positive * n_negative)

        # Alternative direct calculation (less efficient for large datasets)
        # concordant_pairs = 0
        # for i in range(len(p_ij)):
        #     if y_ij[i] == 1:  # positive case
        #         for j in range(len(p_ij)):
        #             if y_ij[j] == 0:  # negative case
        #                 if p_ij[i] > p_ij[j]:
        #                     concordant_pairs += 1
        # auc_direct = concordant_pairs / (n_positive * n_negative)

        # Calculate AUC by bin
        bin_auc_scores = {}
        unique_bins = forecast_obs_df["bin_label"].unique()

        for bin_label in unique_bins:
            bin_data = forecast_obs_df[forecast_obs_df["bin_label"] == bin_label]

            if len(bin_data) > 0:
                bin_p = bin_data["predicted_prob"].values
                bin_y = bin_data["observed_onset"].values

                bin_n_positive = np.sum(bin_y)
                bin_n_negative = np.sum(1 - bin_y)

                if bin_n_positive > 0 and bin_n_negative > 0:
                    bin_positive_probs = bin_p[bin_y == 1]
                    bin_negative_probs = bin_p[bin_y == 0]

                    bin_u_stat, _ = stats.mannwhitneyu(
                        bin_positive_probs, bin_negative_probs, alternative="greater"
                    )
                    bin_auc = bin_u_stat / (bin_n_positive * bin_n_negative)
                else:
                    bin_auc = np.nan

                bin_auc_scores[bin_label] = bin_auc

        auc_results = {
            "auc": auc,
            "bin_auc_scores": bin_auc_scores,
        }

        return auc_results

    # Function to calculate Ranked Probability Score (RPS) and Fair RPS for the model forecasts for (either 15 day or 30 day forecast)
    @staticmethod
    def calculate_rps(forecast_obs_df):
        """Calculate Ranked Probability Score (RPS) and Fair RPS for probabilistic forecasts.

        RPS = (1/n*m) * Σ_i Σ_k (Σ_j≤k (Y_ij - p_ij))²
        Fair RPS = (1/n*m) * Σ_i Σ_k [(Σ_j≤k (Y_ij - p_ij))² - (Σ_j≤k p_ij)(1 - Σ_j≤k p_ij)/(ens-1)]

        where:
        - n = number of forecasts
        - m = number of bins per forecast
        - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
        - p_ij = predicted probability for bin j in forecast i
        - k = cumulative index (1 to m)
        - ens = number of ensemble members

        Parameters:
        -----------
        forecast_obs_df : DataFrame
            Output from create_forecast_observation_pairs_with_bins()
            Must contain columns: 'predicted_prob', 'observed_onset', 'total_members', 'bin_index'

        Returns:
        --------
        dict with RPS metrics
        """
        # Group by forecast (init_time, lat, lon) to get all bins for each forecast
        forecast_groups = forecast_obs_df.groupby(["init_time", "lat", "lon"])

        rps_values = []
        fair_rps_values = []

        for (init_time, lat, lon), group in forecast_groups:
            # Sort by bin_index to ensure proper ordering
            group_sorted = group.sort_values("bin_index")

            # Get predicted probabilities and observations for this forecast
            p_ij = group_sorted["predicted_prob"].values
            y_ij = group_sorted["observed_onset"].values
            total_members = group_sorted["total_members"].iloc[
                0
            ]  # Same for all bins in forecast

            m = len(p_ij)  # Number of bins

            # Calculate RPS for this forecast
            rps_forecast = 0
            fair_rps_forecast = 0

            for k in range(1, m + 1):  # k from 1 to m
                # Cumulative sum up to bin k
                cum_p = np.sum(p_ij[:k])
                cum_y = np.sum(y_ij[:k])

                # RPS component
                diff_cum = cum_y - cum_p
                rps_component = diff_cum**2
                rps_forecast += rps_component

                # Fair RPS correction term
                fair_correction = (cum_p * (1 - cum_p)) / (total_members - 1)
                fair_rps_component = rps_component - fair_correction
                fair_rps_forecast += fair_rps_component

            rps_values.append(rps_forecast)
            fair_rps_values.append(fair_rps_forecast)

        # Calculate overall RPS (average over all forecasts)
        rps = np.mean(rps_values)
        fair_rps = np.mean(fair_rps_values)

        rps_results = {
            "rps": rps,
            "fair_rps": fair_rps,
            "n_forecasts": len(forecast_groups),
        }

        print(f"RPS: {rps_results['rps']:.4f}")
        print(f"Fair RPS: {rps_results['fair_rps']:.4f}")
        print(f"Number of forecasts: {rps_results['n_forecasts']}")

        return rps_results

    @staticmethod
    def calculate_skill_scores(
        brier_forecast, rps_forecast, brier_climatology, rps_climatology
    ):
        """Calculate skill scores for forecast model relative to climatology.

        Skill Score = 1 - (forecast_score / climatology_score)

        Parameters:
        -----------
        brier_forecast : dict
            Brier score results from forecast model
        rps_forecast : dict
            RPS results from forecast model
        brier_climatology : dict
            Brier score results from climatology
        rps_climatology : dict
            RPS results from climatology

        Returns:
        --------
        dict with skill scores
        """
        skill_scores = {}

        print("=" * 60)
        print("SKILL SCORE CALCULATIONS")
        print("=" * 60)

        # Fair Brier Skill Score (1-15 day overall)
        fair_bss_overall = 1 - (
            brier_forecast["fair_brier_score"] / brier_climatology["fair_brier_score"]
        )
        skill_scores["fair_brier_skill_score"] = fair_bss_overall

        print(f"Fair Brier Skill Score (1-15 day): {fair_bss_overall:.4f}")

        # Fair RPS Skill Score (1-15 day overall)
        fair_rpss_overall = 1 - (rps_forecast["fair_rps"] / rps_climatology["fair_rps"])
        skill_scores["fair_rps_skill_score"] = fair_rpss_overall

        print(f"Fair RPS Skill Score (1-15 day): {fair_rpss_overall:.4f}")

        # Automatically extract target bins from the data, excluding unwanted bins
        all_forecast_bins = set(brier_forecast["bin_fair_brier_scores"].keys())
        all_clim_bins = set(brier_climatology["bin_fair_brier_scores"].keys())

        # Get intersection of bins present in both forecast and climatology
        common_bins = all_forecast_bins.intersection(all_clim_bins)

        # Filter out unwanted bins and keep only "Days X-Y" format bins
        target_bins = []
        excluded_bins = []

        for bin_label in common_bins:
            # Include only bins that start with "Days " and don't contain "After" or "Before"
            if (
                bin_label.startswith("Days ")
                and not bin_label.startswith("After")
                and not bin_label.startswith("Before")
            ):
                target_bins.append(bin_label)
            else:
                excluded_bins.append(bin_label)

        # Sort bins by their day ranges
        def extract_day_range(bin_label):
            # Extract the start day from "Days X-Y" format
            if "Days " in bin_label:
                try:
                    day_part = bin_label.replace("Days ", "").split("-")[0]
                    return int(day_part)
                except:
                    return 999  # Put unparseable bins at the end
            return 999

        target_bins = sorted(target_bins, key=extract_day_range)

        print(f"\nAutomatically detected target bins: {target_bins}")
        print(f"Excluded bins: {excluded_bins}")

        # Bin-wise Fair Brier Skill Scores
        bin_fair_bss = {}

        print("\nBin-wise Fair Brier Skill Scores:")
        for bin_label in target_bins:
            if (
                bin_label in brier_forecast["bin_fair_brier_scores"]
                and bin_label in brier_climatology["bin_fair_brier_scores"]
            ):
                forecast_fair_brier_bin = brier_forecast["bin_fair_brier_scores"][
                    bin_label
                ]
                clim_fair_brier_bin = brier_climatology["bin_fair_brier_scores"][
                    bin_label
                ]

                fair_bss_bin = 1 - (forecast_fair_brier_bin / clim_fair_brier_bin)
                bin_fair_bss[bin_label] = fair_bss_bin

                print(f"  {bin_label}: Fair BSS = {fair_bss_bin:.4f}")
            else:
                bin_fair_bss[bin_label] = np.nan
                print(f"  {bin_label}: Fair BSS = NaN (missing data)")

        skill_scores["bin_fair_brier_skill_scores"] = bin_fair_bss

        # Dynamic table header based on detected bins
        header = f"{'Metric':<30} {'Overall (1-15 day)':<18}"
        for bin_name in target_bins:
            # Shorten bin names for table display
            short_name = bin_name.replace("Days ", "")
            header += f" {short_name:<12}"

        # Calculate table width
        table_width = 30 + 18 + 12 * len(target_bins)

        # Summary table
        print("\n" + "=" * table_width)
        print("SKILL SCORE SUMMARY TABLE")
        print("=" * table_width)
        print(header)
        print("-" * table_width)

        # Fair Brier Skill Score row
        fair_bss_row = f"{'Fair Brier Skill Score':<30} {fair_bss_overall:<18.4f}"
        for bin_name in target_bins:
            if bin_name in bin_fair_bss and not pd.isna(bin_fair_bss[bin_name]):
                fair_bss_row += f" {bin_fair_bss[bin_name]:<12.4f}"
            else:
                fair_bss_row += f" {'N/A':<12}"
        print(fair_bss_row)

        # Fair RPS Skill Score row
        fair_rpss_row = f"{'Fair RPS Skill Score':<30} {fair_rpss_overall:<18.4f}"
        for bin_name in target_bins:
            fair_rpss_row += f" {'N/A':<12}"  # RPS is overall only
        print(fair_rpss_row)

        print("-" * table_width)

        # Add interpretation guide
        print("\nInterpretation Guide:")
        print("• Positive skill scores indicate forecast is better than climatology")
        print("• Negative skill scores indicate forecast is worse than climatology")
        print("• Skill score = 0 means forecast equals climatology")
        print("• Perfect score = 1.0")

        # Additional detailed results
        print("\n" + "=" * 60)
        print("DETAILED RESULTS")
        print("=" * 60)

        print(f"\nForecast Fair Brier Score : {brier_forecast['fair_brier_score']:.4f}")
        print(
            f"Climatology Fair Brier Score : {brier_climatology['fair_brier_score']:.4f}"
        )
        print(f"Fair Brier Skill Score: {fair_bss_overall:.4f}")

        print(f"\nForecast Fair RPS : {rps_forecast['fair_rps']:.4f}")
        print(f"Climatology Fair RPS : {rps_climatology['fair_rps']:.4f}")
        print(f"Fair RPS Skill Score: {fair_rpss_overall:.4f}")

        print("\nBin-wise Fair Brier Score Comparisons:")
        for bin_name in target_bins:
            if (
                bin_name in brier_forecast["bin_fair_brier_scores"]
                and bin_name in brier_climatology["bin_fair_brier_scores"]
            ):
                forecast_val = brier_forecast["bin_fair_brier_scores"][bin_name]
                clim_val = brier_climatology["bin_fair_brier_scores"][bin_name]
                skill_val = bin_fair_bss[bin_name]
                print(f"  {bin_name}:")
                print(f"    Forecast: {forecast_val:.4f}")
                print(f"    Climatology: {clim_val:.4f}")
                print(f"    Skill Score: {skill_val:.4f}")
            else:
                print(f"  {bin_name}: Missing data")

        return skill_scores
