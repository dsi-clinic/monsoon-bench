"""Climatology onset metrics computation.

This module provides the ClimatologyOnsetMetrics class for computing
climatological baseline metrics for monsoon onset prediction.
"""

import glob
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from monsoonbench.spatial.regions import points_inside_polygon

from .base import OnsetMetricsBase
from .probabilistic import ProbabilisticOnsetMetrics


class ClimatologyOnsetMetrics(OnsetMetricsBase):
    """Class to compute climatology onset metrics."""

    @staticmethod
    def compute_climatological_onset(imd_folder, thres_file, mok=True):
        """Compute climatological onset dates from all available IMD files.

        Parameters:
        imd_folder: str, folder containing IMD NetCDF files
        thres_file: str, path to threshold file
        mok: bool, if True use June 2nd as start date (MOK), if False use May 1st

        Returns:
        climatological_onset_doy: xarray DataArray with climatological onset day of year
        """
        # Load threshold data
        thresh_ds = xr.open_dataset(thres_file)
        thres_da = thresh_ds["MWmean"]

        # Find all IMD files and extract years
        imd_folder_path = Path(imd_folder)
        imd_files = list(imd_folder_path.glob("*.nc"))
        years = []

        for file_path in imd_files:
            filename = file_path.name
            # Remove .nc extension
            name_without_ext = filename.replace(".nc", "")

            # Try to extract year from different naming patterns
            if name_without_ext.startswith("data_"):
                # Pattern: data_YYYY.nc
                year_str = name_without_ext.replace("data_", "")
            else:
                # Pattern: YYYY.nc
                year_str = name_without_ext

            # Validate that it's a 4-digit year
            try:
                year = int(year_str)
                if 1900 <= year <= 2100:  # Reasonable year range
                    years.append(year)
                else:
                    print(
                        f"Warning: Skipping file {filename} - year {year} outside valid range"
                    )
            except ValueError:
                print(
                    f"Warning: Skipping file {filename} - cannot extract valid year from '{year_str}'"
                )

        years = sorted(years)

        if not years:
            raise ValueError(f"No valid IMD files found in {imd_folder}")

        print(
            f"Computing climatological onset from {len(years)} years: {min(years)}-{max(years)}"
        )

        all_onset_days = []

        for year in years:
            try:
                # Load rainfall data using the existing function that handles both patterns
                rainfall_ds = ClimatologyOnsetMetrics.load_imd_rainfall(
                    year, imd_folder
                )

                # Detect onset for this year
                onset_da = OnsetMetricsBase.detect_observed_onset(
                    rainfall_ds, thres_da, year, mok=mok
                )

                # Convert onset dates to day of year
                onset_doy = onset_da.dt.dayofyear.astype(float)
                onset_doy = onset_doy.where(~onset_da.isnull())  # noqa: PD003

                all_onset_days.append(onset_doy)

            except Exception as e:
                print(f"Warning: Could not process year {year}: {e}")
                continue

        if not all_onset_days:
            raise ValueError("No valid years found for climatology computation")

        # Stack all years and compute mean day of year
        onset_stack = xr.concat(all_onset_days, dim="year")
        climatological_onset_doy = onset_stack.mean(dim="year")

        # Round to nearest integer day
        climatological_onset_doy = np.round(climatological_onset_doy)

        print(f"Climatological onset computed from {len(all_onset_days)} valid years")

        return climatological_onset_doy

    ## This function computes onset dates for all available years in IMD folder and creates a climatological onset dataset
    @staticmethod
    def compute_climatological_onset_dataset(
        imd_folder, thresh_slice, years=None, mok=True
    ):
        """Compute onset dates for all available years in IMD folder and create a climatological dataset.

        Parameters:
        -----------
        imd_folder : str
            Folder containing IMD NetCDF files
        thresh_slice : xarray.DataArray
            Rainfall threshold for each grid point
        years : list, optional
            Specific years to process. If None, will auto-detect available years
        mok : bool, default=True
            Whether to use MOK date filter (June 2nd)

        Returns:
        --------
        xarray.DataArray
            3D array with dimensions [year, lat, lon] containing onset dates
        """
        # Auto-detect available years if not specified
        if years is None:
            years = []
            # Look for files matching common IMD naming patterns
            file_patterns = ["data_*.nc", "*.nc"]

            for pattern in file_patterns:
                files = glob.glob(os.path.join(imd_folder, pattern))
                for file in files:
                    filename = os.path.basename(file)
                    # Extract year from filename
                    if filename.startswith("data_"):
                        year_str = filename.replace("data_", "").replace(".nc", "")
                    else:
                        year_str = filename.replace(".nc", "")
                    year = int(year_str)
                    years.append(year)

            years = sorted(list(set(years)))  # Remove duplicates and sort

        if not years:
            raise ValueError(f"No valid years found in {imd_folder}")

        print(f"Processing {len(years)} years: {years}")

        # Initialize lists to store results
        onset_arrays = []
        valid_years = []

        # Process each year
        for year in years:
            print(f"\nProcessing year {year}...")

            try:
                # Load rainfall data for this year
                rainfall_ds = OnsetMetricsBase.load_imd_rainfall(year, imd_folder)

                # Select the same spatial domain as thresh_slice
                rainfall_slice = rainfall_ds
                # Detect onset for this year
                onset_da = OnsetMetricsBase.detect_observed_onset(
                    rainfall_slice, thresh_slice, year, mok=mok
                )

                # Count valid onsets
                valid_onsets = (~pd.isna(onset_da.values)).sum()
                total_points = onset_da.size

                print(
                    f"Year {year}: Found onset in {valid_onsets}/{total_points} grid points ({valid_onsets / total_points:.1%})"
                )

                # Store the onset array
                onset_arrays.append(onset_da.values)
                valid_years.append(year)

            except Exception as e:
                print(f"Error processing year {year}: {e}")
                continue

        if not onset_arrays:
            raise ValueError("No years were successfully processed")

        # Stack all onset arrays into a 3D array
        onset_3d = np.stack(onset_arrays, axis=0)

        # Create the final DataArray
        climatological_onset_da = xr.DataArray(
            onset_3d,
            coords=[
                ("year", valid_years),
                ("lat", thresh_slice.lat.values),
                ("lon", thresh_slice.lon.values),
            ],
            name="climatological_onset_dates",
            attrs={
                "description": "Onset dates for climatological ensemble",
                "method": "MOK (June 2nd filter)" if mok else "no date filter",
                "years_processed": valid_years,
                "total_years": len(valid_years),
            },
        )

        # Print summary statistics
        total_possible = len(valid_years) * thresh_slice.size
        total_valid = (~pd.isna(climatological_onset_da.values)).sum()

        print(f"\n{'=' * 60}")
        print("CLIMATOLOGICAL ONSET DATASET SUMMARY")
        print(f"{'=' * 60}")
        print(
            f"Years processed: {len(valid_years)} ({min(valid_years)}-{max(valid_years)})"
        )
        print(
            f"Spatial domain: {len(thresh_slice.lat)} lats x {len(thresh_slice.lon)} lons"
        )
        print(
            f"Total valid onsets: {total_valid:,}/{total_possible:,} ({total_valid / total_possible:.1%})"
        )
        print(f"Method: {'MOK (June 2nd filter)' if mok else 'No date filter'}")

        # Show onset statistics by year
        print("\nOnset statistics by year:")
        for i, year in enumerate(valid_years):
            year_onsets = (~pd.isna(climatological_onset_da.isel(year=i).values)).sum()
            print(
                f"  {year}: {year_onsets}/{thresh_slice.size} ({year_onsets / thresh_slice.size:.1%})"
            )

        return climatological_onset_da

    @staticmethod
    def get_initialization_dates(year):
        """Get initialization dates (Mondays and Thursdays from May-July).

        Uses the same logic as get_s2s_deterministic_twice_weekly but only returns dates.
        """
        # Define date range from May 1 to July 31 of 2024 (template)
        start_date = datetime(2024, 5, 1)
        end_date = datetime(2024, 7, 31)
        date_range = pd.date_range(start_date, end_date, freq="D")

        # Find Mondays (weekday=0) and Thursdays (weekday=3)
        is_monday = date_range.weekday == 0
        is_thursday = date_range.weekday == 3
        filtered_dates = date_range[is_monday | is_thursday]

        # Convert to the requested year
        filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f"{year}-%m-%d"))

        return filtered_dates_yr

    @staticmethod
    def compute_climatology_as_forecast(
        climatological_onset_doy,
        year,
        init_dates,
        observed_onset_da,
        max_forecast_day=30,
        mok=True,
        mok_month=6,
        mok_day=2,
    ):
        """Use climatology as a forecast model for the given initialization dates.

        Only processes forecasts initialized before the observed onset date.

        Parameters:
        climatological_onset_doy: xarray DataArray with climatological onset day of year
        year: int, year to evaluate
        init_dates: pandas DatetimeIndex with initialization dates
        observed_onset_da: xarray DataArray with observed onset dates for filtering
        max_forecast_day: int, maximum forecast day to consider
        mok: bool, if True only count onset after June 2nd (MOK date)

        Returns:
        pandas DataFrame with climatology forecast results
        """
        results_list = []

        # Get dimensions
        lats = climatological_onset_doy.lat.values
        lons = climatological_onset_doy.lon.values

        print(
            f"Processing climatology as forecast for {len(init_dates)} init times x {len(lats)} lats x {len(lons)} lons..."
        )
        print(f"Year: {year}")
        print("Only processing forecasts initialized before observed onset dates")

        # Track statistics
        total_potential_inits = 0
        valid_inits = 0
        skipped_no_obs = 0
        skipped_late_init = 0
        onsets_forecasted = 0

        # Loop over all initialization dates and grid points
        for t_idx, init_time in enumerate(init_dates):
            if t_idx % 5 == 0:  # Print progress every 5 init times
                print(
                    f"Processing init time {t_idx + 1}/{len(init_dates)}: {init_time.strftime('%Y-%m-%d')}"
                )

            init_date = pd.to_datetime(init_time)
            mok_date = datetime(year, mok_month, mok_day)  # June 2nd of the same year

            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    total_potential_inits += 1

                    # Get observed onset date for this grid point
                    try:
                        obs_onset = observed_onset_da.isel(lat=i, lon=j).values
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

                    # Get climatological onset day of year for this grid point
                    clim_onset_doy = climatological_onset_doy.isel(lat=i, lon=j).values

                    # Skip if no climatological onset available
                    if np.isnan(clim_onset_doy):
                        continue

                    # Convert climatological day of year to actual date for this year
                    try:
                        clim_onset_date = datetime(year, 1, 1) + timedelta(
                            days=int(clim_onset_doy) - 1
                        )
                        clim_onset_date = pd.to_datetime(clim_onset_date)
                    except (ValueError, OverflowError):
                        continue  # Skip if invalid day of year

                    # Check if climatological onset is within forecast window
                    forecast_window_start = init_date + pd.Timedelta(days=1)
                    forecast_window_end = init_date + pd.Timedelta(
                        days=max_forecast_day
                    )

                    onset_day = None
                    onset_date = None

                    if forecast_window_start <= clim_onset_date <= forecast_window_end:
                        # Climatological onset is within forecast window
                        onset_day = (clim_onset_date - init_date).days

                        # Apply MOK filtering if requested
                        if mok:
                            if clim_onset_date.date() > mok_date.date():
                                # Valid onset after MOK date
                                onset_date = clim_onset_date
                                onsets_forecasted += 1
                            else:
                                # Reset if before MOK date
                                onset_day = None
                                onset_date = None
                        else:
                            # No MOK filtering
                            onset_date = clim_onset_date
                            onsets_forecasted += 1

                    # Store result
                    result = {
                        "init_time": init_time,
                        "lat": lat,
                        "lon": lon,
                        "onset_day": onset_day,  # None if no onset forecasted
                        "onset_date": onset_date.strftime("%Y-%m-%d")
                        if onset_date is not None
                        else None,
                        "climatological_onset_doy": clim_onset_doy,
                        "climatological_onset_date": clim_onset_date.strftime(
                            "%Y-%m-%d"
                        ),
                        "obs_onset_date": obs_onset_dt.strftime(
                            "%Y-%m-%d"
                        ),  # Store observed onset for reference
                    }
                    results_list.append(result)

        # Convert to DataFrame
        climatology_forecast_df = pd.DataFrame(results_list)

        print("\nClimatology Forecast Summary:")
        print(f"Total potential initializations: {total_potential_inits}")
        print(f"Skipped (no observed onset): {skipped_no_obs}")
        print(f"Skipped (initialized after observed onset): {skipped_late_init}")
        print(f"Valid initializations processed: {valid_inits}")
        print(f"Onsets forecasted: {onsets_forecasted}")
        print(
            f"Forecast rate: {onsets_forecasted / valid_inits:.3f}"
            if valid_inits > 0
            else "Forecast rate: 0.000"
        )

        if mok:
            print(
                f"Note: Only onsets on or after {mok_month}/{mok_day} were counted due to MOK flag"
            )

        return climatology_forecast_df

    @staticmethod
    def compute_climatology_metrics_with_windows(
        climatology_forecast_df,
        observed_onset_da,
        tolerance_days=3,
        verification_window=1,
        forecast_days=15,
    ):
        """Compute contingency matrix metrics using climatology forecasts against observed onset.

        Parameters:
        climatology_forecast_df: pandas DataFrame from compute_climatology_as_forecast
        observed_onset_da: xarray DataArray with observed onset dates for the evaluation year
        tolerance_days: int, tolerance in days for considering a prediction as correct
        verification_window: int, days after init to start validation window
        forecast_days: int, length of forecast window in days

        Returns:
        metrics_df: pandas DataFrame with metrics for each grid point
        summary_stats: dict with overall statistics
        """
        print(
            f"Computing climatology forecast metrics with tolerance = {tolerance_days} days"
        )
        print(
            f"Verification window starts {verification_window} days after initialization"
        )
        print(f"Forecast window length: {forecast_days} days")

        # Initialize results list
        results_list = []

        # Get unique grid points
        unique_locations = climatology_forecast_df[["lat", "lon"]].drop_duplicates()

        print(f"Processing {len(unique_locations)} unique grid points...")

        for idx, (_, row) in enumerate(unique_locations.iterrows()):
            lat, lon = row["lat"], row["lon"]

            if idx % 10 == 0:  # Progress update
                print(
                    f"Processing grid point {idx + 1}/{len(unique_locations)}: lat={lat:.2f}, lon={lon:.2f}"
                )

            # Get all climatology forecasts for this grid point
            grid_data = climatology_forecast_df[
                (climatology_forecast_df["lat"] == lat)
                & (climatology_forecast_df["lon"] == lon)
            ].copy()

            # Get observed onset date for this grid point
            lat_idx = np.argmin(np.abs(observed_onset_da.lat.values - lat))
            lon_idx = np.argmin(np.abs(observed_onset_da.lon.values - lon))
            obs_onset = observed_onset_da.isel(lat=lat_idx, lon=lon_idx).values

            # Skip if no observed onset
            if pd.isna(obs_onset):
                continue

            obs_onset_dt = pd.to_datetime(obs_onset)

            # Convert date strings to datetime for calculation
            grid_data["clim_forecast_dt"] = pd.to_datetime(grid_data["onset_date"])
            grid_data["init_dt"] = pd.to_datetime(grid_data["init_time"])

            # Initialize counters
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            num_onset = 0
            num_no_onset = 0
            mae_tp = []
            mae_fp = []

            # Process each initialization
            for _, init_row in grid_data.iterrows():
                t_init = init_row["init_dt"]
                clim_forecast = init_row["clim_forecast_dt"]

                # Define forecast windows
                valid_window_start = t_init + pd.Timedelta(days=verification_window)
                valid_window_end = valid_window_start + pd.Timedelta(
                    days=14
                )  # Always 15 days long

                whole_forecast_window_start = t_init + pd.Timedelta(days=1)
                whole_forecast_window_end = t_init + pd.Timedelta(days=forecast_days)

                # Check if true onset is within whole forecast window
                is_onset_in_whole_window = (
                    whole_forecast_window_start
                    <= obs_onset_dt
                    <= whole_forecast_window_end
                )
                if is_onset_in_whole_window:
                    num_onset += 1
                else:
                    num_no_onset += 1

                # Check if climatology forecasted onset
                has_clim_forecast = not pd.isna(clim_forecast)

                if has_clim_forecast:
                    # Climatology forecasted onset - check if it's within validation window
                    is_clim_in_valid_window = (
                        valid_window_start <= clim_forecast <= valid_window_end
                    )

                    if is_clim_in_valid_window:
                        # Climatology forecast was within validation window
                        abs_diff_days = abs((clim_forecast - obs_onset_dt).days)

                        if abs_diff_days <= tolerance_days:
                            TP += 1
                            mae_tp.append(abs_diff_days)
                        else:
                            FP += 1
                            mae_fp.append(abs_diff_days)

                else:
                    # Climatology had no forecast
                    if is_onset_in_whole_window:
                        # True onset was within whole forecast window but climatology missed it
                        FN += 1
                    else:
                        # True onset was outside whole forecast window and climatology correctly had no forecast
                        TN += 1

            # Calculate metrics
            total_forecasts = len(grid_data)

            # Mean Absolute Error (combining TP and FP)
            mae_combined = mae_tp + mae_fp
            mae = np.mean(mae_combined) if len(mae_combined) > 0 else np.nan
            mae_tp_only = np.mean(mae_tp) if len(mae_tp) > 0 else np.nan

            # Store results
            result = {
                "lat": lat,
                "lon": lon,
                "total_forecasts": total_forecasts,
                "true_positive": TP,
                "true_negative": TN,
                "false_positive": FP,
                "false_negative": FN,
                "num_onset": num_onset,
                "num_no_onset": num_no_onset,
                "mae_combined": mae,
                "mae_tp_only": mae_tp_only,
                "num_tp_errors": len(mae_tp),
                "num_fp_errors": len(mae_fp),
                "tolerance_days": tolerance_days,
                "verification_window": verification_window,
                "forecast_days": forecast_days,
            }
            results_list.append(result)

        # Convert to DataFrame
        metrics_df = pd.DataFrame(results_list)

        # Calculate summary statistics
        summary_stats = {
            "total_grid_points": len(metrics_df),
            "total_forecasts": metrics_df["total_forecasts"].sum(),
            "overall_true_positive": metrics_df["true_positive"].sum(),
            "overall_true_negative": metrics_df["true_negative"].sum(),
            "overall_false_positive": metrics_df["false_positive"].sum(),
            "overall_false_negative": metrics_df["false_negative"].sum(),
            "overall_num_onset": metrics_df["num_onset"].sum(),
            "overall_num_no_onset": metrics_df["num_no_onset"].sum(),
            "overall_mae_combined": metrics_df["mae_combined"].mean(),
            "overall_mae_tp_only": metrics_df["mae_tp_only"].mean(),
            "tolerance_days": tolerance_days,
            "verification_window": verification_window,
            "forecast_days": forecast_days,
        }

        return metrics_df, summary_stats

    @staticmethod
    def compute_climatology_baseline_multiple_years(
        years,
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
        """Compute climatology baseline metrics for multiple years.

        Returns:
        metrics_df_dict: dict, {year: metrics_df}
        climatological_onset_doy: xarray DataArray with climatological onset day of year
        """
        print("Computing climatological onset reference...")

        # Compute climatological onset once (using all available years)
        climatological_onset_doy = ClimatologyOnsetMetrics.compute_climatological_onset(
            imd_folder, thres_file, mok=mok
        )

        # Load threshold data
        thresh_ds = xr.open_dataset(thres_file)
        thres_da = thresh_ds["MWmean"]

        metrics_df_dict = {}

        for year in years:
            print(f"\n{'=' * 50}")
            print(f"Evaluating climatology baseline for year {year}")
            print(f"{'=' * 50}")

            # Get initialization dates for this year (same as model would use)
            init_dates = ClimatologyOnsetMetrics.get_initialization_dates(year)

            # Load observed data for this year
            imd = OnsetMetricsBase.load_imd_rainfall(year, imd_folder)
            observed_onset_da = OnsetMetricsBase.detect_observed_onset(
                imd, thres_da, year, mok=mok
            )

            # Generate climatology forecasts for all initialization dates
            # Now passing observed_onset_da to filter initializations
            climatology_forecast_df = (
                ClimatologyOnsetMetrics.compute_climatology_as_forecast(
                    climatological_onset_doy,
                    year,
                    init_dates,
                    observed_onset_da,
                    max_forecast_day=max_forecast_day,
                    mok=mok,
                    mok_month=mok_month,
                    mok_day=mok_day,
                )
            )

            # Compute metrics
            metrics_df, summary_stats = (
                ClimatologyOnsetMetrics.compute_climatology_metrics_with_windows(
                    climatology_forecast_df,
                    observed_onset_da,
                    tolerance_days=tolerance_days,
                    verification_window=verification_window,
                    forecast_days=forecast_days,
                )
            )

            # Store results
            metrics_df_dict[year] = metrics_df

            print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
            print(
                f"Summary stats: TP={summary_stats['overall_true_positive']}, "
                f"FP={summary_stats['overall_false_positive']}, "
                f"FN={summary_stats['overall_false_negative']}, "
                f"TN={summary_stats['overall_true_negative']}"
            )

        return metrics_df_dict, climatological_onset_doy

    ## This function creates forecast-observation pairs using climatological ensemble where each year is a member
    @staticmethod
    def create_climatological_forecast_obs_pairs(
        clim_onset, target_year, init_dates, day_bins, max_forecast_day=15, mok=True
    ):
        """Create forecast-observation pairs using climatological ensemble where each year is a member.
        Uses day-of-year instead of calendar dates for onset comparison.

        Parameters:
        -----------
        clim_onset : xarray.DataArray
            3D array with dimensions [year, lat, lon] containing onset dates for all years
        target_year : int
            The year to use as "truth" for observations
        init_dates : list or pandas.DatetimeIndex
            Initialization dates for forecasts
        day_bins : list of tuples
            List of (start_day, end_day) tuples for bins within forecast window
            e.g., [(1, 5), (6, 10), (11, 15)]
        max_forecast_day : int, default=15
            Maximum forecast day
        mok : bool, default=True
            Whether to use MOK date filter (June 2nd)

        Returns:
        --------
        DataFrame with forecast-observation pairs
        """
        results_list = []

        # Get the observed onset for the target year
        if target_year not in clim_onset.year.values:
            raise ValueError(
                f"Target year {target_year} not found in climatological dataset"
            )

        obs_onset_da = clim_onset.sel(year=target_year)

        # Use ALL years as ensemble members (including target year)
        ensemble_years = list(clim_onset.year.values)
        ensemble_onset_da = clim_onset.sel(year=ensemble_years)

        # Create extended bins including "before initialization" and "after max_forecast_day" bins
        extended_bins = (
            [(-float("inf"), 0)] + day_bins + [(max_forecast_day + 1, float("inf"))]
        )

        print(f"Creating climatological forecasts for target year {target_year}")
        print(
            f"Using {len(ensemble_years)} years as ensemble members: {ensemble_years}"
        )
        print(f"Processing {len(init_dates)} initialization dates")
        print(f"Day bins: {day_bins}")
        print(
            f"Extended bins include: 'Before initialization' and 'After day {max_forecast_day}'"
        )
        print("Using day-of-year method for onset comparison")

        # Get the actual lat/lon coordinates from the data
        lats = obs_onset_da.lat.values
        lons = obs_onset_da.lon.values

        # Create unique lat-lon pairs (no repetition)
        unique_pairs = list(zip(lons, lats))

        print(f"Processing {len(unique_pairs)} unique lat-lon pairs")

        # Process each initialization date and location
        for init_date in init_dates:
            init_date = pd.to_datetime(init_date)
            init_doy = init_date.dayofyear  # Day of year for initialization

            # Loop over unique lat-lon pairs
            for pair_idx, (lon, lat) in enumerate(unique_pairs):
                # Get observed onset for this location and target year
                try:
                    lat_idx = np.where(np.abs(obs_onset_da.lat.values - lat) < 0.01)[0][
                        0
                    ]
                    lon_idx = np.where(np.abs(obs_onset_da.lon.values - lon) < 0.01)[0][
                        0
                    ]
                    obs_onset = obs_onset_da.isel(lat=lat_idx, lon=lon_idx).values
                except:
                    continue

                # Skip if no observed onset
                if pd.isna(obs_onset):
                    continue

                obs_onset_dt = pd.to_datetime(obs_onset)
                obs_onset_doy = obs_onset_dt.dayofyear  # Day of year for observed onset

                # Only process forecasts that are initialized BEFORE the observed onset (by day of year)
                if init_doy >= obs_onset_doy:
                    continue

                # Get ensemble member onsets for this location using the same indices
                ensemble_onsets = ensemble_onset_da.isel(
                    lat=lat_idx, lon=lon_idx
                ).values

                # Convert ensemble onsets to days from initialization using day-of-year
                ensemble_forecast_days = []
                ensemble_years_with_data = []

                for ens_idx, ens_onset in enumerate(ensemble_onsets):
                    ens_year = ensemble_years[ens_idx]

                    if pd.notna(ens_onset):
                        ens_onset_dt = pd.to_datetime(ens_onset)
                        ens_onset_doy = (
                            ens_onset_dt.dayofyear
                        )  # Day of year for ensemble onset

                        # Calculate days from initialization using day-of-year difference
                        days_from_init = ens_onset_doy - init_doy
                        ensemble_forecast_days.append(days_from_init)
                        ensemble_years_with_data.append(ens_year)
                    else:
                        # No onset predicted by this member
                        ensemble_forecast_days.append(None)
                        ensemble_years_with_data.append(ens_year)

                total_members = len(ensemble_years)

                # First pass: calculate total members with onset across all bins
                total_members_with_onset = 0
                bin_members_onset = []  # Store for second pass

                for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
                    members_with_onset_in_bin = 0

                    # Handle the "before initialization" bin
                    if bin_start == -float("inf"):
                        for i, member_onset_day in enumerate(ensemble_forecast_days):
                            if member_onset_day is not None and member_onset_day <= 0:
                                members_with_onset_in_bin += 1

                    # Handle the "after max_forecast_day" bin
                    elif bin_start > max_forecast_day:
                        for i, member_onset_day in enumerate(ensemble_forecast_days):
                            if (
                                member_onset_day is not None
                                and member_onset_day > max_forecast_day
                            ):
                                members_with_onset_in_bin += 1

                    else:
                        # Regular bin within forecast window
                        for i, member_onset_day in enumerate(ensemble_forecast_days):
                            if (
                                member_onset_day is not None
                                and bin_start <= member_onset_day <= bin_end
                            ):
                                members_with_onset_in_bin += 1

                    bin_members_onset.append(members_with_onset_in_bin)
                    total_members_with_onset += members_with_onset_in_bin

                # Skip if no members showed onset
                if total_members_with_onset == 0:
                    continue

                # Second pass: For each day bin, calculate probabilities using total_members_with_onset
                for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
                    members_with_onset_in_bin = bin_members_onset[bin_idx]

                    # Track which years contribute to this bin
                    contributing_years = []

                    # Handle the "before initialization" bin
                    if bin_start == -float("inf"):
                        bin_label = "Before initialization"

                        # Check if observed onset occurs before initialization (by day of year)
                        observed_onset = int(obs_onset_doy <= init_doy)

                        # Get contributing years
                        for i, member_onset_day in enumerate(ensemble_forecast_days):
                            if member_onset_day is not None and member_onset_day <= 0:
                                contributing_years.append(ensemble_years_with_data[i])

                    # Handle the "after max_forecast_day" bin
                    elif bin_start > max_forecast_day:
                        bin_label = f"After day {max_forecast_day}"

                        # Check if observed onset occurs after max_forecast_day (by day of year)
                        obs_days_from_init = obs_onset_doy - init_doy

                        observed_onset = int(obs_days_from_init > max_forecast_day)

                        # Get contributing years
                        for i, member_onset_day in enumerate(ensemble_forecast_days):
                            if (
                                member_onset_day is not None
                                and member_onset_day > max_forecast_day
                            ):
                                contributing_years.append(ensemble_years_with_data[i])

                    else:
                        # Regular bin within forecast window
                        bin_label = f"Days {bin_start}-{bin_end}"

                        # Check if observed onset falls within this day bin (by day of year)
                        obs_days_from_init = obs_onset_doy - init_doy
                        observed_onset = int(bin_start <= obs_days_from_init <= bin_end)

                        # Get contributing years
                        for i, member_onset_day in enumerate(ensemble_forecast_days):
                            if (
                                member_onset_day is not None
                                and bin_start <= member_onset_day <= bin_end
                            ):
                                contributing_years.append(ensemble_years_with_data[i])

                    # Calculate probability using only members that showed onset
                    predicted_prob = (
                        members_with_onset_in_bin / total_members_with_onset
                    )

                    # Convert contributing years to string for storage
                    contributing_years_str = (
                        ",".join(map(str, sorted(contributing_years)))
                        if contributing_years
                        else ""
                    )

                    # Store result
                    result = {
                        "init_time": init_date.strftime("%Y-%m-%d"),
                        "lat": lat,
                        "lon": lon,
                        "bin_start": bin_start,
                        "bin_end": bin_end,
                        "bin_label": bin_label,
                        "predicted_prob": predicted_prob,
                        "observed_onset": observed_onset,
                        "members_with_onset": members_with_onset_in_bin,
                        "total_members": total_members,
                        "total_members_with_onset": total_members_with_onset,  # New field
                        "contributing_years": contributing_years_str,
                        "n_contributing_years": len(contributing_years),
                        "year": target_year,
                        "obs_onset_date": obs_onset_dt.strftime("%Y-%m-%d"),
                        "obs_onset_doy": obs_onset_doy,
                        "init_doy": init_doy,
                        "obs_days_from_init_doy": obs_days_from_init
                        if "obs_days_from_init" in locals()
                        else (obs_onset_doy - init_doy),
                        "bin_index": bin_idx,
                        "forecast_type": "climatological_doy",
                    }
                    results_list.append(result)

        # Convert to DataFrame
        forecast_obs_df = pd.DataFrame(results_list)

        if len(forecast_obs_df) == 0:
            print("Warning: No forecast-observation pairs generated")
            return forecast_obs_df

        print(
            f"Generated {len(forecast_obs_df)} climatological forecast-observation pairs"
        )
        print(f"Unique lat-lon pairs processed: {len(unique_pairs)}")
        print(f"Total bins per forecast: {len(extended_bins)}")
        print(
            f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}"
        )
        print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
        print(
            f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}"
        )

        # Verify uniqueness
        unique_locations_in_output = len(
            forecast_obs_df[["lat", "lon"]].drop_duplicates()
        )
        print(f"Unique locations in output: {unique_locations_in_output}")

        # Show distribution across bins
        print("\nDistribution across bins:")
        bin_stats = (
            forecast_obs_df.groupby("bin_label")
            .agg(
                {
                    "predicted_prob": ["count", "mean"],
                    "observed_onset": "mean",
                    "n_contributing_years": "mean",
                    "total_members_with_onset": "mean",
                }
            )
            .round(3)
        )
        print(bin_stats)

        return forecast_obs_df

    @staticmethod
    def multi_year_climatological_forecast_obs_pairs(
        clim_onset,
        target_years,
        day_bins,
        mem_num,
        model_forecast_dir,
        date_filter_year=2024,
        file_pattern="tp_4p0_{}.nc",
        max_forecast_day=15,
        mok=True,
    ):
        """Create climatological forecast-observation pairs for multiple target years.

        Parameters:
        -----------
        clim_onset : xarray.DataArray
            3D array with dimensions [year, lat, lon] containing onset dates
        target_years : list
            Years to use as truth for observations
        day_bins : list of tuples
            List of (start_day, end_day) tuples for bins
        max_forecast_day : int, default=15
            Maximum forecast day
        mok : bool, default=True
            Whether to use MOK date filter

        Returns:
        --------
        DataFrame with combined forecast-observation pairs from all target years
        """
        # Load threshold data (same for all years)

        orig_lat = clim_onset.lat.values
        orig_lon = clim_onset.lon.values

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
        clim_onset_slice = clim_onset.sel(lat=inside_lats, lon=inside_lons)

        all_forecast_obs_pairs = []

        for target_year in target_years:
            print(f"\n{'=' * 50}")
            print(f"Processing target year {target_year}")
            print(f"{'=' * 50}")

            try:
                # Get initialization dates for this year
                _, init_dates = (
                    ProbabilisticOnsetMetrics.get_forecast_probabilistic_twice_weekly_2(
                        target_year,
                        model_forecast_dir,
                        mem_num,
                        date_filter_year,
                        file_pattern,
                    )
                )

                # Create forecast-observation pairs for this year
                forecast_obs_pairs = (
                    ClimatologyOnsetMetrics.create_climatological_forecast_obs_pairs(
                        clim_onset=clim_onset_slice,
                        target_year=target_year,
                        init_dates=init_dates,
                        day_bins=day_bins,
                        max_forecast_day=max_forecast_day,
                        mok=mok,
                    )
                )

                if len(forecast_obs_pairs) > 0:
                    all_forecast_obs_pairs.append(forecast_obs_pairs)
                    print(
                        f"Target year {target_year} completed: {len(forecast_obs_pairs)} pairs"
                    )
                else:
                    print(f"No pairs generated for target year {target_year}")

            except Exception as e:
                print(f"Error processing target year {target_year}: {e}")
                continue

        # Combine all years
        if not all_forecast_obs_pairs:
            raise ValueError("No data was successfully processed for any target year")

        combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

        print(f"\n{'=' * 50}")
        print("CLIMATOLOGICAL FORECAST SUMMARY")
        print(f"{'=' * 50}")
        print(f"Target years processed: {target_years}")
        print(f"Total forecast-observation pairs: {len(combined_forecast_obs)}")
        print(
            f"Probability range: {combined_forecast_obs['predicted_prob'].min():.3f} - {combined_forecast_obs['predicted_prob'].max():.3f}"
        )
        print(
            f"Overall observed onset rate: {combined_forecast_obs['observed_onset'].mean():.3f}"
        )

        return combined_forecast_obs

    @staticmethod
    def calculate_brier_score_climatology(forecast_obs_df):
        """Calculate Brier Score and Fair Brier Score for probabilistic forecasts.

        Brier Score = (1/n*m) * Σ(Y_ij - p_ij)²
        Fair Brier Score = (1/n*m) * Σ[(Y_ij - p_ij)² - p_ij(1-p_ij)/(ens-1)]

        where:
        - n = number of forecasts
        - m = number of bins per forecast
        - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
        - p_ij = predicted probability for bin j in forecast i
        - ens = number of ensemble members

        Note: Excludes "Before initialization" bin from calculations

        Parameters:
        -----------
        forecast_obs_df : DataFrame
            Output from create_forecast_observation_pairs_with_bins()
            Must contain columns: 'predicted_prob', 'observed_onset', 'total_members', 'bin_label'

        Returns:
        --------
        dict with Brier score metrics
        """
        # Filter out "Before initialization" bin
        filtered_df = forecast_obs_df[
            forecast_obs_df["bin_label"] != "Before initialization"
        ].copy()

        if len(filtered_df) == 0:
            print(
                "Warning: No data remaining after filtering out 'Before initialization' bin"
            )
            return {
                "brier_score": np.nan,
                "fair_brier_score": np.nan,
                "bin_brier_scores": {},
                "bin_fair_brier_scores": {},
                "n_samples": 0,
                "filtered_bins": [],
            }

        print("Calculating Brier Score excluding 'Before initialization' bin")
        print(
            f"Original samples: {len(forecast_obs_df)}, After filtering: {len(filtered_df)}"
        )

        # Calculate squared differences
        squared_diffs = (
            filtered_df["observed_onset"] - filtered_df["predicted_prob"]
        ) ** 2

        # Calculate overall Brier Score
        brier_score = squared_diffs.mean()

        # Calculate Fair Brier Score correction term
        # ens-1 where ens is the number of ensemble members
        correction_term = (
            filtered_df["predicted_prob"] * (1 - filtered_df["predicted_prob"])
        ) / (filtered_df["total_members_with_onset"] - 1)

        # Fair Brier Score
        fair_brier_components = squared_diffs - correction_term
        fair_brier_score = fair_brier_components.mean()

        # Calculate squared differences for bin-wise analysis
        filtered_df["squared_diff"] = squared_diffs
        filtered_df["fair_brier_component"] = fair_brier_components

        # Bin-wise Brier scores (excluding "Before initialization")
        bin_brier_scores = filtered_df.groupby("bin_label")["squared_diff"].mean()
        bin_fair_brier_scores = filtered_df.groupby("bin_label")[
            "fair_brier_component"
        ].mean()

        brier_results = {
            "brier_score": brier_score,
            "fair_brier_score": fair_brier_score,
            "bin_brier_scores": bin_brier_scores.to_dict(),
            "bin_fair_brier_scores": bin_fair_brier_scores.to_dict(),
            "n_samples": len(filtered_df),
            "filtered_bins": sorted(filtered_df["bin_label"].unique()),
            "excluded_bins": ["Before initialization"],
        }

        print(
            f"Brier Score (excluding 'Before initialization'): {brier_results['brier_score']:.4f}"
        )
        print(
            f"Fair Brier Score (excluding 'Before initialization'): {brier_results['fair_brier_score']:.4f}"
        )
        print(f"Bins included in calculation: {brier_results['filtered_bins']}")

        return brier_results

    @staticmethod
    def calculate_auc_climatology(forecast_obs_df):
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
        forecast_obs_df = forecast_obs_df[
            forecast_obs_df["bin_label"] != "Before initialization"
        ].copy()
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

    @staticmethod
    def calculate_rps_climatology(forecast_obs_df):
        """Calculate Ranked Probability Score (RPS) and Fair RPS for forecasts.

        RPS = (1/(M-1)) * Σ [ (Σ p_k) - (Σ o_k) ]²
        where:
        - M is the number of bins
        - p_k is the predicted probability for bin k
        - o_k is the binary observation (1 or 0) for bin k

        Parameters:
        -----------
        forecast_obs_df : DataFrame
            Must contain: 'init_time', 'lat', 'lon', 'bin_label', 
            'predicted_prob', 'observed_onset', 'total_members_with_onset'
        
        Returns:
        --------
        dict with RPS metrics
        """
        # 1. Filter out unwanted bins
        filtered_df = forecast_obs_df[
            forecast_obs_df["bin_label"] != "Before initialization"
        ].copy()

        if len(filtered_df) == 0:
            return {"rps": np.nan, "fair_rps": np.nan, "n_samples": 0}

        # 2. Extract and sort bins chronologically
        def extract_day_range(bin_label):
            if "Days " in bin_label:
                try:
                    return int(bin_label.replace("Days ", "").split("-")[0])
                except: return 999
            return 999

        sorted_bins = sorted(filtered_df["bin_label"].unique(), key=extract_day_range)
        num_bins = len(sorted_bins)

        # 3. Pivot to align probabilities and observations by bin order
        # Each row is a unique forecast (initialization + location)
        idx_cols = ['init_time', 'lat', 'lon']
        pivot_prob = filtered_df.pivot_table(index=idx_cols, columns='bin_label', values='predicted_prob')[sorted_bins].fillna(0)
        pivot_obs = filtered_df.pivot_table(index=idx_cols, columns='bin_label', values='observed_onset')[sorted_bins].fillna(0)
        
        # Get ensemble sizes for the 'Fair' correction
        ens_sizes = filtered_df.groupby(idx_cols)['total_members_with_onset'].first()

        # 4. Calculate Cumulative Distributions (CDF)
        cum_p = pivot_prob.cumsum(axis=1)
        cum_o = pivot_obs.cumsum(axis=1)

        # 5. Calculate RPS
        # Squared difference of cumulative distributions, summed across bins, normalized by (M-1)
        squared_diff_cdf = (cum_p - cum_o)**2
        # Standard RPS convention usually divides by (num_bins - 1) 
        # to bound the score between 0 and 1
        rps_per_forecast = squared_diff_cdf.sum(axis=1) / (num_bins - 1)

        # 6. Calculate Fair RPS Correction
        # Fair RPS = RPS - Σ [cum_p * (1 - cum_p)] / (ens - 1)
        # This accounts for bias in small ensemble sizes
        correction = (cum_p * (1 - cum_p)).sum(axis=1) / ((ens_sizes - 1) * (num_bins - 1))
        fair_rps_per_forecast = rps_per_forecast - correction

        results = {
            "rps": rps_per_forecast.mean(),
            "fair_rps": fair_rps_per_forecast.mean(),
            "n_samples": len(pivot_prob),
            "bins_evaluated": sorted_bins,
            "rps_per_forecast": rps_per_forecast.to_dict(), # Useful for spatial plotting
        }

        print(f"Calculated RPS across {results['n_samples']} samples.")
        print(f"Overall RPS: {results['rps']:.4f}")
        print(f"Overall Fair RPS: {results['fair_rps']:.4f}")

        return results