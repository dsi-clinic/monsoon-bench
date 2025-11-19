"""Climatology onset metrics computation.

This module provides the ClimatologyOnsetMetrics class for computing
climatological baseline metrics for monsoon onset prediction.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .base import OnsetMetricsBase


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
                    print(f"Warning: Skipping file {filename} - year {year} outside valid range")
            except ValueError:
                print(f"Warning: Skipping file {filename} - cannot extract valid year from '{year_str}'")
        
        years = sorted(years)
        
        if not years:
            raise ValueError(f"No valid IMD files found in {imd_folder}")
        
        print(f"Computing climatological onset from {len(years)} years: {min(years)}-{max(years)}")
        
        all_onset_days = []
        
        for year in years:       
            try:
                # Load rainfall data using the existing function that handles both patterns
                rainfall_ds = ClimatologyOnsetMetrics.load_imd_rainfall(year, imd_folder)
                
                # Detect onset for this year
                onset_da = OnsetMetricsBase.detect_observed_onset(rainfall_ds, thres_da, year, mok=mok)
                
                # Convert onset dates to day of year
                onset_doy = onset_da.dt.dayofyear.astype(float)
                onset_doy = onset_doy.where(~onset_da.isna())
                
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
    def compute_climatology_as_forecast(climatological_onset_doy, year, init_dates, observed_onset_da,
                                    max_forecast_day=30, mok=True, mok_month=6, mok_day=2):
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
        
        print(f"Processing climatology as forecast for {len(init_dates)} init times x {len(lats)} lats x {len(lons)} lons...")
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
                print(f"Processing init time {t_idx+1}/{len(init_dates)}: {init_time.strftime('%Y-%m-%d')}")
            
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
                        clim_onset_date = datetime(year, 1, 1) + timedelta(days=int(clim_onset_doy) - 1)
                        clim_onset_date = pd.to_datetime(clim_onset_date)
                    except (ValueError, OverflowError):
                        continue  # Skip if invalid day of year
                    
                    # Check if climatological onset is within forecast window
                    forecast_window_start = init_date + pd.Timedelta(days=1)
                    forecast_window_end = init_date + pd.Timedelta(days=max_forecast_day)
                    
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
                        "onset_date": onset_date.strftime("%Y-%m-%d") if onset_date is not None else None,
                        "climatological_onset_doy": clim_onset_doy,
                        "climatological_onset_date": clim_onset_date.strftime("%Y-%m-%d"),
                        "obs_onset_date": obs_onset_dt.strftime("%Y-%m-%d")  # Store observed onset for reference
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
        print(f"Forecast rate: {onsets_forecasted/valid_inits:.3f}" if valid_inits > 0 else "Forecast rate: 0.000")
        
        if mok:
            print(f"Note: Only onsets on or after {mok_month}/{mok_day} were counted due to MOK flag")
        
        return climatology_forecast_df

    @staticmethod
    def compute_climatology_metrics_with_windows(climatology_forecast_df, observed_onset_da, 
                                            tolerance_days=3, verification_window=1, forecast_days=15):
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
        print(f"Computing climatology forecast metrics with tolerance = {tolerance_days} days")
        print(f"Verification window starts {verification_window} days after initialization")
        print(f"Forecast window length: {forecast_days} days")
        
        # Initialize results list
        results_list = []
        
        # Get unique grid points
        unique_locations = climatology_forecast_df[["lat", "lon"]].drop_duplicates()
        
        print(f"Processing {len(unique_locations)} unique grid points...")
        
        for idx, (_, row) in enumerate(unique_locations.iterrows()):
            lat, lon = row["lat"], row["lon"]
            
            if idx % 10 == 0:  # Progress update
                print(f"Processing grid point {idx+1}/{len(unique_locations)}: lat={lat:.2f}, lon={lon:.2f}")
            
            # Get all climatology forecasts for this grid point
            grid_data = climatology_forecast_df[(climatology_forecast_df["lat"] == lat) & 
                                            (climatology_forecast_df["lon"] == lon)].copy()
            
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
                valid_window_end = valid_window_start + pd.Timedelta(days=14)  # Always 15 days long
                
                whole_forecast_window_start = t_init + pd.Timedelta(days=1)
                whole_forecast_window_end = t_init + pd.Timedelta(days=forecast_days)
                
                # Check if true onset is within whole forecast window
                is_onset_in_whole_window = whole_forecast_window_start <= obs_onset_dt <= whole_forecast_window_end
                if is_onset_in_whole_window:
                    num_onset += 1
                else:
                    num_no_onset += 1
                
                # Check if climatology forecasted onset
                has_clim_forecast = not pd.isna(clim_forecast)
                
                if has_clim_forecast:
                    # Climatology forecasted onset - check if it's within validation window
                    is_clim_in_valid_window = valid_window_start <= clim_forecast <= valid_window_end
                    
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
                "forecast_days": forecast_days
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
            "forecast_days": forecast_days
        }
        
        return metrics_df, summary_stats  

    @staticmethod
    def compute_climatology_baseline_multiple_years(years, imd_folder, thres_file,
                                                tolerance_days=3, verification_window=1, forecast_days=15,
                                                max_forecast_day=15, mok=True, onset_window=5, mok_month=6, mok_day=2):
        """Compute climatology baseline metrics for multiple years.
        
        Returns:
        metrics_df_dict: dict, {year: metrics_df}
        climatological_onset_doy: xarray DataArray with climatological onset day of year
        """
        print("Computing climatological onset reference...")
        
        # Compute climatological onset once (using all available years)
        climatological_onset_doy = ClimatologyOnsetMetrics.compute_climatological_onset(imd_folder, thres_file, mok=mok)

        # Load threshold data
        thresh_ds = xr.open_dataset(thres_file)
        thres_da = thresh_ds["MWmean"]
        
        metrics_df_dict = {}
        
        for year in years:
            print(f"\n{'='*50}")
            print(f"Evaluating climatology baseline for year {year}")
            print(f"{'='*50}")
            
            # Get initialization dates for this year (same as model would use)
            init_dates =  ClimatologyOnsetMetrics.get_initialization_dates(year)
            
            # Load observed data for this year
            imd = OnsetMetricsBase.load_imd_rainfall(year, imd_folder)
            observed_onset_da = OnsetMetricsBase.detect_observed_onset(imd, thres_da, year, mok=mok)
            
            # Generate climatology forecasts for all initialization dates
            # Now passing observed_onset_da to filter initializations
            climatology_forecast_df = ClimatologyOnsetMetrics.compute_climatology_as_forecast(
                climatological_onset_doy, year, init_dates, observed_onset_da,
                max_forecast_day=max_forecast_day, mok=mok, mok_month=mok_month, mok_day=mok_day
            )
            
            # Compute metrics
            metrics_df, summary_stats = ClimatologyOnsetMetrics.compute_climatology_metrics_with_windows(
                climatology_forecast_df, observed_onset_da,
                tolerance_days=tolerance_days,
                verification_window=verification_window,
                forecast_days=forecast_days
            )
            
            # Store results
            metrics_df_dict[year] = metrics_df
            
            print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
            print(f"Summary stats: TP={summary_stats['overall_true_positive']}, "
                f"FP={summary_stats['overall_false_positive']}, "
                f"FN={summary_stats['overall_false_negative']}, "
                f"TN={summary_stats['overall_true_negative']}")
        
        return metrics_df_dict, climatological_onset_doy