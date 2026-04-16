import xarray as xr
import numpy as np
from monsoonbench.metrics import (
    ProbabilisticOnsetMetrics,
    OnsetMetricsBase,
    ClimatologyOnsetMetrics
)
import pandas as pd
from pathlib import Path
from monsoonbench.spatial.regions import points_inside_polygon


def fig6_multi_year_forecast_obs_pairs(
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
            # polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
            # polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
            polygon1_lon = np.array([67, 101, 101, 67, 67])
            polygon1_lat = np.array([7, 7, 37, 37, 7])
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
        inside_lats = np.unique(inside_lats)
        inside_lons = np.unique(inside_lons)
        thresh_slice = thresh_da.sel(lat=inside_lats, lon=inside_lons)

        # Initialize list to store all forecast-observation pairs
        all_forecast_obs_pairs = []

        print("Lat Check")
        print(inside_lats)

        # Process each year
        for year in years:
            print(f"\n{'=' * 50}")
            print(f"Processing year {year}")
            print(f"{'=' * 50}")

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
                print(
                     f"ERROR CHECK {p_model_slice.lon.values}:"
                )

                print("Loading IMD rainfall data...")
                rainfall_ds = OnsetMetricsBase.load_imd_rainfall(year, imd_folder)
                rainfall_ds_slice = rainfall_ds#.sel(lat=lats, lon=lons)
                print("Detecting observed onset...")
                onset_da = OnsetMetricsBase.detect_observed_onset(
                    rainfall_ds_slice, thresh_slice, year, mok
                )
                print(
                    f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points"
                )

                print("Computing onset for all ensemble members...")
                print("p_model_slice coords:", p_model_slice.lat.values, p_model_slice.lon.values)
                print("onset_da coords:", onset_da.lat.values, onset_da.lon.values)
                print("onset_da non-NaN count:", (~pd.isna(onset_da.values)).sum())
                onset_all_members = (
                    ProbabilisticOnsetMetrics.fig6_compute_onset_for_all_members(
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

            except Exception:
                print(f"Error processing year {year}:")
                import traceback

                traceback.print_exc()  # This reveals the REAL line number and error
                continue

        # Combine all years
        print(f"\n{'=' * 50}")
        print("Combining all years")
        print(f"{'=' * 50}")

        if not all_forecast_obs_pairs:
            raise ValueError("No data was successfully processed for any year")

        combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

        # Print final summary statistics
        print("\nFinal Summary Statistics:")
        print(f"Years processed: {years}")
        return combined_forecast_obs

def fig6_multi_year_climatological_forecast_obs_pairs(
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
        orig_lat  = clim_onset.lat.values
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
            # polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
            # polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
            polygon1_lon = np.array([67, 101, 101, 67, 67])
            polygon1_lat = np.array([7, 7, 37, 37, 7])
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
            print(f"\n{'='*50}")
            print(f"Processing target year {target_year}")
            print(f"{'='*50}")

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

        print(f"\n{'='*50}")
        print("CLIMATOLOGICAL FORECAST SUMMARY")
        print(f"{'='*50}")
        print(f"Target years processed: {target_years}")
        print(f"Total forecast-observation pairs: {len(combined_forecast_obs)}")
        print(
            f"Probability range: {combined_forecast_obs['predicted_prob'].min():.3f} - {combined_forecast_obs['predicted_prob'].max():.3f}"
        )
        print(
            f"Overall observed onset rate: {combined_forecast_obs['observed_onset'].mean():.3f}"
        )

        return combined_forecast_obs