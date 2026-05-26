"""Utility functions for generating Figure 6 of the monsoon benchmark paper."""

import itertools
import warnings
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from plot_config import LARGE_SIZE, MEDIUM_SIZE, SMALL_SIZE, params

from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    OnsetMetricsBase,
    ProbabilisticOnsetMetrics,
)
from monsoonbench.spatial.regions import get_india_outline

warnings.filterwarnings("ignore")

# Apply plot settings
plt.rcParams.update(params)

c = ClimatologyOnsetMetrics()
p = ProbabilisticOnsetMetrics()
d = DeterministicOnsetMetrics()
o = OnsetMetricsBase()

model_str = ["Climatology", "IFS", "AIFS", "FuXi", "Graphcast", "GenCast", "FuXi-S2S", "NGCM"]

# Define the standard grid
lat_grid = np.arange(8, 37, 4)  # 8:4:36
lon_grid = np.arange(68, 101, 4)  # 68:4:100


# Define Core Monsoon Zone polygon (same as reference)
polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])

panel_linewidth = 0.5
map_lw = 0.5
polygon_lw = 1
tick_length = 2
tick_width = 0.5

brier_levels = np.array([-40, -30, -20, -10, 0, 10, 20, 30, 40])
rps_levels = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80])



PROB_MODELS = {"FuXi-S2S", "NGCM", "IFS", "GenCast"}

DET_MODELS = {"AIFS", "FuXi", "Graphcast"}


# Helpers

def points_inside_polygon(
    polygon_lon, polygon_lat, grid_lons, grid_lats
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find grid points that are inside a polygon.

    Parameters:
    polygon_lon: array of polygon longitude vertices
    polygon_lat: array of polygon latitude vertices
    grid_lons: array of grid longitude points
    grid_lats: array of grid latitude points

    Returns:
    inside_mask: boolean array indicating which points are inside
    inside_lons: longitude coordinates of points inside polygon
    inside_lats: latitude coordinates of points inside polygon
    """
    # Create polygon path
    polygon_vertices = np.column_stack((polygon_lon, polygon_lat))
    polygon_path = MplPath(polygon_vertices)

    # Create meshgrid if needed
    if grid_lons.ndim == 1 and grid_lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    else:
        lon_grid, lat_grid = grid_lons, grid_lats

    # Flatten the grids to test each point
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    # Test which points are inside the polygon
    inside_mask = polygon_path.contains_points(points)
    inside_mask = inside_mask.reshape(lon_grid.shape)

    # Get coordinates of points inside polygon
    inside_lons = lon_grid[inside_mask]
    inside_lats = lat_grid[inside_mask]

    return inside_mask, inside_lons, inside_lats




def get_clim_onset_da(config: dict,
                   years = None) -> xr.Dataset:
    """Function to load onset dataarray for climatology"""
    thresh_ds = xr.open_dataset(config["thresh_file"])
    thresh_slice = thresh_ds["MWmean"]
    clim_onset = c.compute_climatological_onset_dataset(
                config["imd_folder"], thresh_slice,
                years=years, mok=config["mok"]
            )
    return clim_onset


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
    ) -> pd.DataFrame:
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

        polygon1_lon = np.array([67, 101, 101, 67, 67])
        polygon1_lat = np.array([7, 7, 37, 37, 7])
        print("Using 4-degree CMZ polygon coordinates")

        inside_mask, inside_lons, inside_lats = points_inside_polygon(
            polygon1_lon, polygon1_lat, orig_lon, orig_lat
        )
        inside_lats = np.unique(inside_lats)
        inside_lons = np.unique(inside_lons)
        thresh_slice = thresh_da.sel(lat=inside_lats, lon=inside_lons)

        # Initialize list to store all forecast-observation pairs
        all_forecast_obs_pairs = []


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
                onset_all_members = (fig6_compute_onset_for_all_members(
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


def fig6_compute_onset_for_all_members(
        p_model, thresh_slice, onset_da, max_forecast_day=15, mok=True
    ) -> pd.DataFrame:
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
        # unique_pairs = list(zip(lons, lats))
        unique_pairs = list(itertools.product(lons, lats))

        date_method = "MOK (June 2nd filter)" if mok else "no date filter"
        print(
            f"Processing {len(init_times)} init times x {len(unique_pairs)} unique locations x {len(members)} members..."
        )
        # print(f"Unique lat-lon pairs: {unique_pairs}")
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
                    f"Processing init time {t_idx + 1}/{len(init_times)}: {pd.to_datetime(init_time).strftime('%Y-%m-%d')}"
                )

            init_date = pd.to_datetime(init_time)
            year = init_date.year
            mok_date = datetime(year, 6, 2)

            # Loop over unique lat-lon pairs only
            for lon in lons:
                for lat in lats:    
                    total_potential_forecasts += len(members)

                    # Get observed onset date for this location
                    try:
                        obs_onset = onset_da.sel(lat=lat, lon=lon).values
                    except Exception:
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
                    thresh = thresh_slice.sel(lat=lat, lon=lon).values

                    for m_idx, member in enumerate(members):
                        valid_forecasts += 1

                        try:
                            # Extract forecast time series for this member and location
                            forecast_series = p_model.sel(
                                init_time=init_time,
                                lat=lat,
                                lon=lon,
                                member=member,
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
        return onset_df


def get_fig_6_model_data(config) -> tuple:
    """Function for loading model data for figure 6
    
    Probabalistic metrics.
    """
    forecast_dfs_15 = {}
    forecast_dfs_30 = {}
    brier_model_paths = {
        "FuXi-S2S": config["model_paths"]["FuXi-S2S"],
        "NGCM": config["model_paths"]["NGCM"],
        "IFS": config["model_paths"]["IFS"],
    }
    for model_name, model_fp in brier_model_paths.items():
        print("=" * 80)
        print(f"Loading data from {model_name}")
        print("=" * 80)
        multi_year_df = fig6_multi_year_forecast_obs_pairs(
            config["common_period"],
            model_forecast_dir=model_fp,
            imd_folder=config["imd_folder"],
            thres_file=config["thresh_file"],
            mem_num=51 if model_name != "IFS" else 11,
            max_forecast_day=15,
            day_bins=config["day_bins_15"],
            date_filter_year=2022 if model_name == "FuXi-S2S" else 2024,
        )
        forecast_dfs_15[model_name] = multi_year_df

        multi_year_df = fig6_multi_year_forecast_obs_pairs(
            config["common_period"],
            model_forecast_dir=model_fp,
            imd_folder=config["imd_folder"],
            thres_file=config["thresh_file"],
            mem_num=51 if model_name != "IFS" else 11,
            max_forecast_day=30,
            day_bins=config["day_bins_30"],
            date_filter_year=2022 if model_name == "FuXi-S2S" else 2024,
        )
        forecast_dfs_30[model_name] = multi_year_df

    return forecast_dfs_15, forecast_dfs_30


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
    ) -> pd.DataFrame:
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


def get_fig_6_clim_data(clim_onset, config,
                         model_forecast_dir=None,
                         date_filter_year=2024,
                         mem_num=None) -> tuple:
    """Figure for loading climatology data for figure 6"""
    model_fp = model_forecast_dir or config["model_paths"]["NGCM"]
    effective_mem_num = mem_num if mem_num is not None else config["mem_num"]

    print("=" * 50)
    print(f"Loading climatology data for model path: {model_fp}")
    print("=" * 50)

    climatology_obs_df_15 = fig6_multi_year_climatological_forecast_obs_pairs(
        clim_onset,
        config["common_period"],
        config["day_bins_15"],
        effective_mem_num,
        model_fp,
        date_filter_year=date_filter_year,
        file_pattern=config["file_pattern"],
        max_forecast_day=15,
        mok=config["mok"],
    )

    climatology_obs_df_30 = fig6_multi_year_climatological_forecast_obs_pairs(
        clim_onset,
        config["common_period"],
        config["day_bins_30"],
        effective_mem_num,
        model_fp,
        date_filter_year=date_filter_year,
        file_pattern=config["file_pattern"],
        max_forecast_day=30,
        mok=config["mok"],
    )

    return climatology_obs_df_15, climatology_obs_df_30


def fig6_metric_calculation(forecast_df, clim_df, n=15, model_name=None) -> pd.DataFrame:
    """Function for calculating Brier and RPS for figure 6"""
    rows = []
    for lat in forecast_df.lat.unique():
        for lon in forecast_df.lon.unique():
            row = {}
            loop_df = forecast_df.loc[
                (forecast_df["lat"] == lat) & (forecast_df["lon"] == lon)
            ].copy()

            if loop_df.empty:
                continue

            # Filter climatology to the same cell
            clim_loop_df = clim_df.loc[
                (clim_df["lat"] == lat) & (clim_df["lon"] == lon)
            ].copy()

            if clim_loop_df.empty:
                continue

            # Compute per-cell climatology scores
            clim_brier = c.calculate_brier_score_climatology(clim_loop_df)  
            clim_rps = p.calculate_rps(clim_loop_df)

            brier = p.calculate_brier_score(loop_df)
            rps = p.calculate_rps(loop_df)

            skill_scores = p.calculate_skill_scores(
                brier_forecast=brier,
                rps_forecast=rps,
                brier_climatology=clim_brier,
                rps_climatology=clim_rps,
            )

            row["fair_brier_skill"] = skill_scores["fair_brier_skill_score"]
            row["fair_rps_skill"] = skill_scores["fair_rps_skill_score"]
            row["lat"] = lat
            row["lon"] = lon
            row["horizon"] = n
            if model_name:
                row["dataset"] = model_name
            rows.append(row)

    return pd.DataFrame(rows)


def create_gridded_data(df, metric, model, horizon) -> xr.DataArray:
    """Convert CSV data to gridded xarray DataArray with standard lat-lon grid"""
    # Filter data for specific model and horizon
    subset = df[(df["dataset"] == model) & (df["horizon"] == horizon)]
    
    if subset.empty:
        # Return empty grid with NaNs if no data
        data_grid = np.full((len(lat_grid), len(lon_grid)), np.nan)
        return xr.DataArray(
            data_grid, 
            coords={"lat": lat_grid, "lon": lon_grid},
            dims=["lat", "lon"]
        )
    
    # Initialize grid with NaNs
    data_grid = np.full((len(lat_grid), len(lon_grid)), np.nan)
    
    # Fill grid with available data
    for _, row in subset.iterrows():
        lat_val = row["lat"]
        lon_val = row["lon"]
        
        # Find nearest grid point
        lat_idx = np.argmin(np.abs(lat_grid - lat_val))
        lon_idx = np.argmin(np.abs(lon_grid - lon_val))
        
        # Only assign if the point is close enough (within 0.1 degrees)
        if (np.abs(lat_grid[lat_idx] - lat_val) < 0.1 and 
            np.abs(lon_grid[lon_idx] - lon_val) < 0.1):
            # Multiply by 100 to convert to percentage
            data_grid[lat_idx, lon_idx] = row[metric] * 100
    
    # Create xarray DataArray
    da = xr.DataArray(
        data_grid, 
        coords={"lat": lat_grid, "lon": lon_grid},
        dims=["lat", "lon"],
        attrs={"units": "skill_score_percent", "model": model, "horizon": horizon, "metric": metric}
    )
    
    return da


def create_discrete_colormap(levels, base_cmap="RdBu") -> tuple:
    """Create a discrete colormap with specified levels"""
    cmap = plt.cm.get_cmap(base_cmap)
    norm = colors.BoundaryNorm(levels, cmap.N, clip=True)
    return cmap, norm


def create_skill_map_panel_xr(ax, data_array, model, metric_type,
                              config, model_labels, levels=None,
                              show_ylabel=True, title=None,
                              ) -> None:
    """Create a skill map panel using xarray DataArray with India boundaries"""
    if data_array.isna().all():
        ax.text(0.5, 0.5, f"No data for {model}", 
                transform=ax.transAxes, ha="center", va="center")
        return None
    
    # Get coordinates
    lats = data_array.lat.values
    lons = data_array.lon.values
    
    # Create edges for pcolormesh
    lat_edges = np.concatenate([lats - 2, [lats[-1] + 2]])
    lon_edges = np.concatenate([lons - 2, [lons[-1] + 2]])
    # lon_edges = np.concatenate([lon - (lon[1]-lon[0])/2, [lon[-1] + (lon[1]-lon[0])/2]])
    # lat_edges = np.concatenate([lat - (lat[1]-lat[0])/2, [lat[-1] + (lat[1]-lat[0])/2]])
    LON_edges, LAT_edges = np.meshgrid(lon_edges, lat_edges)

        
    # Create discrete colormap and normalization
    if levels is not None:
        cmap, norm = create_discrete_colormap(levels, "RdBu")
        vmin, vmax = None, None  # Let norm handle the range
    else:
        cmap = "RdBu"
        norm = None
        vmin, vmax = -100, 100
    
    # Create pcolormesh plot
    im = ax.pcolormesh(LON_edges, LAT_edges, data_array.values,
                      transform=ccrs.PlateCarree(),
                      cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading="flat")
    
    # ...existing code for boundaries, polygon, etc...
    # Add India boundaries using the get_india_outline function
    try:
        india_boundaries = get_india_outline(shp_file_path=config["shpfile_path"])
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            ax.plot(india_lon, india_lat, color="black", linewidth=map_lw, 
                   transform=ccrs.PlateCarree())
    except Exception as e:
        print(f"Warning: Could not load India boundaries: {e}")
        ax.add_feature(cfeature.COASTLINE, linewidth=map_lw, color="black")
    
    # Add Core Monsoon Zone polygon
    polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                     fill=False, edgecolor="black", linewidth=polygon_lw,
                     transform=ccrs.PlateCarree())
    ax.add_patch(polygon)
    
    # Calculate average value inside the polygon
    from matplotlib.path import Path
    polygon_path = Path(list(zip(polygon1_lon, polygon1_lat)))
    
    # Find grid points inside the polygon
    values_in_polygon = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if polygon_path.contains_point((lon, lat)):
                value = data_array.values[i, j]
                if not np.isnan(value):
                    values_in_polygon.append(value)
    
    # Calculate and display average
    if values_in_polygon:
        avg_value = np.mean(values_in_polygon)
        ax.text(0.95, 0.05, f"{avg_value:.1f}%", 
                transform=ax.transAxes,
                horizontalalignment="right", verticalalignment="bottom",
                color="black", fontsize=MEDIUM_SIZE, fontweight="normal")
    
    # Add model name text
    model_label = model_labels.get(model, model.upper())
    ax.text(0.95, 0.95, model_label, transform=ax.transAxes,
            horizontalalignment="right", verticalalignment="top",
            color="black", fontsize=MEDIUM_SIZE, fontweight="normal")
    
    # Set axis limits and ticks
    ax.set_xlim([lons[0]-4, 100])
    ax.set_ylim([lats[0]-4, lats[-1]+4])
    
    # Create tick labels
    yticks = np.arange(lats[0]-2, lats[-1]+3, 8)
    yticklabels = [f"{int(y)}°N" if i % 1 == 0 else "" for i, y in enumerate(yticks)]
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    if show_ylabel:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels([])
    
    xticks = np.arange(lons[0]-2, lons[-1]+3, 8)
    xticklabels = [f"{int(x)}°E" if i % 1 == 0 else "" for i, x in enumerate(xticks)]
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels(xticklabels)
    
    # Styling
    ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE, 
                  length=tick_length, width=tick_width)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(panel_linewidth)
    
    # Remove grid lines
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    
    if title:
        ax.text(0.02, 1.02, title, transform=ax.transAxes, 
                verticalalignment="bottom", fontsize=LARGE_SIZE, fontweight="normal")
    
    return im, levels


# Update the main figure creation function
def create_skill_maps_figure_xr(df,
                                config,
                                ) -> plt.Figure:
    """Create the complete figure using xarray DataArrays with discrete color levels"""
    models = ["IFS", "FuXi-S2S", "NGCM"]
    # Create the main figure
    fig = plt.figure(figsize=(8, 8), dpi=300)
    
    # Create GridSpec for better control - 4 rows, 10 columns
    gs = GridSpec(
        4, 10, figure=fig,
        hspace=0.2, wspace=0.02,
        left=0.08, right=0.9, top=0.95, bottom=0.08,
        height_ratios=[1, 1, 1, 1]
    )
    
    # ...existing titles and row_configs...
    titles = {
        0: "(a) Brier Skill Score: 15-day forecast",
        1: "(b) Brier Skill Score: 30-day forecast", 
        2: "(c) Ranked Probability Skill Score: 15-day forecast",
        3: "(d) Ranked Probability Skill Score: 30-day forecast"
    }
    
    model_labels = {
    "IFS": "IFS",
    "FuXi-S2S": "FuXi-S2S",
    "NGCM": "NGCM"
    }

    row_configs = [
        ("fair_brier_skill", 15),
        ("fair_brier_skill", 30),
        ("fair_rps_skill", 15),
        ("fair_rps_skill", 30)
    ]
    
    axes = []
    data_arrays = []
    colorbars_data = []
    
    for row_idx, (metric, horizon) in enumerate(row_configs):
        row_axes = []
        row_data = []
        
        # Choose levels based on metric
        if "brier" in metric:
            levels = brier_levels
        else:  # RPS
            levels = rps_levels
        
        for model_idx, model in enumerate(models):
            col_start = model_idx * 3
            col_end = col_start + 3
            
            ax = fig.add_subplot(gs[row_idx, col_start:col_end], 
                               projection=ccrs.PlateCarree())
            row_axes.append(ax)
            
            data_array = create_gridded_data(df, metric, model, horizon)
            row_data.append(data_array)
            
            show_ylabel = (model_idx == 0)
            title = titles.get(row_idx) if model_idx == 0 else None
            
            # Create the map with discrete levels
            result = create_skill_map_panel_xr(ax, data_array, model,
                                               metric, config, model_labels=model_labels,
                                               levels=levels, show_ylabel=show_ylabel,
                                               title=title)
            
            if result is not None:
                im, used_levels = result
                if model_idx == 0 and im is not None:
                    colorbars_data.append((row_idx, im, metric, used_levels))
        
        axes.append(row_axes)
        data_arrays.append(row_data)
    
    # Update colorbar creation with discrete levels
    if len(colorbars_data) >= 2:
        brier_data = [item for item in colorbars_data if "brier" in item[2]]
        if brier_data:
            _, im, _, levels = brier_data[0]
            
            pos_row0 = axes[0][0].get_position()
            pos_row1 = axes[1][0].get_position()
            total_height = pos_row0.y1 - pos_row1.y0
            half_height = total_height * 0.5
            center_y = (pos_row0.y1 + pos_row1.y0) / 2
            
            cax_brier = fig.add_axes([0.8, center_y - half_height/2, 0.01, half_height])
            
            cbar_brier = fig.colorbar(im, cax=cax_brier, orientation="vertical", extend="both")
            cbar_brier.set_ticks(levels[::2])  # Show every other tick to avoid crowding
            cbar_brier.set_label("BSS (%)", fontsize=MEDIUM_SIZE, rotation=270, labelpad=8)
            cbar_brier.ax.tick_params(labelsize=SMALL_SIZE, length=2, width=1)
    
    if len(colorbars_data) >= 4:
        rps_data = [item for item in colorbars_data if "rps" in item[2]]
        if rps_data:
            _, im, _, levels = rps_data[0]
            
            pos_row2 = axes[2][0].get_position()
            pos_row3 = axes[3][0].get_position()
            total_height = pos_row2.y1 - pos_row3.y0
            half_height = total_height * 0.5
            center_y = (pos_row2.y1 + pos_row3.y0) / 2
            
            cax_rps = fig.add_axes([0.8, center_y - half_height/2, 0.01, half_height])
            
            cbar_rps = fig.colorbar(im, cax=cax_rps, orientation="vertical", extend="both")
            cbar_rps.set_ticks(levels[::2])  # Show every other tick
            cbar_rps.set_label("RPSS (%)", fontsize=MEDIUM_SIZE, rotation=270, labelpad=8)
            cbar_rps.ax.tick_params(labelsize=SMALL_SIZE, length=2, width=1)
    
    # Remove x-tick labels from top three rows
    for row_idx in range(3):
        for ax in axes[row_idx]:
            ax.set_xticklabels([])
    
    plt.tight_layout()
    
    # Save the figure
    # plt.savefig('outputs/fig6.png', dpi=600, bbox_inches='tight')
    # plt.savefig('outputs/fig6.pdf', dpi=600, bbox_inches='tight')

    return fig, data_arrays


def generate_fig6(config) -> plt.Figure:
    """Function for generating figure 6"""
    try:
        output_dir = config["output_dir"]
        fig6_metrics = pd.read_csv(f"{output_dir}/probabalistic_scores_15_30_day_2004_2021.csv")

    except FileNotFoundError:

        clim_onset = get_clim_onset_da(config)
        forecast_dfs_15, forecast_dfs_30 = get_fig_6_model_data(config)

    # Build a separate climatology for each model using its own init dates
        clim_data = {}
        for model_name, model_fp in config["model_paths"].items():
            if model_name not in ["FuXi-S2S", "NGCM", "IFS"]:
                continue
            date_filter_year = 2022 if model_name == "FuXi-S2S" else 2024
            mem_num = 11 if model_name == "IFS" else 51

            clim_15, clim_30 = get_fig_6_clim_data(
                clim_onset,
                config,
                model_forecast_dir=model_fp,
                date_filter_year=date_filter_year,
                mem_num=mem_num,
            )
            clim_data[model_name] = {"15": clim_15, "30": clim_30}

        fig6_metrics_dict_15 = {}
        fig6_metrics_dict_30 = {}

        for key, value in forecast_dfs_15.items():
            model_clim_15 = clim_data[key]["15"]

            # Pass the full climatology DataFrame — scoring is now per-cell inside
            loop_ret = fig6_metric_calculation(
                value,
                clim_df=model_clim_15,
                n=15,
                model_name=key,
            )
            fig6_metrics_dict_15[key] = loop_ret

        for key, value in forecast_dfs_30.items():
            model_clim_30 = clim_data[key]["30"]

            loop_ret = fig6_metric_calculation(
                value,
                clim_df=model_clim_30,
                n=30,
                model_name=key,
            )
            fig6_metrics_dict_30[key] = loop_ret

        fig6_metrics_15 = pd.concat(fig6_metrics_dict_15.values())
        fig6_metrics_30 = pd.concat(fig6_metrics_dict_30.values())
        fig6_metrics = pd.concat([fig6_metrics_15, fig6_metrics_30])

        save_loco = f"{output_dir}/probabalistic_scores_15_30_day_2004_2021.csv"
        print(f"Saving loaded data to {save_loco}")
        fig6_metrics.to_csv(save_loco)

    skill_fig, gridded_data = create_skill_maps_figure_xr(
        df=fig6_metrics,
        config=config,
    )
    plt.show()

    return skill_fig, gridded_data
