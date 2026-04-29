from monsoonbench.metrics import (
    ProbabilisticOnsetMetrics,
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    OnsetMetricsBase
)

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.path import Path as MplPath


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import matplotlib.colors as colors

import warnings
from monsoonbench.spatial.regions import get_india_outline
warnings.filterwarnings('ignore')
from plot_config import params, contourLevels, colormap, savefig_format, SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE

# Apply plot settings
plt.rcParams.update(params)

c = ClimatologyOnsetMetrics()
p = ProbabilisticOnsetMetrics()
d = DeterministicOnsetMetrics()
o = OnsetMetricsBase()

model_str = ['Climatology', 'IFS', 'AIFS', 'FuXi', 'Graphcast', 'GenCast', 'FuXi-S2S', 'NGCM']

# Define the standard grid
lat_grid = np.arange(8, 37, 4)  # 8:4:36
lon_grid = np.arange(68, 101, 4)  # 68:4:100

# Plot configuration
SMALL_SIZE = 6
MEDIUM_SIZE = 7
LARGE_SIZE = 8

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


    # Define colors for MAE, FAR, and MR (matching your bar chart colors)
mae_col = np.array([217, 95, 14]) / 256  # Orange
far_col = np.array([49, 130, 189]) / 256  # Blue
mr_col = np.array([188, 189, 220]) / 256  # Light purple
    # Create figures for MAE, FAR, and MR

model_years = {
    "FuXi-S2S": [2019, 2020, 2021],
    "IFS": [2019, 2020, 2021, 2022, 2023],
    "Standard": [2019, 2020, 2021, 2022, 2023, 2024],
}

PROB_MODELS = {"FuXi-S2S", "NGCM", "IFS", "GenCast"}

DET_MODELS = {"AIFS", "FuXi", "Graphcast"}

# Helpers
def ensure_lat_lon_sorted(ds: xr.Dataset) -> xr.Dataset:
    """Ensure consistent lat/lon ordering and alignment."""
    return ds.sortby(["lat", "lon"])


def spatial_dict_to_panels(spatial_dict, metric):
    """
    Convert dict[str, xr.Dataset] → list[list[xr.DataArray]]
    Single row of 8 panels.
    """
    row = [ds[metric] for ds in spatial_dict.values()]
    return row


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
                   years = None):
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


def get_fig_6_model_data(config):
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


def get_fig_6_clim_data(clim_onset, config,
                         model_forecast_dir=None,
                         date_filter_year=2024,
                         mem_num=None):
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


def get_clim_brier(df):
    clim_brier = c.calculate_brier_score_climatology(df)
    return clim_brier


def get_clim_rps(df):
    clim_rps = p.calculate_rps(df)
    return clim_rps


# def fig6_metric_calculation(forecast_df,
#                             clim_brier,
#                             clim_rps,
#                             n=15,
#                             model_name=None,
#                             ):
    
#     rows= []
#     for lat in forecast_df.lat.unique():
#         for lon in forecast_df.lon.unique():
#             print("="*50)
#             print(f"Calculating for {lat}, {lon} pair")
#             print("="*50)
#             row = {}
#             loop_df = forecast_df.loc[
#                 (forecast_df["lat"] == lat) & (forecast_df["lon"] == lon)
#                 ].copy()
#             if loop_df.empty:
#                 continue
#             else:
#                 brier = p.calculate_brier_score(loop_df)
#                 rps = p.calculate_rps(loop_df)
#                 skill_scores = p.calculate_skill_scores(
#                             brier_forecast=brier,
#                             rps_forecast=rps,
#                             brier_climatology=clim_brier,
#                             rps_climatology=clim_rps,
#                         )
#                 row["fair_brier_skill"] = skill_scores["fair_brier_skill_score"]
#                 row["fair_rps_skill"] = skill_scores["fair_rps_skill_score"]
#                 row["lat"] = lat
#                 row["lon"] = lon
#                 row["horizon"] = n
#                 if model_name:
#                     row["dataset"] = model_name
#                 rows.append(row)

#     return pd.DataFrame(rows)
                        

def fig6_metric_calculation(forecast_df, clim_df, n=15, model_name=None):
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


def create_gridded_data(df, metric, model, horizon):
    """
    Convert CSV data to gridded xarray DataArray with standard lat-lon grid
    """
    # Filter data for specific model and horizon
    subset = df[(df['dataset'] == model) & (df['horizon'] == horizon)]
    
    if subset.empty:
        # Return empty grid with NaNs if no data
        data_grid = np.full((len(lat_grid), len(lon_grid)), np.nan)
        return xr.DataArray(
            data_grid, 
            coords={'lat': lat_grid, 'lon': lon_grid},
            dims=['lat', 'lon']
        )
    
    # Initialize grid with NaNs
    data_grid = np.full((len(lat_grid), len(lon_grid)), np.nan)
    
    # Fill grid with available data
    for _, row in subset.iterrows():
        lat_val = row['lat']
        lon_val = row['lon']
        
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
        coords={'lat': lat_grid, 'lon': lon_grid},
        dims=['lat', 'lon'],
        attrs={'units': 'skill_score_percent', 'model': model, 'horizon': horizon, 'metric': metric}
    )
    
    return da


def create_discrete_colormap(levels, base_cmap='RdBu'):
    """Create a discrete colormap with specified levels"""
    cmap = plt.cm.get_cmap(base_cmap)
    norm = colors.BoundaryNorm(levels, cmap.N, clip=True)
    return cmap, norm


def create_skill_map_panel_xr(ax, data_array, model, metric_type,
                              config, model_labels, levels=None,
                              show_ylabel=True, title=None,
                              ):
    """
    Create a skill map panel using xarray DataArray with India boundaries
    """
    
    if data_array.isnull().all():
        ax.text(0.5, 0.5, f'No data for {model}', 
                transform=ax.transAxes, ha='center', va='center')
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
        cmap, norm = create_discrete_colormap(levels, 'RdBu')
        vmin, vmax = None, None  # Let norm handle the range
    else:
        cmap = 'RdBu'
        norm = None
        vmin, vmax = -100, 100
    
    # Create pcolormesh plot
    im = ax.pcolormesh(LON_edges, LAT_edges, data_array.values,
                      transform=ccrs.PlateCarree(),
                      cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading='flat')
    
    # ...existing code for boundaries, polygon, etc...
    # Add India boundaries using the get_india_outline function
    try:
        india_boundaries = get_india_outline(shp_file_path=config["shpfile_path"])
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            ax.plot(india_lon, india_lat, color='black', linewidth=map_lw, 
                   transform=ccrs.PlateCarree())
    except Exception as e:
        print(f"Warning: Could not load India boundaries: {e}")
        ax.add_feature(cfeature.COASTLINE, linewidth=map_lw, color='black')
    
    # Add Core Monsoon Zone polygon
    polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                     fill=False, edgecolor='black', linewidth=polygon_lw,
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
        ax.text(0.95, 0.05, f'{avg_value:.1f}%', 
                transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom',
                color='black', fontsize=MEDIUM_SIZE, fontweight='normal')
    
    # Add model name text
    model_label = model_labels.get(model, model.upper())
    ax.text(0.95, 0.95, model_label, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            color='black', fontsize=MEDIUM_SIZE, fontweight='normal')
    
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
    ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, 
                  length=tick_length, width=tick_width)
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_linewidth(panel_linewidth)
    
    # Remove grid lines
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.tick_params(axis='y', which='minor', left=False, right=False)
    
    if title:
        ax.text(0.02, 1.02, title, transform=ax.transAxes, 
                verticalalignment='bottom', fontsize=LARGE_SIZE, fontweight='normal')
    
    return im, levels


# Update the main figure creation function
def create_skill_maps_figure_xr(df,
                                config,
                                models:list = ['IFS', 'FuXi-S2S', 'NGCM']):
    """Create the complete figure using xarray DataArrays with discrete color levels"""
    
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
        0: '(a) Brier Skill Score: 15-day forecast',
        1: '(b) Brier Skill Score: 30-day forecast', 
        2: '(c) Ranked Probability Skill Score: 15-day forecast',
        3: '(d) Ranked Probability Skill Score: 30-day forecast'
    }
    
    model_labels = {
    'IFS': 'IFS',
    'FuXi-S2S': 'FuXi-S2S',
    'NGCM': 'NGCM'
    }

    row_configs = [
        ('fair_brier_skill', 15),
        ('fair_brier_skill', 30),
        ('fair_rps_skill', 15),
        ('fair_rps_skill', 30)
    ]
    
    axes = []
    data_arrays = []
    colorbars_data = []
    
    for row_idx, (metric, horizon) in enumerate(row_configs):
        row_axes = []
        row_data = []
        
        # Choose levels based on metric
        if 'brier' in metric:
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
        brier_data = [item for item in colorbars_data if 'brier' in item[2]]
        if brier_data:
            _, im, _, levels = brier_data[0]
            
            pos_row0 = axes[0][0].get_position()
            pos_row1 = axes[1][0].get_position()
            total_height = pos_row0.y1 - pos_row1.y0
            half_height = total_height * 0.5
            center_y = (pos_row0.y1 + pos_row1.y0) / 2
            
            cax_brier = fig.add_axes([0.8, center_y - half_height/2, 0.01, half_height])
            
            cbar_brier = fig.colorbar(im, cax=cax_brier, orientation='vertical', extend='both')
            cbar_brier.set_ticks(levels[::2])  # Show every other tick to avoid crowding
            cbar_brier.set_label('BSS (%)', fontsize=MEDIUM_SIZE, rotation=270, labelpad=8)
            cbar_brier.ax.tick_params(labelsize=SMALL_SIZE, length=2, width=1)
    
    if len(colorbars_data) >= 4:
        rps_data = [item for item in colorbars_data if 'rps' in item[2]]
        if rps_data:
            _, im, _, levels = rps_data[0]
            
            pos_row2 = axes[2][0].get_position()
            pos_row3 = axes[3][0].get_position()
            total_height = pos_row2.y1 - pos_row3.y0
            half_height = total_height * 0.5
            center_y = (pos_row2.y1 + pos_row3.y0) / 2
            
            cax_rps = fig.add_axes([0.8, center_y - half_height/2, 0.01, half_height])
            
            cbar_rps = fig.colorbar(im, cax=cax_rps, orientation='vertical', extend='both')
            cbar_rps.set_ticks(levels[::2])  # Show every other tick
            cbar_rps.set_label('RPSS (%)', fontsize=MEDIUM_SIZE, rotation=270, labelpad=8)
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


# def generate_fig6(config, clim_onset):

#     # Point 5 fix: actually call the functions with ()
#     forecast_dfs_15, forecast_dfs_30 = get_fig_6_model_data(config)

#     # Point 3 fix: build a separate climatology for each model using its own init dates
#     clim_data = {}
#     for model_name, model_fp in config["model_paths"].items():
#         if model_name not in ["FuXi-S2S", "NGCM", "IFS"]:
#             continue
#         date_filter_year = 2022 if model_name == "IFS" else 2024
#         mem_num = 11 if model_name == "IFS" else 51

#         clim_15, clim_30 = get_fig_6_clim_data(
#             clim_onset,
#             config,
#             model_forecast_dir=model_fp,
#             date_filter_year=date_filter_year,
#             mem_num=mem_num,
#         )
#         clim_data[model_name] = {"15": clim_15, "30": clim_30}

#     fig6_metrics_dict_15 = {}
#     fig6_metrics_dict_30 = {}

#     for key, value in forecast_dfs_15.items():
#         # Point 3 fix: use the model-specific climatology as reference
#         model_clim_15 = clim_data[key]["15"]

#         # Point 4 fix: exclude "earlier"-equivalent rows from Brier, keep all for RPS
#         clim_brier_15 = get_clim_brier(
#             model_clim_15
#         )
#         clim_rps_15 = get_clim_rps(model_clim_15)

#         loop_ret = fig6_metric_calculation(
#             value,
#             clim_brier=clim_brier_15,
#             clim_rps=clim_rps_15,
#             n=15,
#             model_name=key,
#         )
#         fig6_metrics_dict_15[key] = loop_ret

#     for key, value in forecast_dfs_30.items():
#         model_clim_30 = clim_data[key]["30"]

#         clim_brier_30 = get_clim_brier(
#             model_clim_30
#         )
#         clim_rps_30 = get_clim_rps(model_clim_30)

#         loop_ret = fig6_metric_calculation(
#             value,
#             clim_brier=clim_brier_30,
#             clim_rps=clim_rps_30,
#             n=30,
#             model_name=key,
#         )
#         fig6_metrics_dict_30[key] = loop_ret

#     fig6_metrics_15 = pd.concat(fig6_metrics_dict_15.values())
#     fig6_metrics_30 = pd.concat(fig6_metrics_dict_30.values())

#     fig6_metrics = pd.concat([fig6_metrics_15, fig6_metrics_30])

#     skill_fig, gridded_data = create_skill_maps_figure_xr(
#         df=fig6_metrics,
#         config=config,
#     )
#     plt.show()

#     return skill_fig, gridded_data


def generate_fig6(config, clim_onset):

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

    skill_fig, gridded_data = create_skill_maps_figure_xr(
        df=fig6_metrics,
        config=config,
    )
    plt.show()

    return skill_fig, gridded_data


def get_spatial_fig_clim_data(config, n=15):
    if n==15:
        metrics_df_clim_15, onset_da_clim_15 = (
                c.compute_climatology_baseline_multiple_years(
                    years=model_years["Standard"],
                    imd_folder=config["imd_folder"],
                    thres_file=config["thresh_file"],
                    tolerance_days=3,
                    verification_window=1,
                    forecast_days=15,
                    max_forecast_day=15,
                    mok=True,
                    onset_window=5,
                    mok_month=6,
                    mok_day=2,
                )
            )

        spatial_clim_15_day = c.create_spatial_far_mr_mae(
                metrics_df_clim_15, dict.fromkeys(model_years["Standard"], onset_da_clim_15)
            )
        return spatial_clim_15_day
    else:
        metrics_df_clim_30, onset_da_clim_30 = (
        c.compute_climatology_baseline_multiple_years(
            years=model_years["Standard"],
            imd_folder=config["imd_folder"],
            thres_file=config["thresh_file"],
            tolerance_days=5,
            verification_window=16,
            forecast_days=30,
            max_forecast_day=30,
            mok=True,
            onset_window=5,
            mok_month=6,
            mok_day=2,
            )
        )

        spatial_clim_30_day = c.create_spatial_far_mr_mae(
                metrics_df_clim_30, dict.fromkeys(model_years["Standard"], onset_da_clim_30)
            )
        
        return spatial_clim_30_day

def get_spatial_fig_model_data(config, n=15):
    prob_model_paths = {
        "FuXi-S2S": config["model_paths"]["FuXi-S2S"],  # FuXi_S2S model
        "NGCM": config["model_paths"]["NGCM"],  # NGCM model
        "IFS": config["model_paths"]["IFS"],  # AIFS model
        "GenCast": config["model_paths"]["GenCast"],  # GenCast model
    }

    det_model_paths = {
            "AIFS": config["model_paths"]["AIFS"],
            "FuXi": config["model_paths"]["FuXi"],
            "Graphcast": config["model_paths"]["Graphcast"],  # Graphcast model
        }
        # 15 day
    model_dfs = {}
    model_onsets = {}
    if n == 15:
        for model_name, model_fp in prob_model_paths.items():
            print("=" * 80)
            print(f"Loading data from {model_name}")
            print("=" * 80)
            probabilistic_df_15, onset_da_dict_15 = (
                p.compute_metrics_multiple_years(
                    years=(
                        model_years[model_name]
                        if model_name in model_years.keys()
                        else model_years["Standard"]
                    ),
                    imd_folder=config["imd_folder"],
                    thres_file=config["thresh_file"],
                    model_forecast_dir=model_fp,
                    tolerance_days=3,
                    verification_window=1,
                    forecast_days=15,
                    max_forecast_day=15,
                    mok=True,
                    onset_window=5,
                    mok_month=6,
                    mok_day=2,
                )
            )

            model_dfs[model_name] = probabilistic_df_15
            model_onsets[model_name] = onset_da_dict_15

        for model_name, model_fp in det_model_paths.items():
                print("=" * 80)
                print(f"Loading data from {model_name}")
                print("=" * 80)
                deterministic_df_15, onset_da_dict_15 = (
                    p.compute_metrics_multiple_years(
                        years=(
                            model_years[model_name]
                            if model_name in model_years.keys()
                            else model_years["Standard"]
                        ),
                        imd_folder=config["imd_folder"],
                        thres_file=config["thresh_file"],
                        model_forecast_dir=model_fp,
                        tolerance_days=3,
                        verification_window=1,
                        forecast_days=15,
                        max_forecast_day=15,
                        mok=True,
                        onset_window=5,
                        mok_month=6,
                        mok_day=2,
                    )
                )

                model_dfs[model_name] = deterministic_df_15
                model_onsets[model_name] = onset_da_dict_15

        return model_dfs, model_onsets
    else:
        for model_name, model_fp in prob_model_paths.items():
            print("=" * 80)
            print(f"Loading data from {model_name}")
            print("=" * 80)
            probabilistic_df_30, onset_da_dict_30 = (
                p.compute_metrics_multiple_years(
                    years=(
                        model_years[model_name]
                        if model_name in model_years.keys()
                        else model_years["Standard"]
                    ),
                    imd_folder=config["imd_folder"],
                    thres_file=config["thresh_file"],
                    model_forecast_dir=model_fp,
                    tolerance_days=5,
                    verification_window=16,
                    forecast_days=30,
                    max_forecast_day=30,
                    mok=True,
                    onset_window=5,
                    mok_month=6,
                    mok_day=2,
                )
            )
            model_dfs[model_name] = probabilistic_df_30
            model_onsets[model_name] = onset_da_dict_30

        for model_name, model_fp in det_model_paths.items():
            print("=" * 80)
            print(f"Loading data from {model_name}")
            print("=" * 80)
            deterministic_df_30, onset_da_dict_30 = (
                p.compute_metrics_multiple_years(
                    years=(
                        model_years[model_name]
                        if model_name in model_years.keys()
                        else model_years["Standard"]
                    ),
                    imd_folder=config["imd_folder"],
                    thres_file=config["thresh_file"],
                    model_forecast_dir=model_fp,
                    tolerance_days=5,
                    verification_window=16,
                    forecast_days=30,
                    max_forecast_day=30,
                    mok=True,
                    onset_window=5,
                    mok_month=6,
                    mok_day=2,
                )
            )

            model_dfs[model_name] = deterministic_df_30
            model_onsets[model_name] = onset_da_dict_30
        return model_dfs, model_onsets
    
def build_spatial_xarray_dict(config, model_dfs, model_onsets, clim_data):
    """Return dict of xr.Dataset for each model, fully standardized."""
    out={}

    for model_name, df in model_dfs.items():
        onset = model_onsets[model_name]

        if model_name in PROB_MODELS:
            ds = p.create_spatial_far_mr_mae(
                df, onset
            )
        else:
            ds = d.create_spatial_far_mr_mae(
                df, onset
                )
            
        ds["false_alarm_rate"] *= 100
        ds["miss_rate"] *= 100
        if isinstance(ds, dict):
            ds = xr.Dataset(ds)

        out[model_name] = ensure_lat_lon_sorted(ds)

    clim_data = clim_data.copy()
    clim_data["false_alarm_rate"] *= 100
    clim_data["miss_rate"] *= 100

    if isinstance(clim_data, dict):
            clim_data = xr.Dataset(clim_data)
    out["Climatology"] = ensure_lat_lon_sorted(clim_data)

    return {k: out[k] for k in model_str if k in out}


def format_data_for_spatial_fig(config,
                                tuple_of_model_data_dicts: tuple,
                                clim_data: xr.DataArray
                                ) -> dict:
    """Format and merge model and climatology metrics into a single ordered dict for plotting.

    Computes spatial FAR/MR/MAE metrics for each probabilistic and deterministic model,
    scales rates to percentages, appends the climatology baseline, and reorders all
    entries to match the paper's figure layout.

    Parameters
    ----------
    parent_dir : str
        Root directory used to resolve model path configurations.
    tuple_of_model_data_dicts : tuple
        Two-element tuple of (model_dfs, model_onsets) as returned by
        ``load_model_data_fig_7_through_9`` or ``load_model_data_fig_10_through_12``.
    clim_data : xr.DataArray
        Climatology spatial metrics DataArray as returned by the corresponding
        ``load_clim_data_*`` function.

    Returns:
    -------
    dict
        Ordered dictionary mapping model name to spatial metrics xr.Dataset,
        with keys in paper figure order:
        ['Climatology', 'IFS', 'AIFS', 'FuXi', 'Graphcast', 'GenCast', 'FuXi-S2S', 'NGCM'].
    """

    def reorder_dict(dict) -> dict:
        """Reorders dict to match paper format."""
        order = [
                "Climatology",
                "IFS",
                "AIFS",
                "FuXi",
                "Graphcast",
                "GenCast",
                "FuXi-S2S",
                "NGCM",
            ]
        reordered_dict = {key: dict[key] for key in order}
        return reordered_dict
    
    prob_model_paths = {
        "FuXi-S2S": config["model_paths"]["FuXi-S2S"],  # FuXi_S2S model
        "NGCM": config["model_paths"]["NGCM"],  # NGCM model
        "IFS": config["model_paths"]["IFS"],  # AIFS model
        "GenCast": config["model_paths"]["GenCast"],  # GenCast model
    }

    det_model_paths = {
            "AIFS": config["model_paths"]["AIFS"],
            "FuXi": config["model_paths"]["FuXi"],
            "Graphcast": config["model_paths"]["Graphcast"],  # Graphcast model
        }
    prob_plot_data = {}

    model_dfs = tuple_of_model_data_dicts[0]
    model_onsets = tuple_of_model_data_dicts[1]

    for model_name in prob_model_paths.keys():
        probabilistic_df = model_dfs[model_name]
        onset_da_dict = model_onsets[model_name]
        plot_probabilistic_metrics = p.create_spatial_far_mr_mae(
            probabilistic_df, onset_da_dict
        )
        plot_probabilistic_metrics["false_alarm_rate"] = (
            plot_probabilistic_metrics["false_alarm_rate"].round(3) * 100
        )
        plot_probabilistic_metrics["miss_rate"] = (
            plot_probabilistic_metrics["miss_rate"].round(3) * 100
        )
        prob_plot_data[model_name] = plot_probabilistic_metrics

    for model_name in det_model_paths.keys():
        deterministic_df = model_dfs[model_name]
        onset_da_dict = model_onsets[model_name]
        plot_deterministic_metrics = d.create_spatial_far_mr_mae(
            deterministic_df, onset_da_dict
        )
        plot_deterministic_metrics["false_alarm_rate"] = (
            plot_deterministic_metrics["false_alarm_rate"].round(3) * 100
        )
        plot_deterministic_metrics["miss_rate"] = (
            plot_deterministic_metrics["miss_rate"].round(3) * 100
        )

        prob_plot_data[model_name] = plot_deterministic_metrics

    clim_data["false_alarm_rate"] = clim_data["false_alarm_rate"].round(3) * 100
    clim_data["miss_rate"] = clim_data["miss_rate"].round(3) * 100

    prob_plot_data["Climatology"] = clim_data

    prob_plot_data = reorder_dict(prob_plot_data)

    return prob_plot_data

def create_formatted_df_for_plot(spatial_figs_15: dict,
                                 spatial_figs_30: dict):
    def make_df_long(df_short, metric):
        df_long = (
        df_short
        .stack()                # collapse columns into rows
        .reset_index()          # turn index into columns
        .rename(columns={
            "level_0": "lat",
            "level_1": "lon",
            0: metric
        })
    )
        return df_long

    spatial_dfs_final = []
    for key, value in spatial_figs_15.items():
        mae_df = value["mean_mae"].to_pandas()
        mae_df_long = make_df_long(mae_df, "Mean MAE")
        mae_df_long["model"] = key
        mae_df_long["horizon"] = 15

        mr_df = value["miss_rate"].to_pandas()
        mr_df_long = make_df_long(mr_df, "Miss Rate")
        mr_df_long["model"] = key
        mr_df_long["horizon"] = 15


        far_df = value["false_alarm_rate"].to_pandas()
        far_df_long = make_df_long(far_df, "False Alarm Rate")
        far_df_long["model"] = key
        far_df_long["horizon"] = 15

        merged_df = (
        mae_df_long
        .merge(mr_df_long, on=["lat", "lon", "model", "horizon"], how="outer")
        .merge(far_df_long, on=["lat", "lon", "model", "horizon"], how="outer")
        )

        spatial_dfs_final.append(merged_df)

    for key, value in spatial_figs_30.items():
        mae_df = value["mean_mae"].to_pandas()
        mae_df_long = make_df_long(mae_df, "Mean MAE")
        mae_df_long["model"] = key
        mae_df_long["horizon"] = 30

        mr_df = value["miss_rate"].to_pandas()
        mr_df_long = make_df_long(mr_df, "Miss Rate")
        mr_df_long["model"] = key
        mr_df_long["horizon"] = 30


        far_df = value["false_alarm_rate"].to_pandas()
        far_df_long = make_df_long(far_df, "False Alarm Rate")
        far_df_long["model"] = key
        far_df_long["horizon"] = 30

        merged_df = (
        mae_df_long
        .merge(mr_df_long, on=["lat", "lon", "model", "horizon"], how="outer")
        .merge(far_df_long, on=["lat", "lon", "model", "horizon"], how="outer")
        )

        spatial_dfs_final.append(merged_df)

    ret_df = pd.concat(spatial_dfs_final)
    return ret_df


def reformat_csv_as_grids(df, metric):
    model_str = ['Climatology', 'IFS', 'AIFS', 'FuXi', 'Graphcast', 'GenCast', 'FuXi-S2S', 'NGCM']

    lons = np.arange(68,101,4)
    lats=np.arange(8,37,4)
    
    model_vals = df['model'].unique()

    lat_to_idx = {v: i for i, v in enumerate(lats)}
    lon_to_idx = {v: i for i, v in enumerate(lons)}

    # output dict
    data_dict = {}

    for model in model_vals:
        sub = df[df['model'] == model]

        grid = np.full((len(lats), len(lons)), np.nan)

        for _, row in sub.iterrows():
            i = lat_to_idx[row['lat']]
            j = lon_to_idx[row['lon']]
            grid[i, j] = row[metric]

        data_dict[model] = grid

    ordered_dict = {m: data_dict[m] for m in model_str if m in data_dict}
    ordered_dict["None"] = np.full((len(lats), len(lons)), np.nan)

    return ordered_dict


def get_grid_mae_far_mr(df):
    mae_avg = reformat_csv_as_grids(df=df, metric="Mean MAE")
    mae_avg = np.stack([val for key, val in mae_avg.items()], axis=0)

    far = reformat_csv_as_grids(df=df, metric="False Alarm Rate")
    far = np.stack([val for key, val in far.items()], axis=0)


    mr = reformat_csv_as_grids(df=df, metric="Miss Rate")
    mr = np.stack([val for key, val in mr.items()], axis=0)

    return mae_avg, far, mr


def create_map_panel_colored_stats(ax, data, lon, lat, model_idx, model_name, 
                                #   mae_cmz_mean, std_er, far_cmz_mean, mr_cmz_mean,
                                  data_type='MAE', vmin=0, vmax=15, cmap='YlOrRd', n_colors=6, 
                                  show_ylabel=True, show_xlabel=True, title=None,
                                  shpfile_path=None
                                  ):
    """
    Create a map panel with colored statistics text for MAE, FAR, and MR
    """
    
    # Create meshgrid for plotting
    lon_edges = np.concatenate([lon - (lon[1]-lon[0])/2, [lon[-1] + (lon[1]-lon[0])/2]])
    lat_edges = np.concatenate([lat - (lat[1]-lat[0])/2, [lat[-1] + (lat[1]-lat[0])/2]])
    LON_edges, LAT_edges = np.meshgrid(lon_edges, lat_edges)
    
    # Plot the main data
    # plt_ar = data[:, :, model_idx]
    # cmap = plt.cm.get_cmap(cmap, n_colors)
    # masked_data = np.ma.masked_invalid(plt_ar.T)
    if isinstance(data, xr.DataArray):
        plt_ar = data.values   # already (lat, lon)
    else:
        data = np.asarray(data)
        if data.ndim == 3:
            plt_ar = data[model_idx, :, :]
        else:
            plt_ar = data
    cmap = plt.cm.get_cmap(cmap, n_colors)
    masked_data = np.ma.masked_invalid(plt_ar)
    
    # Use pcolormesh for proper grid cell alignment
    im = ax.pcolormesh(LON_edges, LAT_edges, masked_data, 
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    
    # Add India map outline
    india_boundaries = get_india_outline(shp_file_path=shpfile_path)
    for boundary in india_boundaries:
        india_lon, india_lat = boundary
        ax.plot(india_lon, india_lat, color='black', linewidth=map_lw)

    polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
    polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
    
    # Add polygon for Core Monsoon Zone
    polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                     fill=False, edgecolor='black', linewidth=polygon_lw)
    ax.add_patch(polygon)
    
    # Add model name text in top-right
    ax.text(0.95, 0.95, model_name, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            color='black', fontsize=MEDIUM_SIZE, fontweight='normal')

    # NO GRID VALUES - Remove the grid value text completely
    
    # Set axis limits and ticks
    ax.set_xlim([lon[0]-4, 100])
    ax.set_ylim([lat[0]-4, lat[-1]+4])
    
    # Create tick labels
    yticks = np.arange(lat[0]-2, lat[-1]+3, 8)
    yticklabels = [f"{int(y)}°N" if i % 1 == 0 else "" for i, y in enumerate(yticks)]
    ax.set_yticks(yticks)
    if show_ylabel:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels([])
    
    xticks = np.arange(lon[0]-2, lon[-1]+3, 8)
    xticklabels = [f"{int(x)}°E" if i % 1 == 0 else "" for i, x in enumerate(xticks)]
    ax.set_xticks(xticks)
    if show_xlabel:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels([])
        
    polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
    polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
    inside_mask, inside_lons, inside_lats = points_inside_polygon(
        polygon1_lon, polygon1_lat, lon, lat
        )
    
    polygon_data = np.where(inside_mask, plt_ar, np.nan)
    cmz_mean = np.nanmean(polygon_data)

    if data_type == 'MAE':
        metric_text = f'MAE: {cmz_mean:.1f}'
        text_color = 'black'
    elif data_type == 'FAR':
        metric_text = f'FAR: {cmz_mean:.1f}%'
        text_color = 'black'    
    elif data_type == 'MR':
        metric_text = f'MR: {cmz_mean:.1f}%'
        text_color = 'black'

    ax.text(0.96, 0.04, metric_text, transform=ax.transAxes,
            color=text_color, verticalalignment='bottom', 
            horizontalalignment='right', fontweight='normal', fontsize=MEDIUM_SIZE)
    
    # Remove grid lines
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params('both', length=tick_length, width=tick_width, which='major')
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.tick_params(axis='y', which='minor', left=False, right=False)

    if title:
        ax.text(0.02, 1.02, title, transform=ax.transAxes, 
                verticalalignment='bottom', fontsize=LARGE_SIZE, fontweight='normal')

    return im


def create_8_panel_figure(data, lat, lon,
                         data_type='MAE', vmin=0, vmax=15, cmap='YlOrRd', n_colors=10,
                         shpfile_path=None):
    """
    Create an 8-panel figure showing all models in a 4x2 grid with colorbar covering rows 2-3
    data_type: 'MAE', 'FAR', or 'MR' for title and filename
    """
    
    # Create the main figure - adjusted height for 4 rows
    fig = plt.figure(figsize=(6, 9), dpi=300)
    
    # Create GridSpec for 4 rows, 2 columns, plus space for colorbar
    gs = GridSpec(
        4, 3, figure=fig,
        hspace=0.05, wspace=-0.2,  # Reduced wspace from 0.05 to 0.02
        left=0.05, right=0.85, top=0.95, bottom=0.05,
        width_ratios=[1, 1, 0.08]  # Last column for colorbar
    )

    # Create axes for each model (4x2 grid)
    axes = []
    for row in range(4):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
    
    print(f"Creating {data_type} maps for all 8 models in 4x2 layout...")
    
    # Plot each model
    images = []
    for i, (ax, model_name) in enumerate(zip(axes, model_str)):
        print(f"Creating panel {i+1}/8: {model_name}")
        
        # Determine which labels to show
        show_ylabel = (i % 2 == 0)  # Show y-labels only for left column
        show_xlabel = (i >= 6)      # Show x-labels only for bottom row (row 4)
        
        # Create the map panel with the appropriate colormap
        im = create_map_panel_colored_stats(
            ax, data, lon, lat, i, model_name,
            data_type=data_type, vmin=vmin, vmax=vmax, cmap=cmap, n_colors=n_colors,
            show_ylabel=show_ylabel, show_xlabel=show_xlabel,
            shpfile_path=shpfile_path
        )
        images.append(im)
        
        # Style the axis
        ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, 
                      length=tick_length, width=tick_width)
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_linewidth(panel_linewidth)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
    
    # Create colorbar spanning rows 2 and 3 (indices 1 and 2)
    # Get positions of row 2 and row 3 panels
    row2_ax = axes[2]  # First panel in row 2 (index 2)
    row3_ax = axes[5]  # Second panel in row 3 (index 5)
    
    row2_pos = row2_ax.get_position()
    row3_pos = row3_ax.get_position()
    
    # Calculate colorbar position spanning rows 2-3
    colorbar_height = row2_pos.y1 - row3_pos.y0
    colorbar_y_start = row3_pos.y0
    
    cax = fig.add_axes([
        0.87,               # x position (right side)
        colorbar_y_start,   # y position (start of row 3)
        0.025,              # width
        colorbar_height     # height (spanning rows 2-3)
    ])
    
    # Set colorbar properties based on data type
    if data_type == 'MAE':
        cbar = fig.colorbar(images[0], cax=cax, orientation='vertical', extend='max')
        cbar.set_ticks(np.arange(0, vmax+1, 3))
        cbar.set_label('MAE (days)')
    elif data_type == 'FAR':
        cbar = fig.colorbar(images[0], cax=cax, orientation='vertical', extend='max')
        cbar.set_ticks(np.arange(0, vmax+1, 12))
        cbar.set_label('False alarm rate (%)')
    elif data_type == 'MR':
        cbar = fig.colorbar(images[0], cax=cax, orientation='vertical')
        cbar.set_ticks(np.arange(0, vmax+1, 20))
        cbar.set_label('Miss rate (%)')

    cbar.ax.minorticks_off()
    cbar.ax.tick_params(length=2, width=1)
    
    plt.tight_layout()
    
    # Save figure
    # filename = f'outputs/{data_type.lower()}_1_15day_2019_2024.pdf'
    # filename2 = f'outputs/fig_{data_type.lower()}_1_15day_2019_2024.png'
    # plt.savefig(filename, dpi=600, bbox_inches='tight')
    # plt.savefig(filename2, dpi=600, bbox_inches='tight')
    # print(f"Figure saved as {filename}")
    
    return fig



def create_8_panel_figure_xarray(
    spatial_dict,
    metric,
    data_type='MAE',
    vmin=0,
    vmax=15,
    cmap='YlOrRd',
    n_colors=10,
    shpfile_path=None
):
    """
    spatial_dict: dict[str, xr.Dataset]
    metric: str (e.g. 'mean_mae', 'false_alarm_rate', 'miss_rate')
    """

    model_str = list(spatial_dict.keys())
    data_arrays = [spatial_dict[m][metric] for m in model_str]

    fig = plt.figure(figsize=(6, 9), dpi=300)

    gs = GridSpec(
        4, 3, figure=fig,
        hspace=0.05, wspace=-0.2,
        left=0.05, right=0.85, top=0.95, bottom=0.05,
        width_ratios=[1, 1, 0.08]
    )

    axes = []
    for row in range(4):
        for col in range(2):
            axes.append(fig.add_subplot(gs[row, col]))

    print(f"Creating {data_type} maps for all 8 models in 4x2 layout...")

    images = []

    for i, (ax, model_name, da) in enumerate(zip(axes, model_str, data_arrays)):

        print(f"Creating panel {i+1}/8: {model_name}")

        # ---- EXACT SAME LOGIC YOU WANTED ----
        show_ylabel = (i % 2 == 0)
        show_xlabel = (i >= 6)

        # ---- extract coords from xarray ----
        lon = da.lon.values
        lat = da.lat.values

        # ---- call YOUR ORIGINAL plotting function ----
        im = create_map_panel_colored_stats(
            ax,
            da,   # <-- key change: xarray → numpy (keeps your function intact)
            lon,
            lat,
            i,
            model_name,
            data_type=data_type,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            n_colors=n_colors,
            show_ylabel=show_ylabel,
            show_xlabel=show_xlabel,
            shpfile_path=shpfile_path
        )

        images.append(im)

        # ---- styling (UNCHANGED) ----
        ax.tick_params(axis='both', which='major',
                       labelsize=SMALL_SIZE,
                       length=tick_length,
                       width=tick_width)

        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_linewidth(panel_linewidth)

        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)

    # ---- colorbar (UNCHANGED) ----
    row2_pos = axes[2].get_position()
    row3_pos = axes[5].get_position()

    cax = fig.add_axes([
        0.87,
        row3_pos.y0,
        0.025,
        row2_pos.y1 - row3_pos.y0
    ])

    cbar = fig.colorbar(images[0], cax=cax, orientation='vertical', extend='max')

    if data_type == 'MAE':
        cbar.set_label('MAE (days)')
        cbar.set_ticks(np.arange(0, vmax + 1, 3))

    elif data_type == 'FAR':
        cbar.set_label('False alarm rate (%)')
        cbar.set_ticks(np.arange(0, vmax + 1, 12))

    elif data_type == 'MR':
        cbar.set_label('Miss rate (%)')
        cbar.set_ticks(np.arange(0, vmax + 1, 20))

    cbar.ax.minorticks_off()
    cbar.ax.tick_params(length=2, width=1)

    plt.tight_layout()

    return fig


def make_spatial_figs(mae_avg, far, mr, lon, lat,
                      shpfile_path=None):
    print("Creating 8-panel figures...")
        
    # MAE Figure - use YlOrRd colormap
    mae_fig = create_8_panel_figure(
        mae_avg, lat, lon,
        # mae_cmz, std_er, far_cmz, mr_cmz,
        data_type='MAE', vmin=0, vmax=15, cmap='YlOrRd', n_colors=10,
        shpfile_path=shpfile_path
    )
    
    # FAR Figure - use Blues colormap and multiply by 100
    far_fig = create_8_panel_figure(
        far, lat, lon,
        # far_cmz, std_er, far_cmz, mr_cmz,
        data_type='FAR', vmin=0, vmax=60, cmap='Blues', n_colors=10,
        shpfile_path=shpfile_path
    )
    
    # MR Figure - use Blues colormap and multiply by 100
    mr_fig = create_8_panel_figure(
        mr, lat, lon,
        # mr_cmz, std_er, far_cmz, mr_cmz,
        data_type='MR', vmin=0, vmax=100, cmap='Blues', n_colors=10,
        shpfile_path=shpfile_path
    )
    
    plt.show()
    print("All 8-panel figures completed successfully!")
    print("8-panel figure functions created successfully!")

    return mae_fig, far_fig, mr_fig


def generate_fig_7_8_9(config):
    """15-day spatial figures (MAE, FAR, MR)."""

    # Only need 15-day climatology
    spatial_clim_15 = get_spatial_fig_clim_data(config, n=15)

    # Only load 15-day model data
    model_dfs_15, model_onsets_15 = get_spatial_fig_model_data(config, n=15)

    spatial_dict_15 = build_spatial_xarray_dict(config=config,
                              model_dfs=model_dfs_15,
                              model_onsets=model_onsets_15,
                              clim_data=spatial_clim_15
                              )
    
    mae_panel = [spatial_dict_15[m]["mean_mae"] for m in spatial_dict_15]
    far_panel = [spatial_dict_15[m]["false_alarm_rate"] for m in spatial_dict_15]
    miss_panel = [spatial_dict_15[m]["miss_rate"] for m in spatial_dict_15]
        
    lat = next(iter(spatial_dict_15.values())).lat.values
    lon = next(iter(spatial_dict_15.values())).lon.values

    mae_fig = create_8_panel_figure_xarray(spatial_dict_15,
                                 "mean_mae",
                                 data_type="MAE",
                                 vmin=0, vmax=15,
                                 cmap="YlOrRd",
                                 n_colors=10,
                                 shpfile_path=config["shpfile_path"],
                                 )
    far_fig = create_8_panel_figure_xarray(spatial_dict_15,
                                 "false_alarm_rate",
                                 data_type="FAR",
                                 vmin=0, vmax=60, cmap='Blues',
                                 n_colors=10,
                                 shpfile_path=config["shpfile_path"],
                                 )
    mr_fig = create_8_panel_figure_xarray(spatial_dict_15,
                                 "miss_rate",
                                 data_type="MR",
                                 vmin=0, vmax=100, cmap='Blues',
                                 n_colors=10,
                                 shpfile_path=config["shpfile_path"],
                                 )
    plt.show()

    return (mae_fig, far_fig, mr_fig), {
        "mae": mae_panel,
        "far": far_panel,
        "mr": miss_panel,
        "lat": lat,
        "lon": lon
    }

    


    

    # # Format into plottable spatial dict with 15-day climatology
    # spatial_figs_15 = format_data_for_spatial_fig(
    #     config,
    #     (model_dfs_15, model_onsets_15),
    #     spatial_clim_15,
    # )

    # # Pass empty dict for 30-day — create_formatted_df_for_plot handles it gracefully
    # combined_df = create_formatted_df_for_plot(spatial_figs_15, {})

    # mae_avg, far, mr = get_grid_mae_far_mr(combined_df)

    # ref_model = next(iter(spatial_figs_15))
    # lat = spatial_figs_15[ref_model]["mean_mae"].lat.values
    # lon = spatial_figs_15[ref_model]["mean_mae"].lon.values

    # mae_fig, far_fig, mr_fig = make_spatial_figs(mae_avg, far, mr, lon, lat, shpfile_path=config["shpfile_path"])
    # gridded_data = {"mae": mae_avg, "far": far, "mr": mr, "lat": lat, "lon": lon}
    # return (mae_fig, far_fig, mr_fig), gridded_data
    # return None


def generate_fig_10_11_12(config):
    """30-day spatial figures (MAE, FAR, MR)."""

    # Only need 30-day climatology
    spatial_clim_30 = get_spatial_fig_clim_data(config, n=30)

    # Load 30-day model data — this is the key difference from figs 7-9
    model_dfs_30, model_onsets_30 = get_spatial_fig_model_data(config, n=30)

    spatial_dict_30 = build_spatial_xarray_dict(config=config,
                              model_dfs=model_dfs_30,
                              model_onsets=model_onsets_30,
                              clim_data=spatial_clim_30
                              )
    
    mae_panel = [spatial_dict_30[m]["mean_mae"] for m in spatial_dict_30]
    far_panel = [spatial_dict_30[m]["false_alarm_rate"] for m in spatial_dict_30]
    miss_panel = [spatial_dict_30[m]["miss_rate"] for m in spatial_dict_30]
        
    lat = next(iter(spatial_dict_30.values())).lat.values
    lon = next(iter(spatial_dict_30.values())).lon.values

    mae_fig = create_8_panel_figure_xarray(spatial_dict_30,
                                 "mean_mae",
                                 data_type="MAE",
                                 vmin=0, vmax=15,
                                 cmap="YlOrRd",
                                 n_colors=10,
                                 shpfile_path=config["shpfile_path"],
                                 )
    far_fig = create_8_panel_figure_xarray(spatial_dict_30,
                                 "false_alarm_rate",
                                 data_type="FAR",
                                 vmin=0, vmax=60, cmap='Blues',
                                 n_colors=10,
                                 shpfile_path=config["shpfile_path"],
                                 )
    mr_fig = create_8_panel_figure_xarray(spatial_dict_30,
                                 "miss_rate",
                                 data_type="MR",
                                 vmin=0, vmax=100, cmap='Blues',
                                 n_colors=10,
                                 shpfile_path=config["shpfile_path"],
                                 )
    plt.show()

    return (mae_fig, far_fig, mr_fig), {
        "mae": mae_panel,
        "far": far_panel,
        "mr": miss_panel,
        "lat": lat,
        "lon": lon
    }

    # # Format into plottable spatial dict with 30-day climatology
    # spatial_figs_30 = format_data_for_spatial_fig(
    #     config,
    #     (model_dfs_30, model_onsets_30),
    #     spatial_clim_30,
    # )

    # # Pass empty dict for 15-day
    # combined_df = create_formatted_df_for_plot({}, spatial_figs_30)

    # mae_avg, far, mr = get_grid_mae_far_mr(combined_df)

    # ref_model = next(iter(spatial_figs_30))
    # lat = spatial_figs_30[ref_model]["mean_mae"].lat.values
    # lon = spatial_figs_30[ref_model]["mean_mae"].lon.values

    # mae_fig, far_fig, mr_fig = make_spatial_figs(mae_avg, far, mr, lon, lat, shpfile_path=config["shpfile_path"])
    # gridded_data = {"mae": mae_avg, "far": far, "mr": mr, "lat": lat, "lon": lon}
    # return (mae_fig, far_fig, mr_fig), gridded_data



# def compare_gridded_data(paper_orig, recreated_orig):
#     diff = {}
#     for key, value in paper_orig.items():
        
