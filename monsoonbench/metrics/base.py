"""Common utilities for monsoon onset metrics calculations.

Contains shared functionality between deterministic and probabilistic models.
"""

from datetime import datetime
from pathlib import Path as FilePath

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Polygon
from matplotlib.path import Path


class OnsetMetricsBase:
    """Base class containing common functionality for onset metrics"""

    @staticmethod
    def load_imd_rainfall(year: int, imd_folder: str) -> xr.DataArray:
        """Load IMD daily rainfall NetCDF for a given year."""
        file_patterns = [f"data_{year}.nc", f"{year}.nc"]
        folder_path = FilePath(imd_folder)

        imd_file = None
        for pattern in file_patterns:
            test_path = folder_path / pattern
            if test_path.exists():
                imd_file = str(test_path)
                break

        if imd_file is None:
            available_files = [
                f.name for f in folder_path.iterdir() if f.suffix == ".nc"
            ]
            raise FileNotFoundError(
                f"No IMD file found for year {year} in {imd_folder}. "
                f"Tried patterns: {file_patterns}. "
                f"Available files: {available_files}"
            )

        print(f"Loading IMD rainfall from: {imd_file}")

        ds = xr.open_dataset(imd_file)
        rainfall = ds["RAINFALL"]

        # Standardize dimension names
        dim_mapping = {}
        # Check for latitude/longitude dimensions
        if "latitude" in rainfall.dims:
            dim_mapping["latitude"] = "lat"
        if "LATITUDE" in rainfall.dims:
            dim_mapping["LATITUDE"] = "lat"
        if "longitude" in rainfall.dims:
            dim_mapping["longitude"] = "lon"
        if "LONGITUDE" in rainfall.dims:
            dim_mapping["LONGITUDE"] = "lon"
        if "TIME" in rainfall.dims:
            dim_mapping["TIME"] = "time"

        if dim_mapping:
            rainfall = rainfall.rename(dim_mapping)
            print(f"Renamed dimensions: {dim_mapping}")

        return rainfall

    @staticmethod
    def detect_observed_onset(
        rainfall_ds: xr.DataArray,
        thresh_slice: xr.DataArray,
        year: int,
        mok: bool = True,
    ) -> xr.DataArray:
        """Detect observed onset dates for a given year."""
        rain_slice = rainfall_ds
        window = 5

        if mok:
            start_date = datetime(year, 6, 2)  # MOK date: June 2nd
            date_label = "MOK date (June 2nd)"
        else:
            start_date = datetime(year, 5, 1)  # May 1st
            date_label = "May 1st"

        time_dates = pd.to_datetime(rain_slice.time.values)
        start_idx_candidates = np.where(time_dates > start_date)[0]

        if len(start_idx_candidates) == 0:
            print(
                f"Warning: {date_label} ({start_date.strftime('%Y-%m-%d')}) not found in data for year {year}"
            )
            fallback_date = datetime(year, 4, 1)
            start_idx = np.where(time_dates >= fallback_date)[0][0]
            print("Using fallback date: April 1st")
        else:
            start_idx = start_idx_candidates[0]
            print(
                f"Using {date_label} ({start_date.strftime('%Y-%m-%d')}) as start date for onset detection"
            )

        rain_subset = rain_slice.isel(time=slice(start_idx, None))
        rolling_sum = rain_subset.rolling(
            time=window, min_periods=window, center=False
        ).sum()
        rolling_sum_aligned = rolling_sum.shift(time=-(window - 1))

        first_day_condition = rain_subset > 1
        sum_condition = rolling_sum_aligned > thresh_slice
        onset_condition = first_day_condition & sum_condition

        def find_first_true(arr: np.ndarray) -> int:
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

        valid_mask = onset_indices.values >= 0
        time_coords = rain_subset.time.values
        onset_dates_array = np.full(
            onset_indices.shape, np.datetime64("NaT"), dtype="datetime64[ns]"
        )

        for i in range(onset_indices.shape[0]):
            for j in range(onset_indices.shape[1]):
                if valid_mask[i, j]:
                    idx = int(onset_indices[i, j].values)
                    if 0 <= idx < len(time_coords):
                        onset_dates_array[i, j] = time_coords[idx]

        onset_da = xr.DataArray(
            onset_dates_array,
            coords=[("lat", rain_slice.lat.values), ("lon", rain_slice.lon.values)],
            name="onset_date",
        )

        return onset_da

    @staticmethod
    def get_india_outline(shp_file_path):
        """Get India outline coordinates from shapefile."""
        india_gdf = gpd.read_file(shp_file_path)

        boundaries = []
        for geom in india_gdf.geometry:
            if hasattr(geom, "exterior"):
                coords = list(geom.exterior.coords)
                lon_coords = [coord[0] for coord in coords]
                lat_coords = [coord[1] for coord in coords]
                boundaries.append((lon_coords, lat_coords))
            elif hasattr(geom, "geoms"):
                for sub_geom in geom.geoms:
                    if hasattr(sub_geom, "exterior"):
                        coords = list(sub_geom.exterior.coords)
                        lon_coords = [coord[0] for coord in coords]
                        lat_coords = [coord[1] for coord in coords]
                        boundaries.append((lon_coords, lat_coords))
        return boundaries

    @staticmethod
    def compute_onset_metrics_with_windows(
        onset_df, tolerance_days=3, verification_window=1, forecast_days=15
    ):
        """Compute contingency matrix metrics following MATLAB logic with forecast and validation windows."""
        print(f"Computing onset metrics with tolerance = {tolerance_days} days")
        print(
            f"Verification window starts {verification_window} days after initialization"
        )
        print(f"Forecast window length: {forecast_days} days")

        results_list = []
        unique_locations = onset_df[["lat", "lon"]].drop_duplicates()

        print(f"Processing {len(unique_locations)} unique grid points...")

        for idx, (_, row) in enumerate(unique_locations.iterrows()):
            lat, lon = row["lat"], row["lon"]

            if idx % 10 == 0:
                print(
                    f"Processing grid point {idx + 1}/{len(unique_locations)}: lat={lat:.2f}, lon={lon:.2f}"
                )

            grid_data = onset_df[
                (onset_df["lat"] == lat) & (onset_df["lon"] == lon)
            ].copy()

            grid_data["obs_onset_dt"] = pd.to_datetime(grid_data["obs_onset_date"])
            grid_data["model_onset_dt"] = pd.to_datetime(grid_data["onset_date"])
            grid_data["init_dt"] = pd.to_datetime(grid_data["init_time"])

            TP = 0
            FP = 0
            FN = 0
            TN = 0
            num_onset = 0
            num_no_onset = 0
            mae_tp = []
            mae_fp = []

            gt_grd = grid_data["obs_onset_dt"].iloc[0]

            for _, init_row in grid_data.iterrows():
                t_init = init_row["init_dt"]
                model_onset = init_row["model_onset_dt"]

                valid_window_start = t_init + pd.Timedelta(days=verification_window)
                valid_window_end = valid_window_start + pd.Timedelta(days=14)

                whole_forecast_window_start = t_init + pd.Timedelta(days=1)
                whole_forecast_window_end = t_init + pd.Timedelta(days=forecast_days)

                is_onset_in_whole_window = (
                    whole_forecast_window_start <= gt_grd <= whole_forecast_window_end
                )
                if is_onset_in_whole_window:
                    num_onset += 1
                else:
                    num_no_onset += 1

                has_model_onset = not pd.isna(model_onset)

                if has_model_onset:
                    is_model_in_valid_window = (
                        valid_window_start <= model_onset <= valid_window_end
                    )

                    if is_model_in_valid_window:
                        abs_diff_days = abs((model_onset - gt_grd).days)

                        if abs_diff_days <= tolerance_days:
                            TP += 1
                            mae_tp.append(abs_diff_days)
                        else:
                            FP += 1
                            mae_fp.append(abs_diff_days)

                else:
                    if is_onset_in_whole_window:
                        FN += 1
                    else:
                        TN += 1

            total_forecasts = len(grid_data)

            mae_combined = mae_tp + mae_fp
            mae = np.mean(mae_combined) if len(mae_combined) > 0 else np.nan
            mae_tp_only = np.mean(mae_tp) if len(mae_tp) > 0 else np.nan

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

        metrics_df = pd.DataFrame(results_list)

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
    def create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict):
        """Create spatial maps of False Alarm Rate, Miss Rate, yearly MAE, and mean MAE across years."""
        first_year = list(onset_da_dict.keys())[0]
        lats = onset_da_dict[first_year].lat.values
        lons = onset_da_dict[first_year].lon.values

        print("Creating spatial FAR, Miss Rate, yearly MAE, and mean MAE maps...")
        print(f"Grid dimensions: {len(lats)} lats x {len(lons)} lons")
        print(f"Years: {list(metrics_df_dict.keys())}")

        spatial_metrics = {}

        false_alarm_rate_map = np.full((len(lats), len(lons)), np.nan)
        miss_rate_map = np.full((len(lats), len(lons)), np.nan)
        mean_mae_map = np.full((len(lats), len(lons)), np.nan)

        yearly_mae_maps = {}
        for year in metrics_df_dict.keys():
            yearly_mae_maps[year] = np.full((len(lats), len(lons)), np.nan)

        for i, lat_val in enumerate(lats):
            for j, lon_val in enumerate(lons):
                total_FP = 0
                total_TN = 0
                total_FN = 0
                total_num_onset = 0

                mae_values = []
                has_any_valid_data = False

                for year, metrics_df in metrics_df_dict.items():
                    obs_onset_val = onset_da_dict[year].isel(lat=i, lon=j).values

                    if pd.isna(obs_onset_val):
                        continue

                    grid_data = metrics_df[
                        (metrics_df["lat"] == lat_val) & (metrics_df["lon"] == lon_val)
                    ]

                    if len(grid_data) > 0:
                        has_any_valid_data = True
                        row = grid_data.iloc[0]

                        total_FP += row["false_positive"]
                        total_TN += row["true_negative"]
                        total_FN += row["false_negative"]
                        total_num_onset += row["num_onset"]

                        mae_val = row["mae_combined"]
                        if not pd.isna(mae_val):
                            yearly_mae_maps[year][i, j] = mae_val
                            mae_values.append(mae_val)

                if has_any_valid_data:
                    if (total_FP + total_TN) > 0:
                        false_alarm_rate_map[i, j] = total_FP / (total_FP + total_TN)
                    else:
                        false_alarm_rate_map[i, j] = 0

                    if total_num_onset > 0:
                        miss_rate_map[i, j] = total_FN / total_num_onset
                    else:
                        miss_rate_map[i, j] = 0

                    if len(mae_values) > 0:
                        mean_mae_map[i, j] = np.mean(mae_values)

        spatial_metrics["false_alarm_rate"] = xr.DataArray(
            false_alarm_rate_map,
            coords=[("lat", lats), ("lon", lons)],
            name="false_alarm_rate",
            attrs={
                "description": "False Alarm Rate = sum(FP) / sum(FP + TN) across all valid years"
            },
        )

        spatial_metrics["miss_rate"] = xr.DataArray(
            miss_rate_map,
            coords=[("lat", lats), ("lon", lons)],
            name="miss_rate",
            attrs={
                "description": "Miss Rate = sum(FN) / sum(total_onsets) across all valid years"
            },
        )

        spatial_metrics["mean_mae"] = xr.DataArray(
            mean_mae_map,
            coords=[("lat", lats), ("lon", lons)],
            name="mean_mae",
            attrs={
                "description": "Mean MAE across all valid years (omitting NaN values)"
            },
        )

        for year, mae_map in yearly_mae_maps.items():
            spatial_metrics[f"mae_{year}"] = xr.DataArray(
                mae_map,
                coords=[("lat", lats), ("lon", lons)],
                name=f"mae_{year}",
                attrs={"description": f"Mean Absolute Error for year {year}"},
            )

        return spatial_metrics

    @staticmethod
    def plot_spatial_metrics(
        spatial_metrics, shpfile_path, figsize=(18, 6), save_path=None
    ):
        """Plot spatial maps of Mean MAE, False Alarm Rate, and Miss Rate.

        Creates a 1x3 subplot with India outline, CMZ polygon, grid values displayed, and CMZ averages.
        """
        # Extract data
        mean_mae = spatial_metrics["mean_mae"]
        far = spatial_metrics["false_alarm_rate"] * 100  # Convert to percentage
        miss_rate = spatial_metrics["miss_rate"] * 100  # Convert to percentage

        # Get coordinates
        lats = mean_mae.lat.values
        lons = mean_mae.lon.values

        # Detect resolution from latitude spacing
        lat_diff = abs(lats[1] - lats[0])
        print(f"Detected resolution: {lat_diff:.1f} degrees")

        # Define Core Monsoon Zone bounding polygon coordinates based on resolution
        polygon_defined = False
        if abs(lat_diff - 2.0) < 0.1:  # 2-degree resolution
            polygon1_lon = np.array(
                [83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83]
            )
            polygon1_lat = np.array(
                [17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17]
            )
            polygon_defined = True
            print("Using 2-degree CMZ polygon coordinates")
        elif abs(lat_diff - 4.0) < 0.1:  # 4-degree resolution
            polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
            polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
            polygon_defined = True
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
            polygon_defined = True
            print("Using 1-degree CMZ polygon coordinates")
        else:
            print(
                f"Resolution {lat_diff:.1f} degrees not supported for CMZ polygon. Plotting without polygon and CMZ averages."
            )
            polygon_defined = False

        def calculate_cmz_averages(data_array, lons, lats, polygon_lon, polygon_lat):
            """Calculate spatial average within the CMZ polygon"""
            polygon_path = Path(list(zip(polygon_lon, polygon_lat)))

            lon_grid, lat_grid = np.meshgrid(lons, lats)

            points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
            inside_polygon = polygon_path.contains_points(points).reshape(
                lon_grid.shape
            )

            values_inside = data_array.values[inside_polygon]

            if len(values_inside) > 0:
                return np.nanmean(values_inside)
            else:
                return np.nan

        def calculate_mae_stats_across_years(
            spatial_metrics, lons, lats, polygon_lon, polygon_lat
        ):
            """Calculate MAE statistics: spatial average for each year, then mean ± SE across years"""
            yearly_mae_keys = [
                key
                for key in spatial_metrics.keys()
                if key.startswith("mae_") and key != "mae_combined"
            ]

            if not yearly_mae_keys:
                print("Warning: No yearly MAE maps found")
                return np.nan, np.nan, np.nan, np.nan

            cmz_yearly_averages = []
            overall_yearly_averages = []

            for mae_key in yearly_mae_keys:
                year_mae_map = spatial_metrics[mae_key]

                if polygon_defined and polygon_lon is not None:
                    cmz_avg = calculate_cmz_averages(
                        year_mae_map, lons, lats, polygon_lon, polygon_lat
                    )
                    if not np.isnan(cmz_avg):
                        cmz_yearly_averages.append(cmz_avg)

                overall_avg = np.nanmean(year_mae_map.values)
                if not np.isnan(overall_avg):
                    overall_yearly_averages.append(overall_avg)

            if len(cmz_yearly_averages) > 0 and polygon_defined:
                cmz_mean = np.mean(cmz_yearly_averages)
                cmz_se = (
                    np.std(cmz_yearly_averages, ddof=1)
                    / np.sqrt(len(cmz_yearly_averages))
                    if len(cmz_yearly_averages) > 1
                    else 0
                )
            else:
                cmz_mean, cmz_se = np.nan, np.nan

            if len(overall_yearly_averages) > 0:
                overall_mean = np.mean(overall_yearly_averages)
                overall_se = (
                    np.std(overall_yearly_averages, ddof=1)
                    / np.sqrt(len(overall_yearly_averages))
                    if len(overall_yearly_averages) > 1
                    else 0
                )
            else:
                overall_mean, overall_se = np.nan, np.nan

            return cmz_mean, cmz_se, overall_mean, overall_se

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Calculate statistics (only calculate CMZ stats if polygon is defined)
        if polygon_defined:
            cmz_mae_mean, cmz_mae_se, overall_mae_mean, overall_mae_se = (
                calculate_mae_stats_across_years(
                    spatial_metrics, lons, lats, polygon1_lon, polygon1_lat
                )
            )

            cmz_far = calculate_cmz_averages(
                spatial_metrics["false_alarm_rate"] * 100,
                lons,
                lats,
                polygon1_lon,
                polygon1_lat,
            )
            cmz_mr = calculate_cmz_averages(
                spatial_metrics["miss_rate"] * 100,
                lons,
                lats,
                polygon1_lon,
                polygon1_lat,
            )
        else:
            cmz_mae_mean, cmz_mae_se, overall_mae_mean, overall_mae_se = (
                calculate_mae_stats_across_years(
                    spatial_metrics, lons, lats, None, None
                )
            )
            cmz_far = np.nan
            cmz_mr = np.nan

        # Create edges for pcolormesh (cell boundaries)
        lon_edges = np.concatenate(
            [lons - (lons[1] - lons[0]) / 2, [lons[-1] + (lons[1] - lons[0]) / 2]]
        )
        lat_edges = np.concatenate(
            [lats - (lats[1] - lats[0]) / 2, [lats[-1] + (lats[1] - lats[0]) / 2]]
        )
        LON_edges, LAT_edges = np.meshgrid(lon_edges, lat_edges)

        # Plot parameters
        map_lw = 0.75
        polygon_lw = 1.25
        panel_linewidth = 0.5
        tick_length = 3
        tick_width = 0.8
        if abs(lat_diff - 2.0) < 0.1:
            txt_fsize = 8
        elif abs(lat_diff - 4.0) < 0.1:
            txt_fsize = 10
        elif abs(lat_diff - 1.0) < 0.1:
            txt_fsize = 6
        else:
            txt_fsize = 8

        # Panel 1: Mean MAE
        masked_mae = np.ma.masked_invalid(mean_mae.values)
        axes[0].pcolormesh(
            LON_edges,
            LAT_edges,
            masked_mae,
            cmap="OrRd",
            vmin=0,
            vmax=15,
            shading="flat",
        )

        # Add India outline
        india_boundaries = OnsetMetricsBase.get_india_outline(shpfile_path)
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            axes[0].plot(india_lon, india_lat, color="black", linewidth=map_lw)

        # Add CMZ polygon only if defined
        if polygon_defined:
            polygon = Polygon(
                list(zip(polygon1_lon, polygon1_lat)),
                fill=False,
                edgecolor="black",
                linewidth=polygon_lw,
            )
            axes[0].add_patch(polygon)

        # Add text annotations for MAE values
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = mean_mae.values[i, j]
                if not np.isnan(value):
                    text_color = "white" if value > 7.5 else "black"
                    axes[0].text(
                        lon,
                        lat,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=txt_fsize,
                        fontweight="normal",
                    )

        # Add CMZ average text with mean ± SE across years (only if polygon is defined)
        if polygon_defined and not np.isnan(cmz_mae_mean):
            if cmz_mae_se > 0:
                cmz_text = f"MAE: {cmz_mae_mean:.1f}±{cmz_mae_se:.1f} days"
            else:
                cmz_text = f"MAE: {cmz_mae_mean:.1f} days"

            axes[0].text(
                0.98,
                0.02,
                cmz_text,
                transform=axes[0].transAxes,
                color="black",
                fontsize=14,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

        axes[0].text(
            0.98,
            0.98,
            "MAE (in days)",
            transform=axes[0].transAxes,
            color="black",
            fontsize=14,
            fontweight="normal",
            verticalalignment="top",
            horizontalalignment="right",
        )
        axes[0].set_xlabel("Longitude", fontsize=12)
        axes[0].set_ylabel("Latitude", fontsize=12)

        # Panel 2: False Alarm Rate
        masked_far = np.ma.masked_invalid(far.values)
        axes[1].pcolormesh(
            LON_edges,
            LAT_edges,
            masked_far,
            cmap="Reds",
            vmin=0,
            vmax=100,
            shading="flat",
        )

        # Add India outline
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            axes[1].plot(india_lon, india_lat, color="black", linewidth=map_lw)

        # Add CMZ polygon only if defined
        if polygon_defined:
            polygon = Polygon(
                list(zip(polygon1_lon, polygon1_lat)),
                fill=False,
                edgecolor="black",
                linewidth=polygon_lw,
            )
            axes[1].add_patch(polygon)

        # Add text annotations for FAR values
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = far.values[i, j]
                if not np.isnan(value):
                    text_color = "white" if value > 50 else "black"
                    axes[1].text(
                        lon,
                        lat,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=txt_fsize,
                        fontweight="normal",
                    )

        # Add CMZ average text (only if polygon is defined)
        if polygon_defined and not np.isnan(cmz_far):
            cmz_text = f"FAR: {cmz_far:.1f}%"
            axes[1].text(
                0.98,
                0.02,
                cmz_text,
                transform=axes[1].transAxes,
                color="black",
                fontsize=14,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

        axes[1].text(
            0.98,
            0.98,
            "False Alarm Rate (%)",
            transform=axes[1].transAxes,
            color="black",
            fontsize=14,
            fontweight="normal",
            verticalalignment="top",
            horizontalalignment="right",
        )
        axes[1].set_xlabel("Longitude", fontsize=12)

        # Panel 3: Miss Rate
        masked_mr = np.ma.masked_invalid(miss_rate.values)
        axes[2].pcolormesh(
            LON_edges,
            LAT_edges,
            masked_mr,
            cmap="Blues",
            vmin=0,
            vmax=100,
            shading="flat",
        )

        # Add India outline
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            axes[2].plot(india_lon, india_lat, color="black", linewidth=map_lw)

        # Add CMZ polygon only if defined
        if polygon_defined:
            polygon = Polygon(
                list(zip(polygon1_lon, polygon1_lat)),
                fill=False,
                edgecolor="black",
                linewidth=polygon_lw,
            )
            axes[2].add_patch(polygon)

        # Add text annotations for Miss Rate values
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = miss_rate.values[i, j]
                if not np.isnan(value):
                    text_color = "white" if value > 50 else "black"
                    axes[2].text(
                        lon,
                        lat,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=txt_fsize,
                        fontweight="normal",
                    )

        # Add CMZ average text (only if polygon is defined)
        if polygon_defined and not np.isnan(cmz_mr):
            cmz_text = f"MR: {cmz_mr:.1f}%"
            axes[2].text(
                0.98,
                0.02,
                cmz_text,
                transform=axes[2].transAxes,
                color="black",
                fontsize=14,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

        axes[2].text(
            0.98,
            0.98,
            "Miss Rate (%)",
            transform=axes[2].transAxes,
            color="black",
            fontsize=14,
            fontweight="normal",
            verticalalignment="top",
            horizontalalignment="right",
        )

        axes[2].set_xlabel("Longitude", fontsize=12)

        # Set consistent axis limits and styling for all panels
        for i, ax in enumerate(axes):
            ax.set_xlim([lons.min() - 2, lons.max() + 2])
            ax.set_ylim([lats.min() - 2, lats.max() + 2])

            xticks = np.arange(lons.min(), lons.max() + 1, 8)
            xticklabels = [f"{int(x)}°E" for x in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            if i == 0:
                yticks = np.arange(lats.min(), lats.max() + 1, 4)
                yticklabels = [f"{int(y)}°N" for y in yticks]
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
            else:
                ax.set_yticks([])
                ax.set_yticklabels([])

            ax.tick_params(
                axis="both",
                which="major",
                labelsize=10,
                length=tick_length,
                width=tick_width,
            )
            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_linewidth(panel_linewidth)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        # Only print CMZ averages if polygon is defined
        if polygon_defined:
            print("\n=== CORE MONSOON ZONE (CMZ) AVERAGES ===")

            if not np.isnan(cmz_mae_mean):
                print(
                    f"CMZ Mean MAE (avg across years): {cmz_mae_mean:.2f} ± {cmz_mae_se:.2f} days"
                )
            else:
                print("CMZ Mean MAE: N/A")

            print(f"CMZ False Alarm Rate: {cmz_far:.1f} %")
            print(f"CMZ Miss Rate: {cmz_mr:.1f} %")
        else:
            print(
                f"\nNote: CMZ averages not calculated (resolution {lat_diff:.1f}° not supported)"
            )

        return fig, axes
