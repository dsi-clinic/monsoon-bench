"""Spatial visualization for onset metrics.

This module provides functions for creating spatial maps of onset metrics
including MAE, False Alarm Rate, and Miss Rate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath

from monsoonbench.spatial.regions import (
    detect_resolution,
    get_cmz_polygon_coords,
    get_india_outline,
)

# Visualization constants
MAE_DARK_THRESHOLD = 7.5  # Values above this use white text
FAR_MR_DARK_THRESHOLD = 50  # Values above this use white text


def calculate_cmz_averages(
    data_array: xr.DataArray,
    lons: np.ndarray,
    lats: np.ndarray,
    polygon_lon: np.ndarray,
    polygon_lat: np.ndarray,
) -> float:
    """Calculate spatial average within the CMZ polygon.

    Args:
        data_array: Xarray DataArray with spatial data
        lons: Longitude coordinates
        lats: Latitude coordinates
        polygon_lon: CMZ polygon longitude coordinates
        polygon_lat: CMZ polygon latitude coordinates

    Returns:
        Mean value within CMZ polygon, or NaN if no valid data
    """
    polygon_path = MplPath(list(zip(polygon_lon, polygon_lat)))

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    inside_polygon = polygon_path.contains_points(
        points, radius=-.01 #Negative radius to include border points
    ).reshape(lon_grid.shape)

    values_inside = data_array.to_numpy()[inside_polygon]

    if len(values_inside) > 0:
        return float(np.nanmean(values_inside))
    return np.nan


def calculate_mae_stats_across_years(
    spatial_metrics: dict[str, xr.DataArray],
    lons: np.ndarray,
    lats: np.ndarray,
    polygon_lon: np.ndarray | None,
    polygon_lat: np.ndarray | None,
    polygon_defined: bool,
) -> tuple[float, float, float, float]:
    """Calculate MAE statistics: spatial average for each year, then mean ± SE across years.

    Args:
        spatial_metrics: Dictionary of spatial metric DataArrays
        lons: Longitude coordinates
        lats: Latitude coordinates
        polygon_lon: CMZ polygon longitude coordinates (or None)
        polygon_lat: CMZ polygon latitude coordinates (or None)
        polygon_defined: Whether CMZ polygon is available for this resolution

    Returns:
        Tuple of (cmz_mean, cmz_se, overall_mean, overall_se)
    """
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

        if polygon_defined and polygon_lon is not None and polygon_lat is not None:
            cmz_avg = calculate_cmz_averages(
                year_mae_map, lons, lats, polygon_lon, polygon_lat
            )
            if not np.isnan(cmz_avg):
                cmz_yearly_averages.append(cmz_avg)

        overall_avg = np.nanmean(year_mae_map.to_numpy())
        if not np.isnan(overall_avg):
            overall_yearly_averages.append(overall_avg)

    if len(cmz_yearly_averages) > 0 and polygon_defined:
        cmz_mean = np.mean(cmz_yearly_averages)
        cmz_se = (
            np.std(cmz_yearly_averages, ddof=1) / np.sqrt(len(cmz_yearly_averages))
            if len(cmz_yearly_averages) > 1
            else 0.0
        )
    else:
        cmz_mean, cmz_se = np.nan, np.nan

    if len(overall_yearly_averages) > 0:
        overall_mean = np.mean(overall_yearly_averages)
        overall_se = (
            np.std(overall_yearly_averages, ddof=1)
            / np.sqrt(len(overall_yearly_averages))
            if len(overall_yearly_averages) > 1
            else 0.0
        )
    else:
        overall_mean, overall_se = np.nan, np.nan

    return cmz_mean, cmz_se, overall_mean, overall_se


def plot_spatial_metrics(
    spatial_metrics: dict[str, xr.DataArray],
    shpfile_path: str | Path,
    figsize: tuple[float, float] = (18, 6),
    save_path: str | Path | None = None,
) -> tuple:
    """Plot spatial maps of Mean MAE, False Alarm Rate, and Miss Rate.

    Creates a 1x3 subplot with India outline, CMZ polygon (if available),
    grid values displayed, and CMZ averages.

    Args:
        spatial_metrics: Dictionary containing DataArrays for:
            - 'mean_mae': Mean absolute error
            - 'false_alarm_rate': False alarm rate (0-1)
            - 'miss_rate': Miss rate (0-1)
            - 'mae_YYYY': Individual year MAE values
        shpfile_path: Path to India shapefile for boundary outline
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure

    Returns:
        Tuple of (figure, axes) from matplotlib

    Example:
        >>> results = metrics.create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        >>> fig, axes = plot_spatial_metrics(results, "india.shp", save_path="output.png")
    """
    # Extract data
    mean_mae = spatial_metrics["mean_mae"]
    far = spatial_metrics["false_alarm_rate"] * 100  # Convert to percentage
    miss_rate = spatial_metrics["miss_rate"] * 100  # Convert to percentage

    # Get coordinates
    lats = mean_mae.lat.to_numpy()
    lons = mean_mae.lon.to_numpy()

    # Detect resolution from latitude spacing
    lat_diff = detect_resolution(lats)
    print(f"Detected resolution: {lat_diff:.1f} degrees")

    # Get CMZ polygon coordinates for this resolution
    cmz_coords = get_cmz_polygon_coords(lat_diff)
    polygon_defined = cmz_coords is not None

    if polygon_defined:
        polygon1_lon, polygon1_lat = cmz_coords
        print(f"Using {lat_diff:.1f}-degree CMZ polygon coordinates")
    else:
        polygon1_lon, polygon1_lat = None, None
        print(
            f"Resolution {lat_diff:.1f} degrees not supported for CMZ polygon. "
            f"Plotting without polygon and CMZ averages."
        )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Calculate statistics
    if polygon_defined and polygon1_lon is not None and polygon1_lat is not None:
        cmz_mae_mean, cmz_mae_se, _overall_mae_mean, _overall_mae_se = (
            calculate_mae_stats_across_years(
                spatial_metrics, lons, lats, polygon1_lon, polygon1_lat, polygon_defined
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
            spatial_metrics["miss_rate"] * 100, lons, lats, polygon1_lon, polygon1_lat
        )
    else:
        cmz_mae_mean, cmz_mae_se, _overall_mae_mean, _overall_mae_se = (
            calculate_mae_stats_across_years(
                spatial_metrics, lons, lats, None, None, False
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
    lon_edges_grid, lat_edges_grid = np.meshgrid(lon_edges, lat_edges)

    # Plot parameters
    map_lw = 0.75
    polygon_lw = 1.25
    panel_linewidth = 0.5
    tick_length = 3
    tick_width = 0.8

    # Text size based on resolution
    txt_fsize_map = {2.0: 8, 4.0: 10, 1.0: 6}
    txt_fsize = txt_fsize_map.get(round(lat_diff, 1), 8)

    # Get India boundaries
    india_boundaries = get_india_outline(shpfile_path)

    # Panel 1: Mean MAE
    masked_mae = np.ma.masked_invalid(mean_mae.to_numpy())
    _im1 = axes[0].pcolormesh(
        lon_edges_grid,
        lat_edges_grid,
        masked_mae,
        cmap="OrRd",
        vmin=0,
        vmax=15,
        shading="flat",
    )

    for boundary in india_boundaries:
        india_lon, india_lat = boundary
        axes[0].plot(india_lon, india_lat, color="black", linewidth=map_lw)

    if polygon_defined and polygon1_lon is not None and polygon1_lat is not None:
        polygon = Polygon(
            list(zip(polygon1_lon, polygon1_lat)),
            fill=False,
            edgecolor="black",
            linewidth=polygon_lw,
        )
        axes[0].add_patch(polygon)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            value = mean_mae.to_numpy()[i, j]
            if not np.isnan(value):
                text_color = "white" if value > MAE_DARK_THRESHOLD else "black"
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
    masked_far = np.ma.masked_invalid(far.to_numpy())
    _im2 = axes[1].pcolormesh(
        lon_edges_grid,
        lat_edges_grid,
        masked_far,
        cmap="Reds",
        vmin=0,
        vmax=100,
        shading="flat",
    )

    for boundary in india_boundaries:
        india_lon, india_lat = boundary
        axes[1].plot(india_lon, india_lat, color="black", linewidth=map_lw)

    if polygon_defined and polygon1_lon is not None and polygon1_lat is not None:
        polygon = Polygon(
            list(zip(polygon1_lon, polygon1_lat)),
            fill=False,
            edgecolor="black",
            linewidth=polygon_lw,
        )
        axes[1].add_patch(polygon)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            value = far.to_numpy()[i, j]
            if not np.isnan(value):
                text_color = "white" if value > FAR_MR_DARK_THRESHOLD else "black"
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
    masked_mr = np.ma.masked_invalid(miss_rate.to_numpy())
    _im3 = axes[2].pcolormesh(
        lon_edges_grid,
        lat_edges_grid,
        masked_mr,
        cmap="Blues",
        vmin=0,
        vmax=100,
        shading="flat",
    )

    for boundary in india_boundaries:
        india_lon, india_lat = boundary
        axes[2].plot(india_lon, india_lat, color="black", linewidth=map_lw)

    if polygon_defined and polygon1_lon is not None and polygon1_lat is not None:
        polygon = Polygon(
            list(zip(polygon1_lon, polygon1_lat)),
            fill=False,
            edgecolor="black",
            linewidth=polygon_lw,
        )
        axes[2].add_patch(polygon)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            value = miss_rate.to_numpy()[i, j]
            if not np.isnan(value):
                text_color = "white" if value > FAR_MR_DARK_THRESHOLD else "black"
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

    # Print CMZ averages if polygon is defined
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


def datetime_to_day(dt: np.datetime64) -> int:
    """
    Convert a numpy.datetime64 object to day-of-year index (Jan 1 = 0).
    """
    # Ensure day precision
    dt = dt.astype('datetime64[D]')
    
    # Get start of year
    start_of_year = dt.astype('datetime64[Y]').astype('datetime64[D]')
    
    # Compute difference in days
    day_index = (dt - start_of_year).astype(int)
    
    return int(day_index)

def day_to_daystr(day: int, year: int) -> str:
    """
    Convert day-of-year index (Jan 1 = 0) into string like 'Jan 1'.
    Works on Windows, Linux, macOS.
    """
    start = np.datetime64(f"{year}-01-01")
    date = start + np.timedelta64(day, "D")

    py_date = date.astype("datetime64[D]").astype(object)

    return f"{py_date.strftime('%b')} {py_date.day}"

def compare_onset_map(clim_onset, model_onset, model_name, year, province_num, rainfall_path, shpfile_path):
    """Generates a 2x3 grid of rainfall data maps.
    Row 1: Climatological onset date, +5 days, +10 days
    Row 2: Model predicted onset date, +5 days, +10 days
    """

    # --- Extract data ---
    non_zero_coords = []
    for i in range(clim_onset.shape[0]):
        for j in range(clim_onset.shape[1]):
            if not np.isnan(clim_onset.values[i, j]):
                non_zero_coords.append((i, j))

    model_onset = model_onset[model_name][year]

    i, j = non_zero_coords[province_num]
    clim_onset_date = int(clim_onset.values[i, j])
    model_onset_date = datetime_to_day(model_onset.values[i, j])

    rain_data = xr.open_dataset(f"{rainfall_path}/4p0/{year}.nc")
    lats = rain_data["lat"].values
    lons = rain_data["lon"].values

    # Define Core Monsoon Zone bounding polygon
    polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
    polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])

    # Create cell-edge arrays for pcolormesh
    lon_edges = np.concatenate([lons - (lons[1] - lons[0]) / 2, [lons[-1] + (lons[1] - lons[0]) / 2]])
    lat_edges = np.concatenate([lats - (lats[1] - lats[0]) / 2, [lats[-1] + (lats[1] - lats[0]) / 2]])
    LON_edges, LAT_edges = np.meshgrid(lon_edges, lat_edges)

    # Style parameters
    map_lw = 0.75
    polygon_lw = 1.25
    panel_linewidth = 0.5
    tick_length = 3
    tick_width = 0.8
    txt_fsize = 10
    vmin, vmax = 0, 15

    india_boundaries = get_india_outline(shpfile_path)

    offsets = [0, 5, 10]

    # --- Helper function ---
    def plot_rainfall_panel(ax, rainfall, lons, lats, title, show_ylabel=False, highlight_coord=None):
        """Plot a single rainfall panel with India outline, CMZ polygon, and annotations."""
        masked = np.ma.masked_invalid(rainfall)
        pcm = ax.pcolormesh(LON_edges, LAT_edges, masked, cmap="OrRd", vmin=vmin, vmax=vmax, shading="flat")

        # India outline
        for boundary in india_boundaries:
            blon, blat = boundary
            ax.plot(blon, blat, color="black", linewidth=map_lw)

        # CMZ polygon
        polygon = Polygon(
            list(zip(polygon1_lon, polygon1_lat)),
            fill=False, edgecolor="black", linewidth=polygon_lw,
        )
        ax.add_patch(polygon)

        # Highlight the selected province cell
        if highlight_coord is not None:
            hi, hj = highlight_coord
            cell_lon = lons[hj]
            cell_lat = lats[hi]
            dx = lons[1] - lons[0]
            dy = lats[1] - lats[0]
            rect = plt.Rectangle(
                (cell_lon - dx / 2, cell_lat - dy / 2), dx, dy,
                linewidth=2.5, edgecolor="blue", facecolor="none", zorder=5
            )
            ax.add_patch(rect)

        # Value annotations
        for ii, lat in enumerate(lats):
            for jj, lon in enumerate(lons):
                value = rainfall[ii, jj]
                if not np.isnan(value):
                    text_color = "white" if value > (vmax - vmin) / 2 else "black"
                    ax.text(lon, lat, f"{value:.1f}", ha="center", va="center",
                            color=text_color, fontsize=txt_fsize, fontweight="normal")

        # Axis styling
        ax.set_xlim([lons.min() - 2, lons.max() + 2])
        ax.set_ylim([lats.min() - 2, lats.max() + 2])

        xticks = np.arange(lons.min(), lons.max() + 1, 8)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(x)}°E" for x in xticks])
        ax.set_xlabel("Longitude", fontsize=12)

        if show_ylabel:
            yticks = np.arange(lats.min(), lats.max() + 1, 4)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{int(y)}°N" for y in yticks])
            ax.set_ylabel("Latitude", fontsize=12)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.tick_params(axis="both", which="major", labelsize=10, length=tick_length, width=tick_width)
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_linewidth(panel_linewidth)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_title(title, fontsize=12)

        return pcm

    # --- Create 2x3 figure ---
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Row 1: Climatological onset
    for col, offset in enumerate(offsets):
        day_idx = int(clim_onset_date + offset)
        rainfall = rain_data["RAINFALL"].values[day_idx]
        title = f"Climatological ({day_to_daystr(day_idx, year)})"
        pcm_clim = plot_rainfall_panel(axes[0, col], rainfall, lons, lats, title,
                                    show_ylabel=(col == 0), highlight_coord=(i, j))

    # Row 2: Model onset
    for col, offset in enumerate(offsets):
        day_idx = int(model_onset_date + offset)
        rainfall = rain_data["RAINFALL"].values[day_idx]
        title = f"{model_name} Model ({day_to_daystr(day_idx, year)})" if offset else f"{model_name} Model (Day {day_to_daystr(day_idx, year)})"
        pcm_model = plot_rainfall_panel(axes[1, col], rainfall, lons, lats, title,
                                    show_ylabel=(col == 0), highlight_coord=(i, j))

    # One colorbar per row
    cbar1 = fig.colorbar(pcm_clim, ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
    cbar1.set_label("Rainfall (mm)", fontsize=12)
    cbar1.ax.tick_params(labelsize=9)

    cbar2 = fig.colorbar(pcm_model, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02)
    cbar2.set_label("Rainfall (mm)", fontsize=12)
    cbar2.ax.tick_params(labelsize=9)

    plt.suptitle(f"Rainfall Comparison at Predicted Onset Date — Province {province_num}, Year {year}", fontsize=15, y=1.01)
    plt.show()
    return fig, axes
