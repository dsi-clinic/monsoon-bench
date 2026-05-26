"""Utility functions for generating Figures 7-12 of the monsoon benchmark paper."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from plot_config import LARGE_SIZE, MEDIUM_SIZE, SMALL_SIZE, params

from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    OnsetMetricsBase,
    ProbabilisticOnsetMetrics,
)
from monsoonbench.spatial.regions import get_india_outline, points_inside_polygon

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
def load_spatial_dict(save_path, model_names=model_str) -> dict:
    """Load merged NetCDF and split back into per-model dict."""
    merged = xr.open_dataset(save_path)
    return {
        name: merged.sel(model=name).drop_vars("model")
        for name in model_names
        if name in merged.model.values
    }


def ensure_lat_lon_sorted(ds: xr.Dataset) -> xr.Dataset:
    """Ensure consistent lat/lon ordering and alignment."""
    return ds.sortby(["lat", "lon"])


def get_spatial_fig_clim_data(config, n=15) -> xr.DataArray:
    """Function for loading spatial climatological data for figures 7-12"""
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

def get_spatial_fig_model_data(config, n=15) -> tuple:
    """Function for loading spatial model data for figures 7-12"""
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
                    date_filter_year=2024 if model_name != "FuXi-S2S" else 2022
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
                        date_filter_year=2024 if model_name != "FuXi-S2S" else 2022
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
                    date_filter_year=2024 if model_name != "FuXi-S2S" else 2022
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
                    date_filter_year=2024 if model_name != "FuXi-S2S" else 2022
                )
            )

            model_dfs[model_name] = deterministic_df_30
            model_onsets[model_name] = onset_da_dict_30
        return model_dfs, model_onsets
    
def build_spatial_xarray_dict(config, model_dfs, model_onsets, clim_data) -> dict:
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


def create_map_panel_colored_stats(ax, data, lon, lat, model_idx, model_name, 
                                #   mae_cmz_mean, std_er, far_cmz_mean, mr_cmz_mean,
                                  data_type="MAE", vmin=0, vmax=15, cmap="YlOrRd", n_colors=6, 
                                  show_ylabel=True, show_xlabel=True, title=None,
                                  shpfile_path=None
                                  ) -> None:
    """Create a map panel with colored statistics text for MAE, FAR, and MR"""
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
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    
    # Add India map outline
    india_boundaries = get_india_outline(shp_file_path=shpfile_path)
    for boundary in india_boundaries:
        india_lon, india_lat = boundary
        ax.plot(india_lon, india_lat, color="black", linewidth=map_lw)

    polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
    polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
    
    # Add polygon for Core Monsoon Zone
    polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                     fill=False, edgecolor="black", linewidth=polygon_lw)
    ax.add_patch(polygon)
    
    # Add model name text in top-right
    ax.text(0.95, 0.95, model_name, transform=ax.transAxes,
            horizontalalignment="right", verticalalignment="top",
            color="black", fontsize=MEDIUM_SIZE, fontweight="normal")

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

    if data_type == "MAE":
        metric_text = f"MAE: {cmz_mean:.1f}"
        text_color = "black"
    elif data_type == "FAR":
        metric_text = f"FAR: {cmz_mean:.1f}%"
        text_color = "black"    
    elif data_type == "MR":
        metric_text = f"MR: {cmz_mean:.1f}%"
        text_color = "black"

    ax.text(0.96, 0.04, metric_text, transform=ax.transAxes,
            color=text_color, verticalalignment="bottom", 
            horizontalalignment="right", fontweight="normal", fontsize=MEDIUM_SIZE)
    
    # Remove grid lines
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params("both", length=tick_length, width=tick_width, which="major")
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    if title:
        ax.text(0.02, 1.02, title, transform=ax.transAxes, 
                verticalalignment="bottom", fontsize=LARGE_SIZE, fontweight="normal")

    return im


def create_8_panel_figure(data, lat, lon,
                         data_type="MAE", vmin=0, vmax=15, cmap="YlOrRd", n_colors=10,
                         shpfile_path=None) -> plt.Figure:
    """Create an 8-panel figure showing all models in a 4x2 grid with colorbar covering rows 2-3

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
        ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE, 
                      length=tick_length, width=tick_width)
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_linewidth(panel_linewidth)
        ax.set_aspect("equal", adjustable="box")
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
    if data_type == "MAE":
        cbar = fig.colorbar(images[0], cax=cax, orientation="vertical", extend="max")
        cbar.set_ticks(np.arange(0, vmax+1, 3))
        cbar.set_label("MAE (days)")
    elif data_type == "FAR":
        cbar = fig.colorbar(images[0], cax=cax, orientation="vertical", extend="max")
        cbar.set_ticks(np.arange(0, vmax+1, 12))
        cbar.set_label("False alarm rate (%)")
    elif data_type == "MR":
        cbar = fig.colorbar(images[0], cax=cax, orientation="vertical")
        cbar.set_ticks(np.arange(0, vmax+1, 20))
        cbar.set_label("Miss rate (%)")

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
    data_type="MAE",
    vmin=0,
    vmax=15,
    cmap="YlOrRd",
    n_colors=10,
    shpfile_path=None
) -> plt.Figure:
    """spatial_dict: dict[str, xr.Dataset]

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
        ax.tick_params(axis="both", which="major",
                       labelsize=SMALL_SIZE,
                       length=tick_length,
                       width=tick_width)

        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_linewidth(panel_linewidth)

        ax.set_aspect("equal", adjustable="box")
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

    cbar = fig.colorbar(images[0], cax=cax, orientation="vertical", extend="max")

    if data_type == "MAE":
        cbar.set_label("MAE (days)")
        cbar.set_ticks(np.arange(0, vmax + 1, 3))

    elif data_type == "FAR":
        cbar.set_label("False alarm rate (%)")
        cbar.set_ticks(np.arange(0, vmax + 1, 12))

    elif data_type == "MR":
        cbar.set_label("Miss rate (%)")
        cbar.set_ticks(np.arange(0, vmax + 1, 20))

    cbar.ax.minorticks_off()
    cbar.ax.tick_params(length=2, width=1)

    plt.tight_layout()

    return fig


def generate_fig_7_9_11(config) -> tuple:
    """15-day spatial figures (MAE, FAR, MR)."""
    save_path = config["output_dir"] + "/spatial_scores_15_day_2019_2024.nc"

    try:
        spatial_dict_15 = load_spatial_dict(save_path, model_str)

    except FileNotFoundError:
        spatial_clim_15 = get_spatial_fig_clim_data(config, n=15)
        model_dfs_15, model_onsets_15 = get_spatial_fig_model_data(config, n=15)
        spatial_dict_15 = build_spatial_xarray_dict(
            config=config,
            model_dfs=model_dfs_15,
            model_onsets=model_onsets_15,
            clim_data=spatial_clim_15,
        )
        print(f"Saving loaded data to {save_path}")
        tagged = [ds.expand_dims(model=[name]) for name, ds in spatial_dict_15.items()]
        merged = xr.merge(tagged)
        merged.to_netcdf(save_path)
        # save_spatial_dict(spatial_dict_15, config["output_dir"], "spatial_scores_15_day_2019_2024")


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
                                 vmin=0, vmax=60, cmap="Blues",
                                 n_colors=10,
                                 shpfile_path=config["shpfile_path"],
                                 )
    mr_fig = create_8_panel_figure_xarray(spatial_dict_15,
                                 "miss_rate",
                                 data_type="MR",
                                 vmin=0, vmax=100, cmap="Blues",
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

    

def generate_fig_8_10_12(config) -> tuple:
    """30-day spatial figures (MAE, FAR, MR)."""
    save_path = config["output_dir"] + "/spatial_scores_30_day_2019_2024.nc"

    try:
        spatial_dict_30 = load_spatial_dict(save_path, model_str)

    except FileNotFoundError:
        # Only need 30-day climatology
        spatial_clim_30 = get_spatial_fig_clim_data(config, n=30)

        # Load 30-day model data — this is the key difference from figs 7-9
        model_dfs_30, model_onsets_30 = get_spatial_fig_model_data(config, n=30)

        spatial_dict_30 = build_spatial_xarray_dict(config=config,
                                model_dfs=model_dfs_30,
                                model_onsets=model_onsets_30,
                                clim_data=spatial_clim_30
                                )
        print(f"Saving loaded data to {save_path}")
        tagged = [ds.expand_dims(model=[name]) for name, ds in spatial_dict_30.items()]
        merged = xr.merge(tagged)
        merged.to_netcdf(save_path)
        # save_spatial_dict(spatial_dict_30, config["output_dir"], "spatial_scores_30_day_2019_2024")

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
                                 vmin=0, vmax=60, cmap="Blues",
                                 n_colors=10,
                                 shpfile_path=config["shpfile_path"],
                                 )
    mr_fig = create_8_panel_figure_xarray(spatial_dict_30,
                                 "miss_rate",
                                 data_type="MR",
                                 vmin=0, vmax=100, cmap="Blues",
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

