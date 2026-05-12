from monsoonbench.metrics import (
    ProbabilisticOnsetMetrics,
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    OnsetMetricsBase
)

from data_utils import *

import xarray as xr
import numpy as np


import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import matplotlib.colors as colors

import warnings
from monsoonbench.spatial.regions import get_india_outline
warnings.filterwarnings('ignore')
from plot_config import (
    params,
    SMALL_SIZE,
    MEDIUM_SIZE,
    LARGE_SIZE
    )

# Apply plot settings
plt.rcParams.update(params)

c = ClimatologyOnsetMetrics()
p = ProbabilisticOnsetMetrics()
d = DeterministicOnsetMetrics()
o = OnsetMetricsBase()

model_str_camel = ['Climatology', 'IFS', 'AIFS', 'FuXi', 'Graphcast', 'GenCast', 'FuXi-S2S', 'NGCM']


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


PROB_MODELS = {"FuXi-S2S", "NGCM", "IFS", "GenCast"}


def ensure_lat_lon_sorted(ds: xr.Dataset) -> xr.Dataset:
    """Ensure consistent lat/lon ordering and alignment."""
    return ds.sortby(["lat", "lon"])


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

    return {k: out[k] for k in model_str_camel if k in out}


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