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


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.collections import QuadMesh
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Rectangle
import matplotlib.colors as colors


from pathlib import Path

import warnings
from monsoonbench.spatial.regions import get_india_outline
from monsoonbench.spatial.regions import points_inside_polygon
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

lat_grid = np.arange(8, 37, 4)  # 8:4:36
lon_grid = np.arange(68, 101, 4)  # 68:4:100

panel_linewidth = 0.5
map_lw = 0.5
polygon_lw = 1
tick_length = 2
tick_width = 0.5


PROB_MODELS = {"FuXi-S2S", "NGCM", "IFS", "GenCast"}

brier_levels = np.array([-40, -30, -20, -10, 0, 10, 20, 30, 40])
rps_levels = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80])



def get_clim_brier(df):
    clim_brier = c.calculate_brier_score_climatology(df)
    return clim_brier


def get_clim_rps(df):
    clim_rps = p.calculate_rps(df)
    return clim_rps
                        

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


def create_discrete_colormap(levels, base_cmap='RdBu'):
    """Create a discrete colormap with specified levels"""
    cmap = plt.cm.get_cmap(base_cmap)
    norm = colors.BoundaryNorm(levels, cmap.N, clip=True)
    return cmap, norm


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
    
    return fig, data_arrays
