"""Utility functions for paper figures 1 & 4."""

from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import xarray as xr

from examples.paper_figures.plot_config import (
    params, SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE
)

from examples.paper_figures.utils.data_utils import (
    YEAR_RANGES, save_data
)

from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    ProbabilisticOnsetMetrics,
)
from monsoonbench.visualization import (
    create_model_comparison_table,
)

# Apply plot settings
plt.rcParams.update(params)

DEFAULT_WINDOW_BINS: list[tuple[int, int]] = [
    (1, 6),
    (6, 11),
    (11, 16),
    (16, 21),
    (21, 26),
    (26, 31),
]

def get_clim_window_data(
    config: dict[str, str],
):
    c_metrics = ClimatologyOnsetMetrics()

    clim_data = []

    for (lower, upper) in DEFAULT_WINDOW_BINS:
        if lower < 11:
            tol_days = 2
        elif lower < 21:
            tol_days = 3
        else:
            tol_days = 5
        
        multi_yr_metrics, multi_onset_dy = c_metrics.compute_climatology_baseline_multiple_years(        
            years=config["years"],
            imd_folder=config["imd_folder"],
            thres_file=config["thres_file"],
            tolerance_days=tol_days, #Tolerance window
            verification_window=lower, 
            forecast_days=upper,
            max_forecast_day=upper,
            mok=True,
            onset_window=5,
            mok_month=6,
            mok_day=2,
        )
        clim_plot_data = c_metrics.create_spatial_far_mr_mae(
            multi_yr_metrics, dict.fromkeys(config["years"], multi_onset_dy)
        )

        clim_data.append(clim_plot_data)
        
    return clim_data

def get_model_window_data(
    model_paths: dict[str, str],
    year_ranges: dict[str, list[int]],
    config: dict[str, str],
    lower_window: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, xr.DataArray]]:
    """Get model dataframes and onset data arrays for a given set of model paths, year ranges, and forecast period.

    Args:
        model_paths: Dictionary of model names and their file paths.
        year_ranges: Dictionary of model names and their year ranges.
        config: Dictionary of configuration parameters.
        lower_window: Lower bound of the verification window.
    """

    upper_window = lower_window + 5
    metrics = ProbabilisticOnsetMetrics()
    d_metrics = DeterministicOnsetMetrics()

    model_dfs = {}
    model_onsets = {}

    # Validate verification window and tolerance days
    if lower_window < 11:
        tol_days = 2
    elif lower_window < 21:
        tol_days = 3
    else:
        tol_days = 5

    # Compute metrics for each model
    for model_name, model_fp in model_paths.items():
        if model_name.lower() == "fuxi-s2s":
            date_filter_year=2022
        else:
            date_filter_year=2024
        try:
            probabilistic_df, onset_da_dict = metrics.compute_metrics_multiple_years(
                years=year_ranges[model_name],
                imd_folder=config["imd_folder"],
                thres_file=config["thres_file"],
                model_forecast_dir=model_fp,
                tolerance_days=tol_days, #tolerance for metric calculations
                verification_window=lower_window, #start of window
                forecast_days=upper_window, #lower_window + 5
                max_forecast_day=upper_window, #lower_window + 5
                mok=True,
                onset_window=5,
                mok_month=6,
                mok_day=2,
                date_filter_year=date_filter_year,
            )

        except Exception:
            probabilistic_df, onset_da_dict = d_metrics.compute_metrics_multiple_years(
                years=year_ranges[model_name],
                imd_folder=config["imd_folder"],
                thres_file=config["thres_file"],
                model_forecast_dir=model_fp,
                tolerance_days=tol_days,
                verification_window=lower_window,
                forecast_days=upper_window,
                max_forecast_day=upper_window,
                mok=True,
                onset_window=5,
                mok_month=6,
                mok_day=2,
                date_filter_year=date_filter_year,
            )

        model_dfs[model_name] = probabilistic_df
        model_onsets[model_name] = onset_da_dict

    return model_dfs, model_onsets

def load_fig3_data(
    config: dict[str, str],
    model_paths: dict[str, str],
):
    c_metrics = ClimatologyOnsetMetrics()
    clim_data = get_clim_window_data(config)
    clim_window_data = {}

    for i, (lower, upper) in enumerate(DEFAULT_WINDOW_BINS):
        clim_window_data[f"{lower}-{upper}"] = clim_data[i]


    clim_df = create_model_comparison_table(clim_window_data)

    windows = [1,6,11,16,21,26]
    window_dfs = []

    for start_date in windows:
        f3_df, f3_onsets = get_model_window_data(model_paths, YEAR_RANGES, config, lower_window=start_date)
        window_data = {}
        for model_name in model_paths.keys():
            plot_probabilistic_metrics = c_metrics.create_spatial_far_mr_mae(
                f3_df[model_name], f3_onsets[model_name]
            )
            window_data[model_name] = plot_probabilistic_metrics

        window_df = create_model_comparison_table(window_data)
        window_dfs.append(window_df)

    far = []
    mae = []
    std_er = []
    mr = []
    for i in range(len(window_dfs)):
        far_win = [clim_df.iloc[i]["cmz_far_pct"]]
        mae_win = [clim_df.iloc[i]["cmz_mae_mean_days"]]
        std_er_win = [clim_df.iloc[i]["cmz_mae_se_days"]]
        mr_win = [clim_df.iloc[i]["cmz_mr_pct"]]

        far_win += window_dfs[i]["cmz_far_pct"].values.tolist()
        mae_win += window_dfs[i]["cmz_mae_mean_days"].values.tolist()
        std_er_win += window_dfs[i]["cmz_mae_se_days"].values.tolist()
        mr_win += window_dfs[i]["cmz_mr_pct"].values.tolist()

        far.append(np.array(far_win))
        mae.append(np.array(mae_win))
        std_er.append(np.array(std_er_win))
        mr.append(np.array(mr_win))

    out_dict = {
        "far_cmz": np.array(far),
        "mae_cmz": np.array(mae),
        "model_str": np.array(["clim", "ifs", "aifs", "fuxi", "graphcast",
                    "gencast", "fuxis2s", "ngcm51"]),
        "std_er": np.array(std_er),
        "mr_cmz": np.array(mr)
    }

    save_data(out_dict, config["output_dir"], "5day_forecastwindow_cmz_2019_2024")

# ================================================
# Plotting Functions
# ================================================

# Load data from MAT file
def load_weekly_data(data_dir: str):
    """Load weekly deterministic scores from MAT file"""
    try:
        # Update this path to your actual MAT file location
        weekly_file = f"{data_dir}/5day_forecastwindow_cmz_2019_2024.mat"
        data = sio.loadmat(weekly_file)
        
        # Extract data - adjust variable names based on your MAT file structure
        mae_cmz = data['mae_cmz']  # Shape should be (6, 8) for 6 time periods, 8 models
        far_cmz = data['far_cmz']  # Shape should be (6, 8)
        mr_cmz = data['mr_cmz']    # Shape should be (4, 8) for 4 weeks
        std_er = data['std_er']    # Standard errors for MAE
        
        return mae_cmz, far_cmz, mr_cmz, std_er
    
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return mae_cmz, far_cmz, mr_cmz, std_er


def create_climatology_difference_heatmap(data_dir: str):
    """Create a red-blue heatmap showing difference from climatology for each model"""
    
    panel_width = 0.5
    tick_length = 2.5
    tick_width = 1
    cbar_aspect = 16
    cb_pad = 0.14  # Padding for colorbars

    # Load data
    mae_cmz, far_cmz, mr_cmz, std_er = load_weekly_data(data_dir)
    
    # Model names (excluding climatology since we're comparing against it)
    model_names = ['IFS*', 'AIFS$^{\dagger}$', 'FuXi', 'GraphCast', 
                'GenCast', 'FuXi-S2S*', 'NGCM']
    
    # Week labels
    week_labels = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30']
    
    # Calculate differences from climatology (climatology is index 0)
    # Positive values mean worse than climatology, negative means better
    mae_diff = mae_cmz[:, 1:] - mae_cmz[:, 0:1]  # Only weeks 1-4, exclude climatology
    far_diff = far_cmz[:, 1:] - far_cmz[:, 0:1]  # Only weeks 1-4, exclude climatology
    # For MR, reverse the sign so negative = worse (lower MR is worse)
    mr_diff = (mr_cmz[:, 1:] - mr_cmz[:, 0:1])  # Reverse sign for MR
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    
    # Common colormap settings - using RdBu where blue=negative/better, red=positive/worse
    vmin_mae = -4  # Adjust based on your data range
    vmax_mae = 4
    vmin_far = -20
    vmax_far = 20
    vmin_mr = -60
    vmax_mr = 60

    mae_boundaries = np.arange(-4,4.5,0.5)
    mae_norm = BoundaryNorm(mae_boundaries, plt.cm.RdBu_r.N)
    far_boundaries = np.arange(-18,19,3)
    far_norm = BoundaryNorm(far_boundaries, plt.cm.RdBu_r.N)
    mr_boundaries = np.arange(-60,61,10)
    mr_norm = BoundaryNorm(mr_boundaries, plt.cm.RdBu_r.N)

    # MAE difference heatmap
    im1 = ax1.imshow(mae_diff.T, cmap='RdBu_r', aspect='auto', 
                    norm=mae_norm, interpolation='nearest')
    ax1.set_xticks(range(len(week_labels)))
    ax1.set_xticklabels(week_labels, fontsize=LARGE_SIZE)
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names, fontsize=LARGE_SIZE)
    # Make y-ticks go outside the panel
    ax1.tick_params(axis='y', direction='out', length=tick_length, width=tick_width)
    ax1.tick_params(axis='y', right=False)
    ax1.tick_params(axis='x', top=False)
    ax1.set_title('$\Delta$MAE (days)', fontsize=MEDIUM_SIZE, fontweight='normal')
    
    # Add text annotations for MAE
    for i in range(len(model_names)):
        for j in range(len(week_labels)):
            value = mae_diff[j, i]
            text_color = 'white' if abs(value) > 2.5 else 'black'
            ax1.text(j, i, f'{value:.1f}', ha="center", va="center", 
                    color=text_color, fontsize=SMALL_SIZE, fontweight='normal')
    
    # Add horizontal colorbar for MAE (bottom, 75% width, closer to panel)
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', 
                        pad=cb_pad, shrink=0.75, aspect=cbar_aspect, boundaries=mae_boundaries)
    #cbar1.set_label('Days (Blue = Better, Red = Worse)', fontsize=SMALL_SIZE)
    cbar1.ax.tick_params(labelsize=SMALL_SIZE)
    cbar1.ax.tick_params(length=tick_length,width = tick_width)
    cbar1.minorticks_off()
    # FAR difference heatmap
    im2 = ax2.imshow(far_diff.T, cmap='RdBu_r', aspect='auto', 
                    norm=far_norm, interpolation='nearest')
    ax2.set_xticks(range(len(week_labels)))
    ax2.set_xticklabels(week_labels, fontsize=LARGE_SIZE)
    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels([])  # Remove y-tick labels for panel 2
    # Hide y-ticks for panel 2
    ax2.tick_params(axis='y', left=False, right=False)
    ax2.tick_params(axis='x', top=False)
    ax2.set_title('$\Delta$ FAR (\%)', fontsize=MEDIUM_SIZE, fontweight='normal')
    ax2.set_xlabel('Forecast window (days)', fontsize=MEDIUM_SIZE)
    # Add text annotations for FAR
    for i in range(len(model_names)):
        for j in range(len(week_labels)):
            value = far_diff[j, i]
            text_color = 'white' if abs(value) > 15 else 'black'
            ax2.text(j, i, f'{value:.1f}', ha="center", va="center", 
                    color=text_color, fontsize=SMALL_SIZE, fontweight='normal')
    
    # Add horizontal colorbar for FAR (bottom, 75% width, closer to panel)
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', 
                        pad=cb_pad, shrink=0.75, aspect=cbar_aspect, boundaries=far_boundaries)
    cbar2.set_label('(Blue = Better, Red = Worse)', fontsize=SMALL_SIZE)
    cbar2.ax.tick_params(labelsize=SMALL_SIZE)
    cbar2.ax.tick_params(length=tick_length,width = tick_width)
    cbar2.minorticks_off()
    # MR difference heatmap
    im3 = ax3.imshow(mr_diff.T, cmap='RdBu_r', aspect='auto', 
                    norm=mr_norm, interpolation='nearest')
    ax3.set_xticks(range(len(week_labels)))
    ax3.set_xticklabels(week_labels, fontsize=LARGE_SIZE)
    ax3.set_yticks(range(len(model_names)))
    ax3.set_yticklabels([])  # Remove y-tick labels for panel 3
    # Hide y-ticks for panel 3
    ax3.tick_params(axis='y', left=False, right=False)
    ax3.tick_params(axis='x', top=False)
    ax3.set_title('$\Delta$ MR (\%)', fontsize=MEDIUM_SIZE, fontweight='normal')
    
    # Add text annotations for MR
    for i in range(len(model_names)):
        for j in range(len(week_labels)):
            value = mr_diff[j, i]
            text_color = 'white' if abs(value) > 50 else 'black'
            ax3.text(j, i, f'{value:.1f}', ha="center", va="center", 
                    color=text_color, fontsize=SMALL_SIZE, fontweight='normal')
    
    # Add horizontal colorbar for MR (bottom, 75% width, closer to panel)
    cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', 
                        pad=cb_pad, shrink=0.75, aspect=cbar_aspect, boundaries=mr_boundaries)
    #cbar3.set_label('Percentage (Blue = Better, Red = Worse)', fontsize=SMALL_SIZE)
    cbar3.ax.tick_params(labelsize=SMALL_SIZE)
    cbar3.ax.tick_params(length=tick_length,width = tick_width)
    cbar3.minorticks_off()
    # Style all axes
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', which='major', labelsize=SMALL_SIZE,
                    length=2, width=0.5)
        # Set spine width
        for spine in ax.spines.values():
            spine.set_linewidth(panel_width)
    
    # Adjust layout to accommodate horizontal colorbars (less space needed now)
    plt.subplots_adjust(bottom=0.15, top=0.9, wspace=0.05)
    
    # Save figure
    plt.savefig('fig3.png', dpi=600, bbox_inches='tight')
    plt.savefig('fig3.pdf', dpi=600, bbox_inches='tight')

    return fig


def make_fig3(data_dir: str):
    # Create the heatmap figure
    heatmap_fig = create_climatology_difference_heatmap(data_dir)
    return heatmap_fig
    