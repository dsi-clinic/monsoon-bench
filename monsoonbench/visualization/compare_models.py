"""Model comparison utilities for monsoon onset metrics.

This module provides:

1. create_model_comparison_table
   Build a tidy pandas.DataFrame summarizing CMZ MAE, FAR, and Miss Rate
   for multiple models.

2. plot_model_comparison_dual_axis
   Create a grouped bar chart with a dual y-axis:
   - Left axis: CMZ MAE (days)
   - Right axis: CMZ FAR (%) and CMZ Miss Rate (%)

3. compare_models
   Convenience function to create the table and dual-axis plot in one call.

Expected spatial_metrics format (same as CLI/plot_spatial_metrics):
    - "mean_mae": xr.DataArray (days)
    - "false_alarm_rate": xr.DataArray in [0, 1]
    - "miss_rate": xr.DataArray in [0, 1]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os
import seaborn as sns

from monsoonbench.spatial.regions import (
    detect_resolution,
    get_cmz_polygon_coords,
)
from monsoonbench.visualization.spatial import (
    calculate_cmz_averages,
    calculate_mae_stats_across_years,
)

__all__ = [
    "create_model_comparison_table",
    "plot_model_comparison_dual_axis",
    "compare_models",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_spatial_metrics_keys(
    model_name: str, spatial_metrics: Mapping[str, xr.DataArray]
) -> None:
    """Validate that required keys exist in spatial_metrics for a model."""
    required = ["mean_mae", "false_alarm_rate", "miss_rate"]
    missing = [key for key in required if key not in spatial_metrics]
    if missing:
        raise KeyError(
            f"Missing required keys for model '{model_name}': {missing}. "
            "Expected at least 'mean_mae', 'false_alarm_rate', and 'miss_rate'.",
        )


def _global_nanmean(arr: np.ndarray) -> float:
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _summarize_single_model(
    spatial_metrics: Mapping[str, xr.DataArray],
) -> dict[str, float]:
    """Create CMZ summary stats for a single model.

    Returns:
        - cmz_mae_mean_days: CMZ mean MAE (days)
        - cmz_mae_se_days:   CMZ MAE standard error (days)
        - cmz_far_pct:       CMZ mean FAR (%)
        - cmz_mr_pct:        CMZ mean Miss Rate (%)
        - overall_mae_mean_days: (optional) domain-wide mean MAE
        - overall_far_pct:       domain-wide mean FAR
        - overall_mr_pct:        domain-wide mean MR
    """
    mean_mae = spatial_metrics["mean_mae"]
    far = spatial_metrics["false_alarm_rate"] * 100.0  # %
    mr = spatial_metrics["miss_rate"] * 100.0  # %

    lats = mean_mae.lat.to_numpy()
    lons = mean_mae.lon.to_numpy()

    # Detect resolution and CMZ polygon
    lat_diff = detect_resolution(lats)
    cmz_coords = get_cmz_polygon_coords(lat_diff)
    polygon_defined = cmz_coords is not None

    if polygon_defined:
        polygon_lon, polygon_lat = cmz_coords
    else:
        polygon_lon, polygon_lat = None, None

    # MAE stats: this helper already computes CMZ + overall
    cmz_mae_mean, cmz_mae_se, overall_mae_mean, _overall_mae_se = (
        calculate_mae_stats_across_years(
            spatial_metrics=spatial_metrics,
            lons=lons,
            lats=lats,
            polygon_lon=polygon_lon,
            polygon_lat=polygon_lat,
            polygon_defined=polygon_defined,
        )
    )

    # Overall FAR/MR (domain mean)
    overall_far = _global_nanmean(far.to_numpy().ravel())
    overall_mr = _global_nanmean(mr.to_numpy().ravel())

    # CMZ FAR/MR via polygon if available
    if polygon_defined and polygon_lon is not None and polygon_lat is not None:
        cmz_far = float(
            calculate_cmz_averages(
                far,
                lons,
                lats,
                polygon_lon,
                polygon_lat,
            )
        )
        cmz_mr = float(
            calculate_cmz_averages(
                mr,
                lons,
                lats,
                polygon_lon,
                polygon_lat,
            )
        )
    else:
        cmz_far = float("nan")
        cmz_mr = float("nan")

    return {
        "cmz_mae_mean_days": cmz_mae_mean,
        "cmz_mae_se_days": cmz_mae_se,
        "cmz_far_pct": cmz_far,
        "cmz_mr_pct": cmz_mr,
        "overall_mae_mean_days": overall_mae_mean,
        "overall_far_pct": overall_far,
        "overall_mr_pct": overall_mr,
    }


def create_model_comparison_table(
    model_spatial_metrics: dict[str, Mapping[str, xr.DataArray]],
) -> pd.DataFrame:
    """Build tidy comparison table for multiple models (CMZ + overall)."""
    rows: list[dict[str, float | str]] = []

    for model_name, spatial_metrics in model_spatial_metrics.items():
        _validate_spatial_metrics_keys(model_name, spatial_metrics)
        summary = _summarize_single_model(spatial_metrics)
        summary["model"] = model_name
        rows.append(summary)

    comparison_df = pd.DataFrame(rows).set_index("model")

    ordered_cols = [
        "cmz_mae_mean_days",
        "cmz_mae_se_days",
        "cmz_far_pct",
        "cmz_mr_pct",
        "overall_mae_mean_days",
        "overall_far_pct",
        "overall_mr_pct",
    ]
    existing_cols = [c for c in ordered_cols if c in comparison_df.columns]
    return comparison_df[existing_cols]


def plot_model_comparison_dual_axis(
    comparison_df: pd.DataFrame,
    mae_col: str = "cmz_mae_mean_days",  # CMZ MAE
    mae_err_col: str = "cmz_mae_se_days",  # CMZ SE
    rate_cols: Sequence[str] = ("cmz_far_pct", "cmz_mr_pct"),
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str | None = None,
    rotation: int = 0,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot grouped bar chart with dual y-axis (MAE vs FAR/MR)."""
    # ---- basic checks ----
    if mae_col not in comparison_df.columns:
        raise ValueError(f"{mae_col!r} not found in comparison_df columns.")

    missing_rates = [c for c in rate_cols if c not in comparison_df.columns]
    if missing_rates:
        raise ValueError(f"Rate metric(s) not found in comparison_df: {missing_rates}")

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    # Ensure labels are the model names (strings)
    models = comparison_df.index.astype(str).tolist()
    x = np.arange(len(models), dtype=float)

    # Total number of bars per group = 1 (MAE) + len(rate_cols)
    n_bars = 1 + len(rate_cols)
    width = 0.8 / float(n_bars)

    # --- aesthetic choices ---
    mae_color = "#1f77b4"  # blue
    far_color = "#ff7f0e"  # orange
    mr_color = "#2ca02c"  # green
    rate_colors = [far_color, mr_color][: len(rate_cols)]

    # Primary axis (MAE)
    ax_left.spines["left"].set_visible(True)
    ax_left.spines["bottom"].set_visible(True)
    ax_left.spines["top"].set_visible(True)
    ax_left.spines["right"].set_visible(False)

    # Secondary axis (Rates)
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["top"].set_visible(True)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)

    # ax_left.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)

    # ---- MAE on left axis ----
    mae_values = comparison_df[mae_col].to_numpy()
    mae_offset = (0 - (n_bars - 1) / 2.0) * width

    if mae_err_col in comparison_df.columns:
        mae_err = comparison_df[mae_err_col].to_numpy()
    else:
        mae_err = None

    ax_left.bar(
        x + mae_offset,
        mae_values,
        width=width,
        label="MAE (days)",
        color=mae_color,
        edgecolor="black",
        linewidth=0.5,
        yerr=mae_err,
        capsize=3 if mae_err is not None else 0,
    )

    # ---- FAR & MR on right axis ----
    rate_bars = []
    for i, (col, c) in enumerate(zip(rate_cols, rate_colors), start=1):
        offset = (i - (n_bars - 1) / 2.0) * width
        values = comparison_df[col].to_numpy()
        label = (
            "FAR (%)"
            if "far" in col.lower()
            else "Miss Rate (%)"
            if "mr" in col.lower() or "miss" in col.lower()
            else col.replace("_", " ").title()
        )

        bar = ax_right.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=c,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        rate_bars.append(bar)

    # ---- Axes formatting ----
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(models, rotation=rotation)

    ax_left.set_ylabel("MAE (days)")
    ax_right.set_ylabel("Rate (%)")

    # keep FAR/MR comparable on [0, 100]
    ax_right.set_ylim(0, 100)

    # if title:
    #     ax_left.set_title(title)

    # ---- Legend in upper right ----
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    handles = handles_left + handles_right
    labels = labels_left + labels_right

    ax_right.legend(
        handles,
        labels,
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
    return fig, (ax_left, ax_right)


def compare_models(
    model_spatial_metrics: dict[str, Mapping[str, xr.DataArray]],
    mae_col: str = "cmz_mae_mean_days",
    rate_cols: Sequence[str] = ("cmz_far_pct", "cmz_mr_pct"),
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str | None = None,
    rotation: int = 0,
) -> tuple[pd.DataFrame, plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Compare multiple models by creating a table and dual-axis plot."""
    comparison_df = create_model_comparison_table(model_spatial_metrics)
    fig, (ax_left, ax_right) = plot_model_comparison_dual_axis(
        comparison_df=comparison_df,
        mae_col=mae_col,
        rate_cols=rate_cols,
        figsize=figsize,
        title=title,
        rotation=rotation,
    )
    return comparison_df, fig, (ax_left, ax_right)

def get_target_bins(brier_forecast, brier_climatology):
    """Extract and sort target bins"""
    all_forecast_bins = set(brier_forecast['bin_fair_brier_scores'].keys())
    all_clim_bins = set(brier_climatology['bin_fair_brier_scores'].keys())
    common_bins = all_forecast_bins.intersection(all_clim_bins)
    target_bins = []
    for bin_label in common_bins:
        if (bin_label.startswith('Days ') and 
            not bin_label.startswith('After') and 
            not bin_label.startswith('Before')):
            target_bins.append(bin_label)
    
    def extract_day_range(bin_label):
        if 'Days ' in bin_label:
            try:
                day_part = bin_label.replace('Days ', '').split('-')[0]
                return int(day_part)
            except:
                return 999
        return 999
    
    return sorted(target_bins, key=extract_day_range)

def create_heatmap(skill_results, auc_forecast, auc_climatology, 
                  brier_forecast, brier_climatology, model_name, max_forecast_day, save_dir=None):
    """
    Create and save skill score heatmap
    
    Parameters:
    -----------
    ... (other parameters)
    save_dir : str, optional
        Directory to save the heatmap. If None, saves in current directory.
        If directory doesn't exist, it will be created.
    """
    
    # Handle save directory
    if save_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Ensure save_dir ends with a path separator for proper joining
        if not save_dir.endswith(os.sep):
            save_dir += os.sep
    else:
        save_dir = ""
    
    target_bins = get_target_bins(brier_forecast, brier_climatology)
    
    # Prepare data
    bss_values = [skill_results['bin_fair_brier_skill_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    auc_values = [auc_forecast['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    auc_clim_values = [auc_climatology['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    
    bin_labels_short = [bin_name.replace('Days ', '') for bin_name in target_bins]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    
    # Plot 1: BSS heatmap
    bss_data = np.array(bss_values).reshape(1, -1)
    sns.heatmap(bss_data*100, 
                annot=True, 
                fmt='.2g', 
                cmap='RdBu',
                vmin=-40, vmax=40,
                center=0,
                xticklabels=bin_labels_short,
                cbar_kws={"orientation": "horizontal"},
                ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    ax1.set_xlabel('')
    ax1.set_xticklabels([])
    ax1.set_ylabel('BSS (%)', fontsize=14)
    ax1.set_yticklabels([])
    
    # Plot 2: AUC heatmap
    auc_data = np.array(auc_values).reshape(1, -1)
    sns.heatmap(auc_data, 
                annot=False,
                cmap='Blues',
                vmin=0.7, vmax=1.0,
                xticklabels=bin_labels_short,
                cbar_kws={"orientation": "horizontal"},
                ax=ax2)
    
    # Add custom annotations
    for i, (auc_val, auc_clim_val) in enumerate(zip(auc_values, auc_clim_values)):
        if not np.isnan(auc_val) and not np.isnan(auc_clim_val):
            ax2.text(i + 0.5, 0.5, f'{auc_val:.2g}', 
                    ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='black')
            ax2.text(i + 0.5, 0.2, f'({auc_clim_val:.2g})', 
                    ha='center', va='center', 
                    fontsize=8, color='darkblue')
        elif not np.isnan(auc_val):
            ax2.text(i + 0.5, 0.5, f'{auc_val:.2g}', 
                    ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='black')
    
    ax2.set_xlabel('Forecast Day Bins', fontsize=14)
    ax2.set_ylabel('AUC', fontsize=14)
    ax2.set_yticklabels([])
    
    plt.tight_layout()
    
    # Save with model name and forecast days
    figure_filename = f'{save_dir}skill_scores_heatmap_{model_name}_{max_forecast_day}day.png'
    plt.savefig(figure_filename, dpi=300, bbox_inches='tight')
    plt.show() #Can delete for non-notebook vis
    plt.close()
    
    print(f"Figure saved as '{figure_filename}'")
    
    return figure_filename


def plot_reliability_diagram(forecast_obs_pairs_multi, years, max_forecast_day, save_path=None):
    """Plot reliability diagram from forecast-observation pairs."""
    
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    reliability_y = np.zeros(n_bins)
    mean_forecast_prob = np.zeros(n_bins)
    frequency = np.zeros(n_bins)
    n_forecasts_array = np.zeros(n_bins)

    print("\nReliability Analysis:")
    print("Bin Range\t\tN_Forecasts\tMean_Forecast_Prob\tReliability\tFrequency\tError_Bar")
    print("-" * 90)

    results_for_csv = []

    for i in range(n_bins):
        if i == 0:
            in_bin = ((forecast_obs_pairs_multi['predicted_prob'] >= bin_edges[i]) & 
                    (forecast_obs_pairs_multi['predicted_prob'] <= bin_edges[i+1]))
        else:
            in_bin = ((forecast_obs_pairs_multi['predicted_prob'] > bin_edges[i]) & 
                    (forecast_obs_pairs_multi['predicted_prob'] <= bin_edges[i+1]))
        
        n_forecasts = in_bin.sum()
        n_forecasts_array[i] = n_forecasts
        
        if n_forecasts > 0:
            mean_forecast_prob[i] = forecast_obs_pairs_multi.loc[in_bin, 'predicted_prob'].mean()
            reliability_y[i] = forecast_obs_pairs_multi.loc[in_bin, 'observed_onset'].mean()
            frequency[i] = n_forecasts / len(forecast_obs_pairs_multi)
            error_bar = np.sqrt(reliability_y[i] * (1 - reliability_y[i]) / n_forecasts)
        else:
            mean_forecast_prob[i] = np.nan
            reliability_y[i] = np.nan
            frequency[i] = 0
            error_bar = np.nan
        
        bin_range = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        
        print(f"{bin_range}\t\t{n_forecasts}\t\t{mean_forecast_prob[i]:.3f}\t\t\t{reliability_y[i]:.3f}\t\t{frequency[i]:.3f}\t\t{error_bar:.3f}")
        
        results_for_csv.append({
            'Bin_Range': bin_range,
            'N_Forecasts': n_forecasts,
            'Mean_Forecast_Prob': round(mean_forecast_prob[i], 3) if not np.isnan(mean_forecast_prob[i]) else np.nan,
            'Observed_Frequency': round(reliability_y[i], 3) if not np.isnan(reliability_y[i]) else np.nan,
            'Frequency': round(frequency[i], 3),
            'Error_Bar': round(error_bar, 3) if not np.isnan(error_bar) else np.nan
        })

    results_df = pd.DataFrame(results_for_csv)

    error_bars = np.sqrt(reliability_y * (1 - reliability_y) / n_forecasts_array)
    error_bars = np.where(n_forecasts_array > 0, error_bars, 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    valid_bins = ~np.isnan(reliability_y) & ~np.isnan(mean_forecast_prob)
    ax.errorbar(mean_forecast_prob[valid_bins], reliability_y[valid_bins], 
                yerr=error_bars[valid_bins], fmt='o-', 
                color='blue', linewidth=2, markersize=8, capsize=5, capthick=2,
                label='Reliability')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Reliability')

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.bar(bin_centers, frequency, width=0.08, alpha=0.3, color='gray', label='Frequency')
    max_freq = max(frequency)
    min_freq = min([f for f in frequency if f > 0]) if any(f > 0 for f in frequency) else 1e-4
    ax2.set_ylim(min_freq * 0.5, max_freq * 2)
    ax2.set_ylabel('Forecast frequency', fontsize=12)

    ax.set_xlabel('Forecast Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)

    if len(years) > 1:
        year_str = f"{min(years)}-{max(years)}"
    else:
        year_str = str(years[0])

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Save figure if save_path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_save_path = os.path.join(save_path, f'reliability_{max_forecast_day}day.png')
        fig.savefig(fig_save_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {fig_save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, results_df


