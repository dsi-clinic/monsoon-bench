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

import os
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    ProbabilisticOnsetMetrics,
)
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
    "plot_probabilistic_model_comparison_dual_axis",
    "create_probabilistic_model_comparison_table",
    "compare_probabilistic_models",
    "run_reliability_analysis",
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


def create_probabilistic_model_comparison_table(
    base_config, model_info
) -> pd.DataFrame:
    """Create table with probabilistic scores for each model."""
    pr = ProbabilisticOnsetMetrics()
    cl = ClimatologyOnsetMetrics()

    model_results = []

    day_bins = get_day_bins(base_config["max_forecast_day"])

    # Climatology Baseline
    thresh_ds = xr.open_dataset(base_config["thres_file"])
    clim_onset = cl.compute_climatological_onset_dataset(
        base_config["imd_folder"],
        thresh_ds["MWmean"],
        years=None,
        mok=base_config["mok"],
    )

    for name, info in model_info.items():
        config = info["config"]
        model_forecast_dir = info["model_forecast_dir"]
        print(config)
        print(model_forecast_dir)

        baseline_df = cl.multi_year_climatological_forecast_obs_pairs(
            clim_onset,
            config["years"],
            day_bins,
            config["mem_num"],
            model_forecast_dir,
            max_forecast_day=config["max_forecast_day"],
            mok=config["mok"],
            file_pattern=config["file_pattern"],
            date_filter_year=config["date_filter_year"],
        )
        brier_clim = cl.calculate_brier_score_climatology(baseline_df)
        rps_clim = pr.calculate_rps(baseline_df)

        print(f"Processing {name}...")
        forecast_obs_df = pr.multi_year_forecast_obs_pairs(
            config["years"],
            model_forecast_dir,
            config["imd_folder"],
            config["thres_file"],
            config["mem_num"],
            config["max_forecast_day"],
            day_bins,
            mok=config["mok"],
            file_pattern=config["file_pattern"],
            date_filter_year=config["date_filter_year"],
        )

        brier_forecast = pr.calculate_brier_score(forecast_obs_df)
        rps_forecast = pr.calculate_rps(forecast_obs_df)
        auc_forecast = pr.calculate_auc(forecast_obs_df)
        skill = pr.calculate_skill_scores(
            brier_forecast, rps_forecast, brier_clim, rps_clim
        )

        # Store as a list of dicts to maintain sequence
        model_results.append(
            {
                "model": name,
                "fair_brier_skill_score": skill["fair_brier_skill_score"] * 100,
                "fair_rps_skill_score": skill["fair_rps_skill_score"] * 100,
                "auc_score": auc_forecast["auc"],
            }
        )

    # Create DataFrame
    comparison_df = pd.DataFrame(model_results).set_index("model")
    return comparison_df


def plot_probabilistic_model_comparison_dual_axis(
    comparison_df: pd.DataFrame, title: str | None = None
) -> None:
    """Plots BSS and RPSS and AUC in dual axis comparison graph."""
    # Reverse to keep first model at the top
    df_plot = comparison_df.iloc[::-1]
    models = df_plot.index.tolist()
    y_pos = np.arange(len(models))

    # Increase bar height to fill more vertical space
    height = 0.35

    # Adjust figsize to be more balanced
    fig_width = 8.0
    fig_height = max(6.0, len(models) * 1.2)

    fig, ax_bottom = plt.subplots(figsize=(fig_width, fig_height))
    ax_top = ax_bottom.twiny()

    # Colors
    color_bss = "#a6cee3"  # Light Blue
    color_rpss = "#1f78b4"  # Dark Blue
    color_auc = "#ff7f0e"  # Orange

    # Bottom Axis: BSS & RPSS
    ax_bottom.barh(
        y_pos + height / 2,
        df_plot["fair_brier_skill_score"],
        height,
        label="BSS (%)",
        color=color_bss,
        edgecolor="black",
        linewidth=0.8,
    )
    ax_bottom.barh(
        y_pos - height / 2,
        df_plot["fair_rps_skill_score"],
        height,
        label="RPSS (%)",
        color=color_rpss,
        edgecolor="black",
        linewidth=0.8,
    )

    # Top Axis: AUC
    ax_top.barh(
        y_pos,
        df_plot["auc_score"],
        height / 2,
        label="AUC",
        color=color_auc,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
    )

    # Axis Formatting
    ax_bottom.set_xlim(-20, 50)
    ax_bottom.set_xlabel("Skill Score (BSS/RPSS) %", fontweight="bold")
    ax_bottom.set_yticks(y_pos)
    ax_bottom.set_yticklabels(models, fontsize=10)
    ax_bottom.axvline(0, color="navy", linewidth=1, alpha=0.4)

    ax_top.set_xlim(0.8, 1.0)
    ax_top.set_xlabel("AUC Score", color=color_auc, fontweight="bold")
    ax_top.tick_params(axis="x", colors=color_auc)

    lines_b, labels_b = ax_bottom.get_legend_handles_labels()
    lines_t, labels_t = ax_top.get_legend_handles_labels()
    ax_bottom.legend(
        lines_b + lines_t,
        labels_b + labels_t,
        loc="upper right",
        bbox_to_anchor=(1, 0.15),
        frameon=True,
    )

    if title:
        plt.title(title, pad=45, fontsize=12, fontweight="bold")

    plt.tight_layout()


def compare_probabilistic_models(base_config, model_info) -> pd.DataFrame:
    """Create probabilistic score table and plot dual comparison graph."""
    comparison_df = create_probabilistic_model_comparison_table(base_config, model_info)
    fig = plot_probabilistic_model_comparison_dual_axis(
        comparison_df=comparison_df, title="Probabilistic Skill"
    )
    # Save with model name and forecast days
    figure_filename = f"{base_config["save_dir"]}probabilistic_comparison_graph.png"
    plt.savefig(figure_filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Figure saved as '{figure_filename}'")
    return comparison_df


def get_target_bins(brier_forecast, brier_climatology):
    """Extract and sort target bins"""
    all_forecast_bins = set(brier_forecast["bin_fair_brier_scores"].keys())
    all_clim_bins = set(brier_climatology["bin_fair_brier_scores"].keys())
    common_bins = all_forecast_bins.intersection(all_clim_bins)
    target_bins = []
    for bin_label in common_bins:
        if (
            bin_label.startswith("Days ")
            and not bin_label.startswith("After")
            and not bin_label.startswith("Before")
        ):
            target_bins.append(bin_label)

    def extract_day_range(bin_label):
        if "Days " in bin_label:
            try:
                day_part = bin_label.replace("Days ", "").split("-")[0]
                return int(day_part)
            except:
                return 999
        return 999

    return sorted(target_bins, key=extract_day_range)


def create_heatmap(config):
    heatmap_data = generate_heatmap_data(config)
    create_heatmap_visual(
        heatmap_data,
        config["model_name"],
        config["max_forecast_day"],
        save_dir=str(config["save_dir"]),
    )


def generate_heatmap_data(config):
    # Initialize Metric Computation Classes
    pr = ProbabilisticOnsetMetrics()
    cl = ClimatologyOnsetMetrics()

    # Set day bins
    day_bins = get_day_bins(config["max_forecast_day"])

    # Print config metadata
    print("=" * 60)
    print("S2S MONSOON ONSET SKILL SCORE ANALYSIS")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Years: {config['years']}")
    print(f"Max forecast day: {config['max_forecast_day']}")
    print(f"Day bins: {day_bins}")
    print(f"MOK filter: {config['mok']} ({'June 2nd' if config['mok'] else 'May 1st'})")
    if config["save_dir"]:
        print(f"Output directory: {config['save_dir']}")
    else:
        print("Output directory: current directory")
    print("=" * 60)

    # Create forecast observation dataframe
    print("\n1. Processing forecast model...")
    forecast_obs_df = pr.multi_year_forecast_obs_pairs(
        config["years"],
        config["model_forecast_dir"],
        config["imd_folder"],
        config["thres_file"],
        config["mem_num"],
        config["max_forecast_day"],
        day_bins,
        date_filter_year=config["date_filter_year"],
        file_pattern=config["file_pattern"],
        mok=config["mok"],
    )

    # Calculate brier and AUC forecast scores from observations
    print("\n2. Calculating brier and AUC forecast scores...")
    brier_forecast = pr.calculate_brier_score(forecast_obs_df)
    rps_forecast = pr.calculate_rps(forecast_obs_df)
    auc_forecast = pr.calculate_auc(forecast_obs_df)

    print("\n3. Computing climatological dataset...")
    thresh_ds = xr.open_dataset(config["thres_file"])
    thresh_slice = thresh_ds["MWmean"]
    clim_onset = cl.compute_climatological_onset_dataset(
        config["imd_folder"], thresh_slice, years=None, mok=config["mok"]
    )

    print("\n4. Processing climatological forecasts...")
    climatology_obs_df = cl.multi_year_climatological_forecast_obs_pairs(
        clim_onset,
        config["years"],
        day_bins,
        config["mem_num"],
        config["model_forecast_dir"],
        date_filter_year=config["date_filter_year"],
        file_pattern=config["file_pattern"],
        max_forecast_day=config["max_forecast_day"],
        mok=config["mok"],
    )

    print("\n5. Calculating climatology scores...")
    brier_climatology = cl.calculate_brier_score_climatology(climatology_obs_df)
    rps_climatology = pr.calculate_rps(climatology_obs_df)
    auc_climatology = cl.calculate_auc_climatology(climatology_obs_df)

    # Calculate skill scores
    print("\n6. Calculating skill scores...")
    skill_results = pr.calculate_skill_scores(
        brier_forecast, rps_forecast, brier_climatology, rps_climatology
    )

    heatmap_data = {
        "skill_results": skill_results,
        "auc_forecast": auc_forecast,
        "auc_climatology": auc_climatology,
        "brier_forecast": brier_forecast,
        "brier_climatology": brier_climatology,
    }

    return heatmap_data


def create_heatmap_visual(heatmap_data, model_name, max_forecast_day, save_dir=None):
    """Create and save skill score heatmap

    Parameters:
    -----------
    ... (other parameters)
    save_dir : str, optional
        Directory to save the heatmap. If None, saves in current directory.
        If directory doesn't exist, it will be created.
    """
    skill_results = heatmap_data.get("skill_results")
    auc_forecast = heatmap_data.get("auc_forecast")
    auc_climatology = heatmap_data.get("auc_climatology")
    brier_forecast = heatmap_data.get("brier_forecast")
    brier_climatology = heatmap_data.get("brier_climatology")

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
    bss_values = [
        skill_results["bin_fair_brier_skill_scores"].get(bin_name, np.nan)
        for bin_name in target_bins
    ]
    auc_values = [
        auc_forecast["bin_auc_scores"].get(bin_name, np.nan) for bin_name in target_bins
    ]
    auc_clim_values = [
        auc_climatology["bin_auc_scores"].get(bin_name, np.nan)
        for bin_name in target_bins
    ]

    bin_labels_short = [bin_name.replace("Days ", "") for bin_name in target_bins]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))

    # Plot 1: BSS heatmap
    bss_data = np.array(bss_values).reshape(1, -1)
    sns.heatmap(
        bss_data * 100,
        annot=True,
        fmt=".2g",
        cmap="RdBu",
        vmin=-40,
        vmax=40,
        center=0,
        xticklabels=bin_labels_short,
        cbar_kws={"orientation": "horizontal"},
        ax=ax1,
        annot_kws={"size": 12, "weight": "bold"},
    )

    ax1.set_xlabel("")
    ax1.set_xticklabels([])
    ax1.set_ylabel("BSS (%)", fontsize=14)
    ax1.set_yticklabels([])

    # Plot 2: AUC heatmap
    auc_data = np.array(auc_values).reshape(1, -1)
    sns.heatmap(
        auc_data,
        annot=False,
        cmap="Blues",
        vmin=0.5,
        vmax=1.0,
        xticklabels=bin_labels_short,
        cbar_kws={"orientation": "horizontal"},
        ax=ax2,
    )

    # Add custom annotations
    for i, (auc_val, auc_clim_val) in enumerate(zip(auc_values, auc_clim_values)):
        if not np.isnan(auc_val) and not np.isnan(auc_clim_val):
            ax2.text(
                i + 0.5,
                0.5,
                f"{auc_val:.2g}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )
            ax2.text(
                i + 0.5,
                0.2,
                f"({auc_clim_val:.2g})",
                ha="center",
                va="center",
                fontsize=8,
                color="darkblue",
            )
        elif not np.isnan(auc_val):
            ax2.text(
                i + 0.5,
                0.5,
                f"{auc_val:.2g}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )

    ax2.set_xlabel("Forecast Day Bins", fontsize=14)
    ax2.set_ylabel("AUC", fontsize=14)
    ax2.set_yticklabels([])

    plt.tight_layout()

    # Save with model name and forecast days
    figure_filename = (
        f"{save_dir}skill_scores_heatmap_{model_name}_{max_forecast_day}day.png"
    )
    plt.savefig(figure_filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Figure saved as '{figure_filename}'")

    return figure_filename

def get_day_bins(max_days):
    """Helper to generate bins based on forecast window."""
    if max_days == 15:
        return [(1, 5), (6, 10), (11, 15)]
    if max_days == 30:
        return [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
    raise ValueError(f"Unsupported max_forecast_day: {max_days}")

def calculate_reliability_metrics(df, n_bins=10):
    """Logic preserved: calculates reliability, frequency, and error bars per bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    results = []

    for i in range(n_bins):
        # Precise bin logic preserved: i=0 is inclusive, others are (lower, upper]
        if i == 0:
            in_bin = df["predicted_prob"].between(bin_edges[i], bin_edges[i + 1])
        else:
            in_bin = (df["predicted_prob"] > bin_edges[i]) & (
                df["predicted_prob"] <= bin_edges[i + 1]
            )

        n_fcst = in_bin.sum()

        # Calculate statistics (Matches your original math)
        mean_prob = df.loc[in_bin, "predicted_prob"].mean() if n_fcst > 0 else np.nan
        obs_freq = df.loc[in_bin, "observed_onset"].mean() if n_fcst > 0 else np.nan
        rel_freq = n_fcst / len(df)
        err = np.sqrt(obs_freq * (1 - obs_freq) / n_fcst) if n_fcst > 0 else np.nan

        results.append(
            {
                "Bin_Range": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                "N_Forecasts": n_fcst,
                "Mean_Forecast_Prob": round(mean_prob, 3)
                if not np.isnan(mean_prob)
                else np.nan,
                "Observed_Frequency": round(obs_freq, 3)
                if not np.isnan(obs_freq)
                else np.nan,
                "Frequency": round(rel_freq, 3),
                "Error_Bar": round(err, 3) if not np.isnan(err) else np.nan,
                "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,  # For plotting
            }
        )

    return pd.DataFrame(results)

def plot_reliability_diagram(metrics_df, config):
    """Generate and save the reliability diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Reliability Curve
    valid = metrics_df.dropna(subset=["Observed_Frequency"])
    ax.errorbar(
        valid["Mean_Forecast_Prob"],
        valid["Observed_Frequency"],
        yerr=valid["Error_Bar"],
        fmt="o-",
        color="blue",
        lw=2,
        markersize=8,
        capsize=5,
        capthick=2,
        label="Reliability",
    )

    # Reference Line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect Reliability")

    # Frequency Bar Chart (Secondary Axis)
    ax2 = ax.twinx()
    ax2.bar(
        metrics_df["bin_center"],
        metrics_df["Frequency"],
        width=0.08,
        alpha=0.3,
        color="gray",
        label="Frequency",
    )
    ax2.set_yscale("log")
    ax2.set_ylim(0.001, 1)
    ax2.set_ylabel("Forecast frequency", fontsize=12)

    # Styling matches original
    ax.set_xlabel("Forecast Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Save logic
    if config.get("save_dir"):
        os.makedirs(config["save_dir"], exist_ok=True)
        path = os.path.join(
            config["save_dir"], f"reliability_{config['max_forecast_day']}day.png"
        )
        plt.savefig(path, dpi=600, bbox_inches="tight")
        print(f"Figure saved to: {path}")

    plt.tight_layout()
    plt.show()


def run_reliability_analysis(config) -> pd.DataFrame:
    """Main function to run the full monsoon onset reliability pipeline."""
    pr = ProbabilisticOnsetMetrics()
    day_bins = get_day_bins(config["max_forecast_day"])

    # Print Metadata
    print("=" * 60 + "\nS2S MONSOON ONSET SKILL SCORE ANALYSIS\n" + "=" * 60)
    print(f"Model: {config['model_name']} | Years: {config['years']}")

    # Process Data
    forecast_obs_df = pr.multi_year_forecast_obs_pairs(
        config["years"],
        config["model_forecast_dir"],
        config["imd_folder"],
        config["thres_file"],
        config["mem_num"],
        config["max_forecast_day"],
        day_bins,
        date_filter_year=config["date_filter_year"],
        file_pattern=config["file_pattern"],
        mok=config["mok"],
    )

    # Calculate & Display Table
    metrics_df = calculate_reliability_metrics(forecast_obs_df)
    print("\nReliability Analysis Results:")
    print(metrics_df.drop(columns=["bin_center"]))  # Hide plotting helper column

    # Plot
    plot_reliability_diagram(metrics_df, config)

    return metrics_df
