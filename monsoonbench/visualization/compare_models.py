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
    mr = spatial_metrics["miss_rate"] * 100.0          # %

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
    mae_col: str = "cmz_mae_mean_days",      # CMZ MAE
    mae_err_col: str = "cmz_mae_se_days",    # CMZ SE
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
    mae_color = "#1f77b4"   # blue
    far_color = "#ff7f0e"   # orange
    mr_color = "#2ca02c"    # green
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
            "FAR (%)" if "far" in col.lower() else
            "Miss Rate (%)" if "mr" in col.lower() or "miss" in col.lower()
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