"""Model comparison utilities for monsoon onset metrics.

This module provides:

1. create_model_comparison_table
   Build a tidy pandas.DataFrame summarizing key metrics for multiple models
   using their spatial metric fields.

2. plot_model_comparison
   Create grouped or stacked bar charts from the summary table for quick
   visual comparison across models.

Typical usage:
--------------
>>> from monsoonbench.visualization.compare_models import (
...     create_model_comparison_table,
...     plot_model_comparison,
... )
>>> comparison_df = create_model_comparison_table(
...     {
...         "Model_A": spatial_metrics_a,
...         "Model_B": spatial_metrics_b,
...     }
... )
>>> fig, ax = plot_model_comparison(
...     comparison_df,
...     metrics=["cmz_mae_mean", "cmz_far"],
...     style="grouped",
...     title="CMZ MAE and FAR by Model",
...     ylabel="Value",
... )

Notes:
- Expects each spatial_metrics dict to follow the same schema as used in
  `plot_spatial_metrics`:
    - "mean_mae": xr.DataArray
    - "false_alarm_rate": xr.DataArray in [0, 1]
    - "miss_rate": xr.DataArray in [0, 1]
    - "mae_YYYY": yearly MAE maps used by calculate_mae_stats_across_years
- Assumes metrics for a given model share a common (lat, lon) grid.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from monsoonbench.spatial.regions import (
    detect_resolution,
    get_cmz_polygon_coords,
)
from monsoonbench.visualization import (
    calculate_cmz_averages,
    calculate_mae_stats_across_years,
)

__all__ = [
    "create_model_comparison_table",
    "plot_model_comparison",
    "compare_models",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_spatial_metrics_keys(
    model_name: str, spatial_metrics: dict[str, xr.DataArray]
) -> None:
    """Validate that required keys exist in spatial_metrics for a model.

    Args:
        model_name: Name of the model being validated.
        spatial_metrics: Mapping of metric names to DataArray objects.
    """
    required = ["mean_mae", "false_alarm_rate", "miss_rate"]
    missing = [key for key in required if key not in spatial_metrics]
    if missing:
        raise KeyError(
            f"Missing required keys for model '{model_name}': {missing}. "
            "Expected at least 'mean_mae', 'false_alarm_rate', and 'miss_rate'.",
        )


def _summarize_single_model(
    spatial_metrics: dict[str, xr.DataArray],
) -> dict[str, float]:
    """Create a consistent summary of metrics for a single model.

    Expected spatial_metrics keys:
        - "mean_mae": xr.DataArray
        - "false_alarm_rate": xr.DataArray (0-1)
        - "miss_rate": xr.DataArray (0-1)
        - "mae_YYYY": yearly MAE maps (for MAE stats across years)

    Returns:
        Dictionary with:
            - cmz_mae_mean, cmz_mae_se
            - overall_mae_mean, overall_mae_se
            - cmz_far, overall_far
            - cmz_mr, overall_mr
    """
    mean_mae = spatial_metrics["mean_mae"]
    far = spatial_metrics["false_alarm_rate"] * 100.0  # %
    mr = spatial_metrics["miss_rate"] * 100.0  # %

    lats = mean_mae.lat.to_numpy()
    lons = mean_mae.lon.to_numpy()

    # CMZ geometry
    lat_diff = detect_resolution(lats)
    cmz_coords = get_cmz_polygon_coords(lat_diff)
    polygon_defined = cmz_coords is not None

    if polygon_defined:
        polygon_lon, polygon_lat = cmz_coords
    else:
        polygon_lon, polygon_lat = None, None

    # MAE statistics (existing helper)
    cmz_mae_mean, cmz_mae_se, overall_mae_mean, overall_mae_se = (
        calculate_mae_stats_across_years(
            spatial_metrics=spatial_metrics,
            lons=lons,
            lats=lats,
            polygon_lon=polygon_lon,
            polygon_lat=polygon_lat,
            polygon_defined=polygon_defined,
        )
    )

    # FAR / MR: overall means
    overall_far = float(np.nanmean(far.to_numpy()))
    overall_mr = float(np.nanmean(mr.to_numpy()))

    # FAR / MR: CMZ means if polygon available
    if polygon_defined and polygon_lon is not None and polygon_lat is not None:
        cmz_far = float(
            calculate_cmz_averages(
                far,
                lons,
                lats,
                polygon_lon,
                polygon_lat,
            ),
        )
        cmz_mr = float(
            calculate_cmz_averages(
                mr,
                lons,
                lats,
                polygon_lon,
                polygon_lat,
            ),
        )
    else:
        cmz_far = np.nan
        cmz_mr = np.nan

    return {
        "cmz_mae_mean": cmz_mae_mean,
        "cmz_mae_se": cmz_mae_se,
        "overall_mae_mean": overall_mae_mean,
        "overall_mae_se": overall_mae_se,
        "cmz_far": cmz_far,
        "overall_far": overall_far,
        "cmz_mr": cmz_mr,
        "overall_mr": overall_mr,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_model_comparison_table(
    model_spatial_metrics: dict[str, dict[str, xr.DataArray]],
) -> pd.DataFrame:
    """Build a tidy comparison table for multiple models.

    Args:
        model_spatial_metrics:
            Mapping from model name -> spatial_metrics dict in the same format
            used by `plot_spatial_metrics`.

    Returns:
        DataFrame indexed by model name with columns:
            ["cmz_mae_mean", "cmz_mae_se",
             "overall_mae_mean", "overall_mae_se",
             "cmz_far", "overall_far",
             "cmz_mr", "overall_mr"]
    """
    rows: list[dict[str, float | str]] = []

    for model_name, spatial_metrics in model_spatial_metrics.items():
        _validate_spatial_metrics_keys(model_name, spatial_metrics)
        summary = _summarize_single_model(spatial_metrics)
        summary["model"] = model_name
        rows.append(summary)

    comparison_df = pd.DataFrame(rows).set_index("model")

    ordered_cols = [
        "cmz_mae_mean",
        "cmz_mae_se",
        "overall_mae_mean",
        "overall_mae_se",
        "cmz_far",
        "overall_far",
        "cmz_mr",
        "overall_mr",
    ]
    existing_cols = [col for col in ordered_cols if col in comparison_df.columns]
    return comparison_df[existing_cols]


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: list[str],
    style: str = "grouped",
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str | None = None,
    ylabel: str | None = None,
    rotation: int = 0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a bar chart comparing selected metrics across models.

    Args:
        comparison_df:
            Output of `create_model_comparison_table` (index = model name).
        metrics:
            List of column names from comparison_df to visualize.
        style:
            "grouped" -> side-by-side columns for each metric per model.
            "stacked" -> stacked columns for each model.
        figsize:
            Size of the figure in inches (width, height).
        title:
            Optional plot title.
        ylabel:
            Optional y-axis label.
        rotation:
            Rotation (degrees) for x-axis tick labels.

    Returns:
        Tuple of (fig, ax): Matplotlib figure and axes.
    """
    if not metrics:
        raise ValueError("At least one metric must be provided for plotting.")

    missing = [metric for metric in metrics if metric not in comparison_df.columns]
    if missing:
        raise ValueError(f"Metrics not found in comparison_df: {missing}")

    data = comparison_df[metrics]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(data.index), dtype=float)  # models
    n_metrics = len(metrics)

    if style == "grouped":
        # Side-by-side bars per model
        width = 0.8 / float(n_metrics)
        for i, metric in enumerate(metrics):
            offset = (i - (n_metrics - 1) / 2.0) * width
            values = data[metric].to_numpy()
            ax.bar(x + offset, values, width=width, label=metric)
    elif style == "stacked":
        # Stacked bars per model
        bottom = np.zeros(len(data.index), dtype=float)
        for metric in metrics:
            values = data[metric].to_numpy()
            ax.bar(x, values, bottom=bottom, label=metric)
            bottom += values
    else:
        raise ValueError("style must be 'grouped' or 'stacked'.")

    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=rotation)
    ax.legend(frameon=False)

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.tight_layout()
    return fig, ax


def compare_models(
    model_spatial_metrics: dict[str, dict[str, xr.DataArray]],
    metrics: list[str],
    style: str = "grouped",
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str | None = None,
    ylabel: str | None = None,
    rotation: int = 0,
) -> tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """Create a comparison table and plot for multiple models in one call.

    Args:
        model_spatial_metrics:
            Mapping from model name to its spatial_metrics dictionary.
        metrics:
            List of metric column names (from the comparison table)
            to visualize in the bar chart.
        style:
            Bar chart style: "grouped" for side-by-side bars per model,
            or "stacked" for stacked bars per model.
        figsize:
            Size of the figure in inches (width, height).
        title:
            Optional title for the generated plot.
        ylabel:
            Optional y-axis label for the plot.
        rotation:
            Rotation (degrees) for x-axis tick labels.

    Returns:
        Tuple of:
            - comparison_df: pandas DataFrame with summary metrics.
            - fig: Matplotlib Figure object for the comparison plot.
            - ax: Matplotlib Axes object for the comparison plot.
    """
    comparison_df = create_model_comparison_table(model_spatial_metrics)
    fig, ax = plot_model_comparison(
        comparison_df=comparison_df,
        metrics=metrics,
        style=style,
        figsize=figsize,
        title=title,
        ylabel=ylabel,
        rotation=rotation,
    )
    return comparison_df, fig, ax
