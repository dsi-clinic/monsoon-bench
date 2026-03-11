"""Plotting utilities for CMZ forecast-window delta diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm

__all__ = [
    "plot_multi_model_window_deltas",
    "plot_window_delta_heatmap",
    "plot_batch_delta_panels",
]


def plot_window_delta_heatmap(
    delta_df: pd.DataFrame,
    model_name: str,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one model's 3x6 delta table as a compact heatmap.

    Args:
        delta_df: Long-form delta table returned by the window pipeline.
        model_name: Label for the figure title.
        save_path: Optional output PNG path.

    Returns:
        ``(figure, axis)`` for additional customization in notebooks.
    """
    heatmap_df = delta_df.pivot_table(index="metric", columns="window", values="delta")
    metric_order = ["MAE (days)", "FAR (%)", "MR (%)"]
    window_order = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30"]
    heatmap_df = heatmap_df.loc[metric_order, window_order]
    arr = heatmap_df.to_numpy(dtype=float)

    vmax = np.nanmax(np.abs(arr))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    step = max(0.5, round(vmax / 8, 2))
    boundaries = np.arange(-vmax, vmax + step, step)
    norm = BoundaryNorm(boundaries=boundaries, ncolors=plt.cm.RdBu_r.N)

    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(
        arr,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_xticklabels(heatmap_df.columns)
    ax.set_yticks(np.arange(arr.shape[0]))
    ax.set_yticklabels(heatmap_df.index)
    ax.set_xlabel("Forecast Window (days)")
    ax.set_title(f"Window-Delta Metrics vs Climatology | {model_name}")

    for row_idx in range(arr.shape[0]):
        for col_idx in range(arr.shape[1]):
            value = arr[row_idx, col_idx]
            text = "nan" if np.isnan(value) else f"{value:.2f}"
            text_color = (
                "white" if np.isfinite(value) and abs(value) > 0.55 * vmax else "black"
            )
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    colorbar = plt.colorbar(image, ax=ax, shrink=0.92, boundaries=boundaries)
    colorbar.set_label("Model - Climatology (negative is better)")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_multi_model_window_deltas(
    delta_for_plot: dict[str, pd.DataFrame],
    model_order: list[str],
    window_labels: list[str],
    model_labels: dict[str, str],
    save_path: Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot 3 side-by-side delta panels for MAE/FAR/MR across models.

    Args:
        delta_for_plot: Mapping with keys ``"MAE (days)"``, ``"FAR (%)"``,
            and ``"MR (%)"``. Each value is a model x window matrix.
        model_order: Row order for models.
        window_labels: Column order for forecast windows.
        model_labels: Display labels per model key.
        save_path: Optional output PNG path.

    Returns:
        ``(figure, axes)`` for notebook-level extensions.
    """
    panel_specs: list[tuple[str, float, float, np.ndarray, list[int]]] = [
        ("MAE (days)", -4, 4, np.arange(-4, 4.5, 0.5), [-4, -3, -2, -1, 0, 1, 2, 3, 4]),
        ("FAR (%)", -18, 18, np.arange(-18, 19, 3), [-18, -12, -6, 0, 6, 12, 18]),
        ("MR (%)", -60, 60, np.arange(-60, 61, 10), [-60, -40, -20, 0, 20, 40, 60]),
    ]

    for metric, _, _, _, _ in panel_specs:
        if metric not in delta_for_plot:
            raise ValueError(f"Missing metric '{metric}' in delta_for_plot.")

    fig, axes = plt.subplots(
        1, 3, figsize=(13.8, 5.2), dpi=170, constrained_layout=True
    )

    for ax, (metric, vmin, vmax, boundaries, ticks) in zip(axes, panel_specs):
        metric_df = delta_for_plot[metric].loc[model_order, window_labels]
        arr = metric_df.to_numpy(dtype=float)
        norm = BoundaryNorm(boundaries=boundaries, ncolors=plt.cm.RdBu_r.N)
        image = ax.imshow(
            arr,
            cmap="RdBu_r",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(f"$\\Delta$ {metric}", fontsize=14)
        ax.set_xticks(np.arange(len(window_labels)))
        ax.set_xticklabels(window_labels, fontsize=10)
        ax.set_yticks(np.arange(len(model_order)))
        if ax is axes[0]:
            ax.set_yticklabels(
                [model_labels[name] for name in model_order], fontsize=14
            )
        else:
            ax.set_yticklabels([])

        for row_idx in range(arr.shape[0]):
            for col_idx in range(arr.shape[1]):
                value = arr[row_idx, col_idx]
                if np.isnan(value):
                    text = "nan"
                    color = "black"
                else:
                    text = f"{value:.1f}"
                    threshold = 0.55 * max(abs(vmin), abs(vmax))
                    color = "white" if abs(value) > threshold else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=color,
                )

        colorbar = fig.colorbar(
            image,
            ax=ax,
            orientation="horizontal",
            pad=0.14,
            fraction=0.07,
            boundaries=boundaries,
        )
        colorbar.set_ticks(ticks)

    axes[1].set_xlabel("Forecast window (days)", fontsize=14)
    fig.text(
        0.5, -0.02, "(Blue = Better, Red = Worse)", ha="center", va="top", fontsize=12
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig, axes


def plot_batch_delta_panels(
    batch_result: dict[str, Any],
    save_path: Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot delta panels directly from a batch pipeline result dictionary."""
    return plot_multi_model_window_deltas(
        delta_for_plot=batch_result["delta_for_plot"],
        model_order=batch_result["plot_model_order"],
        window_labels=batch_result["window_labels"],
        model_labels=batch_result["plot_model_labels"],
        save_path=save_path,
    )
