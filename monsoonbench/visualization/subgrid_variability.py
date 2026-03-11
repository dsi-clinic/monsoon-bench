"""Plotting utilities for subgrid onset-variability map pairs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from monsoonbench.spatial.regions import get_india_outline

__all__ = ["plot_subgrid_variability_map_pair"]


def _degree_formatter_lon(x: float, _pos: float) -> str:
    return f"{int(x)}$^\\circ$E"


def _degree_formatter_lat(x: float, _pos: float) -> str:
    return f"{int(x)}$^\\circ$N"


def _draw_cmz_grid(ax: Axes, edgecolor: str = "black", linewidth: float = 1.0) -> None:
    """Draw CMZ 4-degree box overlay similar to the paper figure."""
    lon_lines = [70, 74, 78, 82, 86]
    lat_lines = [18, 22, 26, 30]

    for lon in lon_lines:
        ax.plot([lon, lon], [18, 30], color=edgecolor, linewidth=linewidth, zorder=5)
    for lat in lat_lines:
        ax.plot([70, 86], [lat, lat], color=edgecolor, linewidth=linewidth, zorder=5)


def plot_subgrid_variability_map_pair(
    variability_ds: xr.Dataset,
    shp_file: Path,
    save_path: Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot side-by-side subgrid-variability maps.

    Args:
        variability_ds: Dataset containing ``panel_a_1deg`` and ``panel_b_025deg``.
        shp_file: Path to India shapefile.
        save_path: Optional output PNG path.

    Returns:
        Tuple of ``(figure, axes)``.
    """
    india_boundaries = get_india_outline(str(shp_file))

    panel_a = variability_ds["panel_a_1deg"]
    panel_b = variability_ds["panel_b_025deg"]

    # Author-like visual style
    map_lw = 0.9
    grid_lw = 1.1
    tick_length = 4
    tick_width = 1.0
    panel_linewidth = 1.0

    # Discrete color bins (as in reference figure)
    bounds = np.arange(0, 33, 3)
    cmap = plt.get_cmap("YlOrBr", len(bounds) - 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)

    # Typography tuned to match notebook reference
    tick_fs = 15
    panel_fs = 20
    cbar_label_fs = 18
    cbar_tick_fs = 14

    fig, axes = plt.subplots(
        1, 2, figsize=(14.8, 6.8), sharex=True, sharey=True, constrained_layout=True
    )
    fig.patch.set_facecolor("white")

    res_a_lat = float(np.median(np.diff(panel_a["lat1"].values)))
    res_a_lon = float(np.median(np.diff(panel_a["lon1"].values)))
    extent_a = [
        float(panel_a["lon1"].min() - res_a_lon / 2),
        float(panel_a["lon1"].max() + res_a_lon / 2),
        float(panel_a["lat1"].min() - res_a_lat / 2),
        float(panel_a["lat1"].max() + res_a_lat / 2),
    ]

    res_b_lat = float(np.median(np.diff(panel_b["lat025"].values)))
    res_b_lon = float(np.median(np.diff(panel_b["lon025"].values)))
    extent_b = [
        float(panel_b["lon025"].min() - res_b_lon / 2),
        float(panel_b["lon025"].max() + res_b_lon / 2),
        float(panel_b["lat025"].min() - res_b_lat / 2),
        float(panel_b["lat025"].max() + res_b_lat / 2),
    ]

    im1 = axes[0].imshow(
        np.ma.masked_invalid(panel_a.values),
        origin="lower",
        extent=extent_a,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )
    im2 = axes[1].imshow(
        np.ma.masked_invalid(panel_b.values),
        origin="lower",
        extent=extent_b,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )
    _ = im1

    for ax in axes:
        ax.set_facecolor("white")
        for india_lon, india_lat in india_boundaries:
            ax.plot(india_lon, india_lat, color="black", linewidth=map_lw, zorder=4)

        _draw_cmz_grid(ax, edgecolor="black", linewidth=grid_lw)

        ax.set_xlim(66.5, 100.5)
        ax.set_ylim(6.0, 38.5)

        yticks = np.arange(10, 36, 5)
        xticks = np.arange(70, 101, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.yaxis.set_major_formatter(FuncFormatter(_degree_formatter_lat))
        ax.xaxis.set_major_formatter(FuncFormatter(_degree_formatter_lon))

        ax.grid(False)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=tick_fs,
            length=tick_length,
            width=tick_width,
        )
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_linewidth(panel_linewidth)

    axes[1].set_yticklabels([])
    axes[0].text(
        0.015,
        0.98,
        "(a)",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=panel_fs,
    )
    axes[1].text(
        0.015,
        0.98,
        "(b)",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=panel_fs,
    )

    cbar = fig.colorbar(
        im2,
        ax=axes.ravel().tolist(),
        shrink=0.82,
        pad=0.02,
        boundaries=bounds,
        ticks=[0, 6, 12, 18, 24, 30],
        extend="max",
    )
    cbar.set_label("days", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs, length=3, width=1)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, facecolor="white", bbox_inches="tight")

    return fig, axes
