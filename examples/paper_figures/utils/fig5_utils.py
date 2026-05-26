"""Functions used to generate figure 5"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples.paper_figures.plot_config import (
    LARGE_SIZE,
    MEDIUM_SIZE,
    SMALL_SIZE,
    params,
)
from examples.paper_figures.utils.data_utils import YEAR_RANGES_COM
from examples.paper_figures.utils.fig2_utils import (
    generate_heatmap_data,
    heatmap_data_to_dataframe,
    run_reliability_analysis,
)

# ==============================================
# Data Loading
# ==============================================

def load_fig5_data(config, model_paths) -> None:
    """Loads data for figure 5.
    
    Fuxi-s2s, ifs, ngcm
    """
    model_paths2 = model_paths.copy()
    output_dir = config["output_dir"]
    include_models = ["fuxi-s2s", "ifs", "ngcm"]
    for model in list(model_paths):
        if model.lower() not in include_models:
            model_paths2.pop(model)

    model_paths = model_paths2

    hm_data_15 = generate_heatmap_data(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES_COM,
        max_forecast_day=15,
    )
    hm_data_30 = generate_heatmap_data(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES_COM,
        max_forecast_day=30,
    )

    df_15 = heatmap_data_to_dataframe(hm_data_15, max_forecast_day=15, include_clim=False)
    df_30 = heatmap_data_to_dataframe(hm_data_30, max_forecast_day=30, include_clim=True)
    df_15_30 = pd.concat([df_15, df_30], ignore_index=True)
    df_15_30.to_csv(f"{output_dir}/monsoon_bin_metrics_2004_2021.csv", index=False, na_rep="NA")

    run_reliability_analysis(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES_COM,
        max_forecast_day=15,
    )
    return 


# ==============================================
# Figure 5 Plotting Code
# ==============================================

def _create_5day_bins_plot(ax1, ax2, df_bins):
    """Create the 5-day binned skill score heatmaps (panels a and b)."""
    from matplotlib.colors import BoundaryNorm

    tick_length = 2
    tick_width = 0.5

    bin_labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30"]
    bin_suffixes = ["d1_5", "d6_10", "d11_15", "d16_20", "d21_25", "d26_30"]
    models = ["clim", "ifss2s", "fuxis2s", "ngcm"]
    model_names = {
        "clim": "Climatology",
        "ifss2s": "IFS",
        "fuxis2s": r"FuXi-S2S$^{\dagger}$",
        "ngcm": r"NGCM$^{\dagger}$",
    }

    fbss_matrix = np.full((len(models), len(bin_labels)), np.nan)
    auc_matrix = np.full((len(models), len(bin_labels)), np.nan)

    for i, model in enumerate(models):
        for j, bin_suffix in enumerate(bin_suffixes):
            model_data = df_bins[(df_bins["model_label"] == model) & (df_bins["horizon"] == 30)]
            fbss_col = f"fair_brier_skill_{bin_suffix}"
            auc_col = f"auc_{bin_suffix}"
            if fbss_col in df_bins.columns and len(model_data) > 0:
                value = model_data[fbss_col].iloc[0]
                if pd.notna(value):
                    fbss_matrix[i, j] = value * 100
            if auc_col in df_bins.columns and len(model_data) > 0:
                value = model_data[auc_col].iloc[0]
                if pd.notna(value):
                    auc_matrix[i, j] = value

    model_labels = [model_names[m] for m in models]
    fbss_boundaries = np.arange(-20, 21, 2.5)
    fbss_norm = BoundaryNorm(fbss_boundaries, plt.cm.RdBu.N)
    auc_boundaries = np.arange(0.5, 1.025, 0.05)
    auc_norm = BoundaryNorm(auc_boundaries, plt.cm.Blues.N)

    im1 = ax1.imshow(fbss_matrix, aspect="auto", cmap="RdBu",
                     norm=fbss_norm, interpolation="nearest")

    def _fbss_text_color(value):
        if np.isnan(value):
            return "black"
        if abs(value) < 12.5:
            return "black"
        return "white"

    for i in range(len(models)):
        for j in range(len(bin_labels)):
            if not np.isnan(fbss_matrix[i, j]):
                color = _fbss_text_color(fbss_matrix[i, j])
                label = "0" if (models[i] == "clim" and fbss_matrix[i, j] == 0.0) \
                    else f"{fbss_matrix[i, j]:.1f}"
                ax1.text(j, i, label, ha="center", va="center", color=color, fontsize=MEDIUM_SIZE)

    ax1.set_title(r"(a) Brier Skill Score (\%)", fontweight="normal", pad=5, fontsize=LARGE_SIZE)
    ax1.set_xlabel("Forecast window (days)", fontweight="normal", fontsize=MEDIUM_SIZE)
    ax1.set_xticks(range(len(bin_labels)))
    ax1.set_xticklabels(bin_labels)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(model_labels)
    ax1.tick_params(axis="y", which="major", labelsize=LARGE_SIZE, direction="out")
    ax1.tick_params(axis="x", which="major", labelsize=LARGE_SIZE)

    fbss_tick_values = fbss_boundaries[::4]
    cbar1 = plt.colorbar(im1, ax=ax1, orientation="horizontal", fraction=0.06, pad=0.2,
                         shrink=0.75, aspect=20, boundaries=fbss_boundaries, ticks=fbss_tick_values)
    cbar1.minorticks_off()

    im2 = ax2.imshow(auc_matrix, aspect="auto", cmap="Blues",
                     norm=auc_norm, interpolation="nearest")

    for i in range(len(models)):
        for j in range(len(bin_labels)):
            if not np.isnan(auc_matrix[i, j]):
                color = "white" if auc_matrix[i, j] >= 0.8 else "black"
                ax2.text(j, i, f"{auc_matrix[i, j]:.2f}", ha="center", va="center",
                         color=color, fontsize=MEDIUM_SIZE)

    ax2.set_title("(b) AUC", fontweight="normal", pad=5, fontsize=LARGE_SIZE)
    ax2.set_xlabel("Forecast window (days)", fontweight="normal", fontsize=MEDIUM_SIZE)
    ax2.set_xticks(range(len(bin_labels)))
    ax2.set_xticklabels(bin_labels)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([])

    auc_tick_values = auc_boundaries[::2]
    cbar2 = plt.colorbar(im2, ax=ax2, orientation="horizontal", fraction=0.06, pad=0.2,
                         shrink=0.75, aspect=20, boundaries=auc_boundaries, ticks=auc_tick_values)
    cbar2.minorticks_off()

    ax1.tick_params(axis="both", which="major", length=tick_length, width=tick_width)
    ax1.tick_params(axis="x", which="major", top=False)
    ax2.tick_params(axis="both", which="major", length=tick_length, width=tick_width)
    ax2.tick_params(axis="x", which="major", top=False, labelsize=LARGE_SIZE)


def _create_dual_axis_plot(ax, data, title, panel_num, model_order):
    """Create a horizontal bar chart comparing BSS, RPSS, and AUC (panels c and d)."""
    auc_col = np.array([217, 95, 14]) / 256
    rpss_col = np.array([33, 102, 172]) / 256
    bss_col = np.array([146, 197, 222]) / 256

    data_ordered = data.set_index("model_label").reindex(model_order).reset_index()
    data_no_clim = data_ordered[data_ordered["model_label"] != "Climatology"]
    models = data_no_clim["model_label"].values
    y_pos = np.arange(len(models))

    clim_auc = data_ordered[data_ordered["model_label"] == "Climatology"]["auc"].values[0]

    ax2 = ax.twiny()
    height = 0.2

    ax2.barh(y_pos + height, data_no_clim["auc"], height, label="AUC", alpha=0.8, color=auc_col)
    ax.barh(y_pos, data_no_clim["fair_brier_skill_pct"], height, label="BSS", alpha=0.8, color=bss_col)
    ax.barh(y_pos - height, data_no_clim["fair_rps_skill_pct"], height, label="RPSS", alpha=0.8, color=rpss_col)

    ax2.axvline(x=clim_auc, color=auc_col, linestyle="-", linewidth=1.25, alpha=0.8)

    ax.set_xlabel(r"BSS/RPSS (\%)", fontsize=SMALL_SIZE)
    ax.set_title(f"{title} day forecast", fontsize=LARGE_SIZE, fontweight="normal", loc="left")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, rotation=0, ha="right", fontsize=LARGE_SIZE)
    ax.set_xlim(-20, 50)
    ax.set_ylim(-0.5, 2.5)

    ax2.set_xlabel("AUC", fontsize=SMALL_SIZE, color=auc_col)
    ax2.tick_params(axis="x", colors=auc_col)
    ax2.spines["top"].set_color(auc_col)
    ax2.set_xlim(0.8, 1.0)
    ax2.set_xticks(np.arange(0.8, 1.02, 0.05))

    if panel_num == 1:
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True)
        ax2.tick_params(labelbottom=False, labeltop=True)
    else:
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=False)
        ax2.tick_params(labeltop=True)

    clim_line = plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.25, label="Climatology")

    if panel_num == 2:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2 + [clim_line],
            labels1 + labels2 + ["Climatology"],
            loc="lower right", frameon=False,
        )


def _create_reliability_plot(ax, model_name, horizon_days, col_num, data_dir):
    """Plot reliability diagram for a single model (panel e)."""
    model_file_map = {
        "IFS": "ifss2s",
        r"FuXi-S2S$^{\dagger}$": "fuxis2s",
        r"NGCM$^{\dagger}$": "ngcm",
    }

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])

    if model_name not in model_file_map:
        ax.text(0.5, 0.5, f"No data for\n{model_name}",
                transform=ax.transAxes, ha="center", va="center")
        ax.text(0.05, 0.95, model_name, transform=ax.transAxes,
                fontsize=MEDIUM_SIZE, fontweight="normal", va="top", ha="left")
        if col_num > 1:
            ax.tick_params(labelleft=False)
        return

    file_path = os.path.join(
        data_dir,
        f"reliability_{horizon_days}d_{model_file_map[model_name]}_2004_2021_data.csv"
    )

    try:
        rel_data = pd.read_csv(file_path)
        error_bars = np.sqrt(rel_data["obs_freq"] * (1 - rel_data["obs_freq"]) / rel_data["n"])

        ax.errorbar(rel_data["pred_mean"], rel_data["obs_freq"], yerr=error_bars,
                    fmt="o-", color="blue", markersize=2, linewidth=1, capsize=2,
                    capthick=1, elinewidth=1, label="Reliability")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=1, label="Perfect reliability")

        ax_freq = ax.twinx()
        ax_freq.bar(rel_data["pred_mean"], rel_data["n"] / rel_data["n"].sum(),
                    width=0.05, alpha=0.3, color="gray")
        ax_freq.set_yscale("log")
        ax_freq.set_ylim(0.001, 1)
        if col_num == 3:
            ax_freq.tick_params(labelright=True)
            ax_freq.set_ylabel("Forecast frequency", fontsize=SMALL_SIZE, rotation=270, labelpad=5)
        else:
            ax_freq.tick_params(labelright=False)

        ax.set_xlabel("Forecast Probability", fontsize=SMALL_SIZE)
        if col_num == 1:
            ax.set_ylabel("Observed Frequency", fontsize=SMALL_SIZE)

        ax.text(0.05, 0.95, model_name, transform=ax.transAxes,
                fontsize=MEDIUM_SIZE, fontweight="normal", va="top", ha="left")
        if col_num > 1:
            ax.tick_params(labelleft=False)
        ax.grid(True, alpha=0.3)

    except FileNotFoundError:
        ax.text(0.5, 0.5, f"File not found:\n{file_path}",
                transform=ax.transAxes, ha="center", va="center", fontsize=SMALL_SIZE)
        ax.text(0.05, 0.95, model_name, transform=ax.transAxes,
                fontsize=MEDIUM_SIZE, fontweight="normal", va="top", ha="left")
        if col_num > 1:
            ax.tick_params(labelleft=False)


def make_fig5(data_dir, save_path=None) -> plt.figure:
    """Produce Figure 5
    
    Binned skill heatmaps, bar charts, and reliability diagrams
    for the common 2004–2021 period (IFS, FuXi-S2S, NGCM only).

    Args:
        data_dir: Root directory containing 'output/' and 'fig_data/' subdirectories.
        save_path: If provided, save the figure (writes both .png at 600 dpi and .pdf).
            The extension is stripped and both formats are written.

    Returns:
        The matplotlib Figure object.
    """
    plt.rcParams.update(params)

    metric_df = pd.read_csv(os.path.join(data_dir, "monsoon_basic_metrics_2004_2021_new.csv"))
    df_bins = pd.read_csv(os.path.join(data_dir, "monsoon_bin_metrics_2004_2021.csv"))

    metric_df = metric_df[metric_df["model_label"] != "clim_fuxi"]
    metric_df["model_label"] = metric_df["model_label"].replace({
        "gencast": "GenCast",
        "clim": "Climatology",
        "ifss2s": "IFS",
        "fuxis2s": r"FuXi-S2S$^{\dagger}$",
        "ngcm": r"NGCM$^{\dagger}$",
    })
    metric_df["fair_brier_skill_pct"] = metric_df["fair_brier_skill"] * 100
    metric_df["fair_rps_skill_pct"] = metric_df["fair_rps_skill"] * 100

    model_order = [r"NGCM$^{\dagger}$", r"FuXi-S2S$^{\dagger}$", "IFS", "Climatology"]

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(5, 6, height_ratios=[1.25, -0.05, 0.9, -0.1, 0.8],
                          hspace=0.4, wspace=0.5)

    # Row 1 (gs[0,:]): 5-day binned heatmaps
    ax_bin1 = fig.add_subplot(gs[0, 0:3])
    ax_bin2 = fig.add_subplot(gs[0, 3:6])
    _create_5day_bins_plot(ax_bin1, ax_bin2, df_bins)

    # Row 3 (gs[2,:]): dual-axis bar plots
    ax_c = fig.add_subplot(gs[2, 0:3])
    ax_d = fig.add_subplot(gs[2, 3:6])
    rpss_color = np.array([33, 102, 172]) / 256

    data_15 = metric_df[metric_df["horizon"] == 15].copy()
    data_30 = metric_df[metric_df["horizon"] == 30].copy()
    _create_dual_axis_plot(ax_c, data_15, "(c) 1-15", 1, model_order)
    _create_dual_axis_plot(ax_d, data_30, "(d) 1-30", 2, model_order)

    ax_c.axvline(x=0, color=rpss_color, linewidth=1.25, alpha=0.8)
    ax_d.axvline(x=0, color=rpss_color, linewidth=1.25, alpha=0.8)

    tick_length = 3
    tick_width = 0.5
    for _ax in (ax_c, ax_d):
        _ax.tick_params(axis="y", which="major", right=False, length=tick_length, width=tick_width)
        _ax.tick_params(axis="x", which="major", bottom=True, length=tick_length, width=tick_width)

    # Row 5 (gs[4,:]): reliability diagrams — 3 panels each spanning 2 grid columns
    models_for_reliability = ["IFS", r"FuXi-S2S$^{\dagger}$", r"NGCM$^{\dagger}$"]
    for i, model in enumerate(models_for_reliability):
        ax_rel = fig.add_subplot(gs[4, i * 2:(i + 1) * 2])
        ax_rel.tick_params(axis="y", length=tick_length, width=tick_width)
        ax_rel.tick_params(axis="x", length=tick_length, width=tick_width)
        if i == 0:
            ax_rel.text(0.02, 1.1, "(e) Reliability of 1-15 day forecast",
                        transform=ax_rel.transAxes, fontsize=LARGE_SIZE,
                        va="top", ha="left", fontweight="normal")
        _create_reliability_plot(ax_rel, model, 15, i + 1, data_dir)

    plt.tight_layout()

    if save_path is not None:
        base = Path(save_path).with_suffix("")
        plt.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
        plt.savefig(f"{base}.pdf", bbox_inches="tight")

    return fig
