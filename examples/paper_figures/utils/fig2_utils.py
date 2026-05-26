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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from examples.paper_figures.plot_config import (
    LARGE_SIZE,
    MEDIUM_SIZE,
    SMALL_SIZE,
    params,
)
from examples.paper_figures.utils.data_utils import YEAR_RANGES, YEAR_RANGES_COM
from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    ProbabilisticOnsetMetrics,
)
from monsoonbench.spatial.regions import (
    points_inside_polygon,
)

MEM_NUMS = {
    "IFS": 11,
    "GenCast": 51,
    "FuXi-S2S": 51,
    "NGCM": 51,
}

def get_day_bins2(max_days) -> list:
    """Helper to generate bins based on forecast window."""
    if max_days == 15:
        return [(1, 6), (6, 11), (11, 16)]
    if max_days == 30:
        return [(1, 6), (6, 11), (11, 16), (16, 21), (21, 26), (26, 31)]
    raise ValueError(f"Unsupported max_forecast_day: {max_days}")

def get_day_bins(max_days) -> list:
    """Helper to generate bins based on forecast window."""
    if max_days == 15:
        return [(1, 5), (6, 10), (11, 15)]
    if max_days == 30:
        return [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
    raise ValueError(f"Unsupported max_forecast_day: {max_days}")

def get_model_tag(model_name) -> str:
    """Returns model label based on input"""
    if "ifs" in model_name.lower():
        return "ifss2s"
    if "fuxi" in model_name.lower():
        return "fuxis2s"
    if "gen" in model_name.lower():
        return "gencast"
    if "ngcm" in model_name.lower():
        return "ngcm"


def generate_heatmap_data(
    config: dict[str, str],
    model_paths: dict[str, str],
    year_ranges: dict[str, np.ndarray],
    max_forecast_day: int = 15
) -> dict:
    """Function to generate heatmap data"""
    # Initialize Metric Computation Classes
    pr = ProbabilisticOnsetMetrics()
    cl = ClimatologyOnsetMetrics()

    print("\n3. Computing climatological dataset...")
    thresh_ds = xr.open_dataset(config["thres_file"])
    thresh_slice = thresh_ds["MWmean"]
    clim_onset = cl.compute_climatological_onset_dataset(
        config["imd_folder"], thresh_slice, years=None, mok=True
    )

    output = {}

    # Create forecast observation dataframe
    print("\n1. Processing forecast model...")
    for model_name, model_fp in model_paths.items():
        # Model-specific parameters
        if model_name.lower() == "fuxi-s2s":
            date_filter_year=2022
        else:
            date_filter_year=2024
        if "ifs" in model_name.lower():
            mem_num=11
        else:
            mem_num=51
        
        
        #Calculate climatological skill scores
        print("\n4. Processing climatological forecasts...")
        climatology_obs_df = cl.multi_year_climatological_forecast_obs_pairs(
            clim_onset,
            target_years=year_ranges[model_name],
            model_forecast_dir=model_fp,
            mem_num=mem_num,
            max_forecast_day=max_forecast_day,
            day_bins=get_day_bins(max_forecast_day),
            date_filter_year=date_filter_year,
            file_pattern=config["file_pattern"],
            mok=True,
        )
        
        #Load and compute forcast observation pairs for each model
        forecast_obs_df = pr.multi_year_forecast_obs_pairs(
            years=year_ranges[model_name],
            imd_folder=config["imd_folder"],
            thres_file=config["thres_file"],
            model_forecast_dir=model_fp,
            mem_num=mem_num,
            max_forecast_day=max_forecast_day,
            day_bins=get_day_bins(max_forecast_day),
            date_filter_year=date_filter_year,
            file_pattern=config["file_pattern"],
            mok=True,
        )

        # Calculate brier and AUC forecast scores from observations
        print("\n2. Calculating brier and AUC forecast scores...")
        brier_forecast = pr.calculate_brier_score(forecast_obs_df)
        rps_forecast = pr.calculate_rps(forecast_obs_df)
        auc_forecast = pr.calculate_auc(forecast_obs_df)
    
        

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

        output[get_model_tag(model_name)] = heatmap_data

    return output


_BIN_NAME_TO_SUFFIX = {
    "Days 1-5": "d1_5",
    "Days 6-10": "d6_10",
    "Days 11-15": "d11_15",
    "Days 16-20": "d16_20",
    "Days 21-25": "d21_25",
    "Days 26-30": "d26_30",
}

_COL_ORDER = [
    "model_label", "horizon",
    "fair_brier_d11_15", "auc_d11_15", "fair_brier_skill_d11_15",
    "fair_brier_d1_5", "auc_d1_5", "fair_brier_skill_d1_5",
    "fair_brier_d6_10", "auc_d6_10", "fair_brier_skill_d6_10",
    "fair_brier_later", "auc_later", "fair_brier_skill_later",
    "fair_brier_d16_20", "auc_d16_20", "fair_brier_skill_d16_20",
    "fair_brier_d21_25", "auc_d21_25", "fair_brier_skill_d21_25",
    "fair_brier_d26_30", "auc_d26_30", "fair_brier_skill_d26_30",
]


def _build_model_row(model_tag: str, data: dict, max_forecast_day: int, later_key: str) -> dict:
    skill_scores = data["skill_results"]["bin_fair_brier_skill_scores"]
    auc_f = data["auc_forecast"]["bin_auc_scores"]
    brier_f = data["brier_forecast"]["bin_fair_brier_scores"]
    brier_c = data["brier_climatology"]["bin_fair_brier_scores"]

    row = {"model_label": model_tag, "horizon": max_forecast_day}
    for bin_name, suffix in _BIN_NAME_TO_SUFFIX.items():
        row[f"fair_brier_{suffix}"] = brier_f.get(bin_name, float("nan"))
        row[f"auc_{suffix}"] = auc_f.get(bin_name, float("nan"))
        row[f"fair_brier_skill_{suffix}"] = skill_scores.get(bin_name, float("nan"))

    later_f = brier_f.get(later_key, float("nan"))
    later_c = brier_c.get(later_key, float("nan"))
    row["fair_brier_later"] = later_f
    row["auc_later"] = auc_f.get(later_key, float("nan"))
    if later_c and not np.isnan(float(later_f)) and not np.isnan(float(later_c)):
        row["fair_brier_skill_later"] = 1.0 - float(later_f) / float(later_c)
    else:
        row["fair_brier_skill_later"] = float("nan")

    return row


def _build_clim_row(data: dict, max_forecast_day: int, later_key: str) -> dict:
    auc_c = data["auc_climatology"]["bin_auc_scores"]
    brier_c = data["brier_climatology"]["bin_fair_brier_scores"]

    row = {"model_label": "clim", "horizon": max_forecast_day}
    for bin_name, suffix in _BIN_NAME_TO_SUFFIX.items():
        row[f"fair_brier_{suffix}"] = brier_c.get(bin_name, float("nan"))
        row[f"auc_{suffix}"] = auc_c.get(bin_name, float("nan"))
        row[f"fair_brier_skill_{suffix}"] = 0.0

    row["fair_brier_later"] = brier_c.get(later_key, float("nan"))
    row["auc_later"] = auc_c.get(later_key, float("nan"))
    row["fair_brier_skill_later"] = 0.0

    return row


def heatmap_data_to_dataframe(
    hm_data: dict,
    max_forecast_day: int,
    include_clim: bool = True,
) -> pd.DataFrame:
    """Convert generate_heatmap_data output to a DataFrame matching monsoon_bin_metrics_*.csv.

    Args:
        hm_data: Output of generate_heatmap_data, keyed by model tag.
        max_forecast_day: 15 or 30 — determines which bins and the 'later' key.
        include_clim: If True, prepend one climatology row (drawn from the first
            model's brier_climatology / auc_climatology). Set False when combining
            multiple horizon DataFrames to avoid duplicate clim rows.
    """
    later_key = f"After day {max_forecast_day}"
    rows = []

    for i, (model_tag, data) in enumerate(hm_data.items()):
        if include_clim and i == 0:
            rows.append(_build_clim_row(data, max_forecast_day, later_key))
        rows.append(_build_model_row(model_tag, data, max_forecast_day, later_key))

    forecast_df = pd.DataFrame(rows)
    cols = [c for c in _COL_ORDER if c in forecast_df.columns]
    return forecast_df[cols]

# =======================================================
# Dual Axis Comparison Code
# =======================================================

def create_probabilistic_model_comparison_table(
    config, 
    model_paths,
    year_ranges,
    max_forecast_day,
) -> pd.DataFrame:
    """Create table with probabilistic scores for each model."""
    pr = ProbabilisticOnsetMetrics()
    cl = ClimatologyOnsetMetrics()

    output = {}

    day_bins = get_day_bins(max_forecast_day)

    # Climatology Baseline
    thresh_ds = xr.open_dataset(config["thres_file"])
    clim_onset = cl.compute_climatological_onset_dataset(
        config["imd_folder"],
        thresh_ds["MWmean"],
        years=None,
        mok=True,
    )

    for model_name, fp in model_paths.items():
        model_forecast_dir = fp
        print(model_forecast_dir)
        if "fuxi" in model_name.lower():
            date_filter_year = 2022
        else:
            date_filter_year = 2024

        baseline_df = cl.multi_year_climatological_forecast_obs_pairs(
            clim_onset,
            year_ranges[model_name],
            day_bins,
            MEM_NUMS[model_name],
            fp,
            max_forecast_day=max_forecast_day,
            mok=True,
            file_pattern=config["file_pattern"],
            date_filter_year=date_filter_year,
        )
        brier_clim = cl.calculate_brier_score_climatology(baseline_df)
        rps_clim = pr.calculate_rps(baseline_df)
        auc_clim = cl.calculate_auc_climatology(baseline_df)

        print(f"Processing {model_name}...")
        forecast_obs_df = pr.multi_year_forecast_obs_pairs(
            years=year_ranges[model_name],
            model_forecast_dir=fp,
            imd_folder=config["imd_folder"],
            thres_file=config["thres_file"],
            mem_num=MEM_NUMS[model_name],
            max_forecast_day=max_forecast_day,
            day_bins=day_bins,
            mok=True,
            file_pattern=config["file_pattern"],
            date_filter_year=date_filter_year,
        )

        brier_forecast = pr.calculate_brier_score(forecast_obs_df)
        rps_forecast = pr.calculate_rps(forecast_obs_df)
        auc_forecast = pr.calculate_auc(forecast_obs_df)
        skill_results = pr.calculate_skill_scores(
            brier_forecast, rps_forecast, brier_clim, rps_clim
        )

        data = {
            "skill_results": skill_results,
            "auc_forecast": auc_forecast,
            "auc_climatology": auc_clim,
            "brier_forecast": brier_forecast,
            "brier_climatology": brier_clim,
            "rps_forecast": rps_forecast,
            "rps_climatology": rps_clim,
        }

        output[get_model_tag(model_name)] = data

    return output


_BASIC_COL_ORDER = [
    "model_label", "horizon",
    "brier", "brier_fair",
    "rps", "rps_fair",
    "auc",
    "fair_brier_skill", "fair_rps_skill",
]


def _build_basic_model_row(model_tag: str, data: dict, max_forecast_day: int) -> dict:
    return {
        "model_label": model_tag,
        "horizon": max_forecast_day,
        "brier": data["brier_forecast"]["brier_score"],
        "brier_fair": data["brier_forecast"]["fair_brier_score"],
        "rps": data["rps_forecast"]["rps"],
        "rps_fair": data["rps_forecast"]["fair_rps"],
        "auc": data["auc_forecast"]["auc"],
        "fair_brier_skill": data["skill_results"]["fair_brier_skill_score"],
        "fair_rps_skill": data["skill_results"]["fair_rps_skill_score"],
    }


def _build_basic_clim_row(data: dict, max_forecast_day: int) -> dict:
    return {
        "model_label": "clim",
        "horizon": max_forecast_day,
        "brier": data["brier_climatology"]["brier_score"],
        "brier_fair": data["brier_climatology"]["fair_brier_score"],
        "rps": data["rps_climatology"]["rps"],
        "rps_fair": data["rps_climatology"]["fair_rps"],
        "auc": data["auc_climatology"]["auc"],
        "fair_brier_skill": 0.0,
        "fair_rps_skill": 0.0,
    }


def model_comparison_to_dataframe(
    comparison_data: dict,
    max_forecast_day: int,
    include_clim: bool = True,
) -> pd.DataFrame:
    """Convert create_probabilistic_model_comparison_table output to a DataFrame

    Output matches monsoon_basic_metrics_*.csv.

    Args:
        comparison_data: Output of create_probabilistic_model_comparison_table,
            keyed by model tag. If a "clim" key is present (from
            compute_standalone_climatology_data), it is used directly and
            include_clim is ignored.
        max_forecast_day: 15 or 30.
        include_clim: If True and no "clim" key exists, prepend one climatology
            row drawn from the first model's embedded brier_climatology /
            rps_climatology / auc_climatology.
    """
    has_standalone_clim = "clim" in comparison_data
    rows = []
    for i, (model_tag, data) in enumerate(comparison_data.items()):
        if include_clim and i == 0 and not has_standalone_clim:
            rows.append(_build_basic_clim_row(data, max_forecast_day))
        rows.append(_build_basic_model_row(model_tag, data, max_forecast_day))

    return pd.DataFrame(rows, columns=_BASIC_COL_ORDER)


# CMZ polygon vertices by resolution, matching multi_year_climatological_forecast_obs_pairs
_CMZ_POLYGONS = {
    1.0: (
        np.array([74, 85, 85, 86, 86, 87, 87, 88, 88, 88, 85, 85, 82, 82, 79, 79, 78, 78, 69, 69, 74, 74]),
        np.array([18, 18, 19, 19, 20, 20, 21, 21, 21, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 21, 21, 18]),
    ),
    2.0: (
        np.array([83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83]),
        np.array([17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17]),
    ),
    4.0: (
        np.array([86, 74, 74, 70, 70, 82, 82, 86, 86]),
        np.array([18, 18, 22, 22, 30, 30, 26, 26, 18]),
    ),
}


def _slice_to_cmz(clim_onset: xr.DataArray) -> xr.DataArray:
    """Filter a clim_onset DataArray to the CMZ region."""
    lat_diff = float(abs(clim_onset.lat.values[1] - clim_onset.lat.values[0]))
    resolution = min(_CMZ_POLYGONS.keys(), key=lambda r: abs(r - lat_diff))
    polygon_lon, polygon_lat = _CMZ_POLYGONS[resolution]
    _, inside_lons, inside_lats = points_inside_polygon(
        polygon_lon, polygon_lat, clim_onset.lon.values, clim_onset.lat.values
    )
    return clim_onset.sel(lat=inside_lats, lon=inside_lons)


def compute_standalone_climatology_data(
    config: dict,
    years: np.ndarray,
    max_forecast_day: int,
    clim_onset: xr.DataArray | None = None,
    date_filter_year: int = 2024,
) -> dict:
    """Compute standalone climatology scores not tied to any model's forecast files.

    Uses get_initialization_dates (all Mon/Thu in May–July) rather than scanning
    a model directory, so the result is a model-agnostic baseline suitable for
    the generic 'clim' row in monsoon_basic_metrics_*.csv.

    Args:
        config: Must contain keys 'imd_folder' and 'thres_file'.
        years: Target years over which to evaluate the climatology.
        max_forecast_day: 15 or 30.
        clim_onset: Pre-computed climatological onset DataArray. If None, it is
            computed from config['imd_folder'] (expensive — pass it in to reuse
            the result from generate_heatmap_data or
            create_probabilistic_model_comparison_table).
        date_filter_year: Year used as the weekday template for initialization
            dates (default 2024).

    Returns:
        Dict in the same format as entries in create_probabilistic_model_comparison_table
        output, with skill_results set to 0. Suitable as the "clim" key in
        comparison_data before calling model_comparison_to_dataframe.
    """
    pr = ProbabilisticOnsetMetrics()
    cl = ClimatologyOnsetMetrics()

    day_bins = get_day_bins(max_forecast_day)

    if clim_onset is None:
        print("Computing climatological onset dataset...")
        thresh_ds = xr.open_dataset(config["thres_file"])
        clim_onset = cl.compute_climatological_onset_dataset(
            config["imd_folder"], thresh_ds["MWmean"], years=None, mok=True
        )

    clim_onset_cmz = _slice_to_cmz(clim_onset)

    all_pairs = []
    for year in years:
        print(f"\nProcessing standalone climatology for year {year}...")
        init_dates = ClimatologyOnsetMetrics.get_initialization_dates(
            year, date_filter_year=date_filter_year
        )
        pairs = ClimatologyOnsetMetrics.create_climatological_forecast_obs_pairs(
            clim_onset=clim_onset_cmz,
            target_year=year,
            init_dates=init_dates,
            day_bins=day_bins,
            max_forecast_day=max_forecast_day,
            mok=True,
        )
        if len(pairs) > 0:
            all_pairs.append(pairs)

    if not all_pairs:
        raise ValueError("No forecast-observation pairs generated for any year.")

    combined = pd.concat(all_pairs, ignore_index=True)

    brier_results = cl.calculate_brier_score_climatology(combined)
    rps_results = pr.calculate_rps(combined)
    auc_results = cl.calculate_auc_climatology(combined)

    return {
        "brier_forecast": brier_results,
        "rps_forecast": rps_results,
        "auc_forecast": auc_results,
        "brier_climatology": brier_results,
        "rps_climatology": rps_results,
        "auc_climatology": auc_results,
        "skill_results": {
            "fair_brier_skill_score": 0.0,
            "fair_rps_skill_score": 0.0,
            "bin_fair_brier_skill_scores": {},
        },
    }


def calculate_reliability_metrics(df, n_bins=10) -> pd.DataFrame:
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


def run_reliability_analysis(
        config,
        model_paths,
        year_ranges,
        max_forecast_day
) -> pd.DataFrame:
    """Main function to run the full monsoon onset reliability pipeline."""
    pr = ProbabilisticOnsetMetrics()

    output = {}

    for model_name, model_fp in model_paths.items():
        if model_name.lower() == "fuxi-s2s":
            date_filter_year=2022
        else:
            date_filter_year=2024
        if "ifs" in model_name.lower():
            mem_num=11
        else:
            mem_num=51

        # Process Data
        forecast_obs_df = pr.multi_year_forecast_obs_pairs(
                years=year_ranges[model_name],
                imd_folder=config["imd_folder"],
                thres_file=config["thres_file"],
                model_forecast_dir=model_fp,
                mem_num=mem_num,
                max_forecast_day=max_forecast_day,
                day_bins=get_day_bins(max_forecast_day),
                date_filter_year=date_filter_year,
                file_pattern=config["file_pattern"],
                mok=True,
            )

        # Filter to regular day-bins only — multi_year_forecast_obs_pairs also
        # generates an "After day N" bin that the original reliability code never
        # produced. Including it inflates pair counts and skews high-probability bins.
        day_bin_df = forecast_obs_df[
            forecast_obs_df["bin_label"].str.startswith("Days ")
        ].copy()

        # Calculate & Display Table
        metrics_df = calculate_reliability_metrics(day_bin_df)

        p_bins = ["[0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]",
            "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]", "(0.8,0.9]", "(0.9,1]"]

        out_df = pd.DataFrame({
            "p_bin": p_bins,
            "pred_mean": metrics_df["Mean_Forecast_Prob"].values,
            "obs_freq": metrics_df["Observed_Frequency"].values,
            "n": metrics_df["N_Forecasts"].values
        })

        print("\nReliability Analysis Results:")
        print(metrics_df.drop(columns=["bin_center"]))  # Hide plotting helper column
        if 2010 in year_ranges[model_name]:
            out_fp = f'{config["output_dir"]}/reliability_{max_forecast_day}d_{get_model_tag(model_name)}_2004_2021_data.csv'
        else:
            out_fp = f'{config["output_dir"]}/reliability_{max_forecast_day}d_{get_model_tag(model_name)}_2019_2024_data.csv'
            
        out_df.to_csv(out_fp, index=False)
        output[get_model_tag(model_name)] = out_df

    return output

def load_fig2_data(config, model_paths: dict[str,str]) -> None:
    """Function for loading data for figure 2"""
    model_paths2 = model_paths.copy()
    output_dir = config["output_dir"]
    include_models = ["fuxi-s2s", "ifs", "gencast", "ngcm"]
    for model in model_paths:
        if model.lower() not in include_models:
            model_paths2.pop(model)
    
    model_paths = model_paths2

    # Plot 2 data (common period)
    comparison_data_15_com = create_probabilistic_model_comparison_table(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES_COM,
        max_forecast_day=15
    )


    #Plot 1 data
    hm_data_15 = generate_heatmap_data(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES,
        max_forecast_day=15
    )

    hm_data_30 = generate_heatmap_data(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES,
        max_forecast_day=30
    )

    df_15 = heatmap_data_to_dataframe(hm_data_15, max_forecast_day=15, include_clim=False)
    df_30 = heatmap_data_to_dataframe(hm_data_30, max_forecast_day=30, include_clim=True)
    df_15_30 = pd.concat([df_15, df_30], ignore_index=True)
    df_15_30.to_csv(f"{output_dir}/monsoon_bin_metrics_2019_2024.csv", index=False, na_rep="NA")

    #Plot 2 data 
    comparison_data_15 = create_probabilistic_model_comparison_table(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES,
        max_forecast_day=15
    )

    clim_data_15 = compute_standalone_climatology_data(
        config=config, years=config["years"], max_forecast_day=15
    )
    comparison_data_15["clim"] = clim_data_15

    comparison_data_30 = create_probabilistic_model_comparison_table(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES,
        max_forecast_day=30
    )

    clim_data_30 = compute_standalone_climatology_data(
        config=config, years=config["years"], max_forecast_day=30
    )
    comparison_data_30["clim"] = clim_data_30
        
    df_15 = model_comparison_to_dataframe(comparison_data_15, max_forecast_day=15)
    df_30 = model_comparison_to_dataframe(comparison_data_30, max_forecast_day=30, include_clim=True)
    df_15_30 = pd.concat([df_15, df_30], ignore_index=True)
    df_15_30.to_csv(f"{output_dir}/monsoon_basic_metrics_2019_2024_new.csv", index=False)
    print(f"Saved basic metrics data to {output_dir}/monsoon_basic_metrics_2019_2024_new.csv")    

    clim_data_15_com = compute_standalone_climatology_data(
        config=config, years=config["common_years"], max_forecast_day=15
    )
    comparison_data_15_com["clim"] = clim_data_15_com

    comparison_data_30_com = create_probabilistic_model_comparison_table(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES_COM,
        max_forecast_day=30
    )

    clim_data_30_com = compute_standalone_climatology_data(
        config=config, years=config["common_years"], max_forecast_day=30
    )
    comparison_data_30_com["clim"] = clim_data_30_com

    df_15_com = model_comparison_to_dataframe(comparison_data_15_com, max_forecast_day=15)
    df_30_com = model_comparison_to_dataframe(comparison_data_30_com, max_forecast_day=30, include_clim=True)
    df_com = pd.concat([df_15_com, df_30_com], ignore_index=True)
    df_com.to_csv(f"{output_dir}/monsoon_basic_metrics_2004_2021_new.csv", index=False)
    print(f"Saved basic metrics data to {output_dir}/monsoon_basic_metrics_2004_2021_new.csv")

    #Reliability data
    run_reliability_analysis(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES,
        max_forecast_day=15
    )
    return 

# =======================================================
# Figure 2 Plotting Code
# =======================================================

def _create_5day_bins_plot(ax1, ax2, df_bins):
    """Create the 5-day binned skill score heatmaps (panels a and b)."""
    from matplotlib.colors import BoundaryNorm

    tick_length = 2
    tick_width = 0.5

    bin_labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30"]
    bin_suffixes = ["d1_5", "d6_10", "d11_15", "d16_20", "d21_25", "d26_30"]
    models = ["clim", "ifss2s", "gencast", "fuxis2s", "ngcm"]
    model_names = {
        "clim": "Climatology", "ifss2s": "IFS*", "fuxis2s": "FuXi-S2S*",
        "ngcm": "NGCM", "gencast": "GenCast",
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
    fbss_boundaries = np.arange(-40, 41, 5)
    fbss_norm = BoundaryNorm(fbss_boundaries, plt.cm.RdBu.N)
    auc_boundaries = np.arange(0.5, 1.025, 0.05)
    auc_norm = BoundaryNorm(auc_boundaries, plt.cm.Blues.N)

    im1 = ax1.imshow(fbss_matrix, aspect="auto", cmap="RdBu",
                     norm=fbss_norm, interpolation="nearest")

    def _fbss_text_color(value):
        if np.isnan(value):
            return "black"
        if abs(value) < 20:
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


def _create_dual_axis_plot(ax, data, title, horizon_days, panel_num, data_dir, model_order):
    """Create a horizontal bar chart comparing BSS, RPSS, and AUC (panels c and d)."""
    import matplotlib.patches as mpatches

    auc_col = np.array([217, 95, 14]) / 256
    rpss_col = np.array([33, 102, 172]) / 256
    bss_col = np.array([146, 197, 222]) / 256

    data_ordered = data.set_index("model_label").reindex(model_order).reset_index()

    df_2004_2021 = pd.read_csv(
        os.path.join(data_dir, "monsoon_basic_metrics_2004_2021_new.csv")
    )
    df_2004_2021 = df_2004_2021[df_2004_2021["model_label"] != "clim_fuxi"]
    df_2004_2021["model_label"] = df_2004_2021["model_label"].replace({
        "clim": "Climatology", "ifss2s": "IFS*", "fuxis2s": "FuXi-S2S*", "ngcm": "NGCM",
    })
    df_2004_2021["fair_brier_skill_pct"] = df_2004_2021["fair_brier_skill"] * 100
    df_2004_2021["fair_rps_skill_pct"] = df_2004_2021["fair_rps_skill"] * 100

    data_2004_2021 = df_2004_2021[df_2004_2021["horizon"] == horizon_days].copy()
    data_2004_2021_ordered = data_2004_2021.set_index("model_label").reindex(model_order).reset_index()

    data_no_clim = data_ordered[data_ordered["model_label"] != "Climatology"]
    data_2004_2021_no_clim = data_2004_2021_ordered[data_2004_2021_ordered["model_label"] != "Climatology"]
    models = data_no_clim["model_label"].values
    y_pos = np.arange(len(models))

    clim_auc = data_ordered[data_ordered["model_label"] == "Climatology"]["auc"].values[0]
    clim_auc_2004_2021 = data_2004_2021_ordered[
        data_2004_2021_ordered["model_label"] == "Climatology"
    ]["auc"].values[0]

    ax2 = ax.twiny()
    height = 0.2

    ax2.barh(y_pos + height, data_no_clim["auc"], height, label="AUC", alpha=0.8, color=auc_col)
    ax.barh(y_pos, data_no_clim["fair_brier_skill_pct"], height, label="BSS", alpha=0.8, color=bss_col)
    ax.barh(y_pos - height, data_no_clim["fair_rps_skill_pct"], height, label="RPSS", alpha=0.8, color=rpss_col)

    ax2.axvline(x=clim_auc, color=auc_col, linestyle="-", linewidth=1.25, alpha=0.8)

    for i, model in enumerate(models):
        if model != "GenCast":
            m_data = data_2004_2021_no_clim[data_2004_2021_no_clim["model_label"] == model]
            if not m_data.empty:
                ax2.plot(m_data["auc"].values[0], y_pos[i] + height,
                         marker="s", zorder=5, markersize=5,
                         markerfacecolor=auc_col, markeredgecolor="black", markeredgewidth=0.75, alpha=0.7)
                ax.plot(m_data["fair_brier_skill_pct"].values[0], y_pos[i],
                        marker="s", markersize=5, color=bss_col,
                        markeredgecolor="black", markeredgewidth=0.75, zorder=5)
                ax.plot(m_data["fair_rps_skill_pct"].values[0], y_pos[i] - height,
                        marker="s", markersize=5, color=rpss_col,
                        markeredgecolor="black", markeredgewidth=0.75, zorder=5)

    ax2.plot(clim_auc_2004_2021, 3.5, marker="s", markersize=5,
             markeredgecolor="black", markeredgewidth=0.75,
             markerfacecolor=auc_col, alpha=0.75, zorder=5, clip_on=False)

    ax.set_xlabel(r"BSS/RPSS (\%)", fontsize=SMALL_SIZE)
    ax.set_title(f"{title} day forecast", fontsize=LARGE_SIZE, fontweight="normal", loc="left")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, rotation=0, ha="right", fontsize=LARGE_SIZE)
    ax.set_xlim(-20, 50)
    ax.set_ylim(-0.5, 3.5)

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
    common_marker = plt.Line2D([0], [0], marker="s", color="black", linestyle="None",
                               markersize=5, markerfacecolor="white",
                               markeredgecolor="black", markeredgewidth=0.75, label="2004-2021")

    if panel_num == 2:
        recent_patch = mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.75,
                                      label="Recent test period\n(2019-2024)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2 + [recent_patch, clim_line, common_marker],
            labels1 + labels2 + ["Recent test period\n(2019-2024)", "Climatology", "Common period\n(2004-2021)"],
            loc="lower right", frameon=False,
        )


def _create_reliability_plot(ax, model_name, horizon_days, col_num, data_dir):
    """Plot reliability diagram for a single model (panel e)."""
    model_file_map = {
        "IFS*": "ifss2s", "GenCast": "gencast", "FuXi-S2S*": "fuxis2s", "NGCM": "ngcm",
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
        f"reliability_{horizon_days}d_{model_file_map[model_name]}_2019_2024_data.csv"
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
        if col_num == 4:
            ax_freq.tick_params(labelright=True)
            ax_freq.set_ylabel("Forecast frequency", fontsize=SMALL_SIZE, rotation=270, labelpad=15)
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


def make_fig2(data_dir, save_path=None) -> plt.Figure:
    """Produce Figure 2: binned skill heatmaps, bar charts, and reliability diagrams.

    Args:
        data_dir: Directory containing generated CSV files
        save_path: If provided, save the figure (writes both .png at 600 dpi and .pdf).
            The extension is stripped and both formats are written.

    Returns:
        The matplotlib Figure object.
    """
    plt.rcParams.update(params)

    metric_df = pd.read_csv(os.path.join(data_dir, "monsoon_basic_metrics_2019_2024_new.csv"))
    df_bins = pd.read_csv(os.path.join(data_dir, "monsoon_bin_metrics_2019_2024.csv"))

    metric_df = metric_df[metric_df["model_label"] != "clim_fuxi"]
    metric_df["model_label"] = metric_df["model_label"].replace({
        "gencast": "GenCast", "clim": "Climatology",
        "ifss2s": "IFS*", "fuxis2s": "FuXi-S2S*", "ngcm": "NGCM",
    })
    metric_df["fair_brier_skill_pct"] = metric_df["fair_brier_skill"] * 100
    metric_df["fair_rps_skill_pct"] = metric_df["fair_rps_skill"] * 100

    model_order = ["NGCM", "FuXi-S2S*", "GenCast", "IFS*", "Climatology"]

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(5, 4, height_ratios=[1.25, -0.05, 0.9, -0.1, 0.8],
                          hspace=0.4, wspace=0.2)

    # Row 1 (gs[0,:]): 5-day binned heatmaps
    ax_bin1 = fig.add_subplot(gs[0, 0:2])
    ax_bin2 = fig.add_subplot(gs[0, 2:4])
    _create_5day_bins_plot(ax_bin1, ax_bin2, df_bins)

    # Row 3 (gs[2,:]): dual-axis bar plots
    ax_c = fig.add_subplot(gs[2, 0:2])
    ax_d = fig.add_subplot(gs[2, 2:4])
    rpss_color = np.array([33, 102, 172]) / 256

    data_15 = metric_df[metric_df["horizon"] == 15].copy()
    data_30 = metric_df[metric_df["horizon"] == 30].copy()
    _create_dual_axis_plot(ax_c, data_15, "(c) 1-15", 15, 1, data_dir, model_order)
    _create_dual_axis_plot(ax_d, data_30, "(d) 1-30", 30, 2, data_dir, model_order)

    ax_c.axvline(x=0, color=rpss_color, linewidth=1.25, alpha=0.8)
    ax_d.axvline(x=0, color=rpss_color, linewidth=1.25, alpha=0.8)

    tick_length = 3
    tick_width = 0.5
    for _ax in (ax_c, ax_d):
        _ax.tick_params(axis="y", which="major", right=False, length=tick_length, width=tick_width)
        _ax.tick_params(axis="x", which="major", bottom=True, length=tick_length, width=tick_width)

    # Row 5 (gs[4,:]): reliability diagrams
    models_for_reliability = ["IFS*", "GenCast", "FuXi-S2S*", "NGCM"]
    for i, model in enumerate(models_for_reliability):
        ax_rel = fig.add_subplot(gs[4, i])
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


def load_fig5_data(config, model_paths, output_dir) -> None:
    """Function for loading data for figure 5"""
    #Plot 1 data

    model_paths2 = model_paths.copy()
    output_dir = config["output_dir"]
    include_models = ["fuxi-s2s", "ifs", "gencast", "ngcm"]
    for model in model_paths:
        if model.lower() not in include_models:
            model_paths2.pop(model)
    
    model_paths = model_paths2

    hm_data_15 = generate_heatmap_data(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES_COM,
        max_forecast_day=15
    )

    hm_data_30 = generate_heatmap_data(
        config=config,
        model_paths=model_paths,
        year_ranges=YEAR_RANGES_COM,
        max_forecast_day=30
    )

    df_15 = heatmap_data_to_dataframe(hm_data_15, max_forecast_day=15, include_clim=False)
    df_30 = heatmap_data_to_dataframe(hm_data_30, max_forecast_day=30, include_clim=True)
    df_15_30 = pd.concat([df_15, df_30], ignore_index=True)
    df_15_30.to_csv(f"{output_dir}/monsoon_bin_metrics_2004_2021.csv", index=False, na_rep="NA")

    return