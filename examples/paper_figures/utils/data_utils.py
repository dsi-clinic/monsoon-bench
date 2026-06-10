"""Utility functions for paper figures 1 & 4."""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import savemat

from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    ProbabilisticOnsetMetrics,
)
from monsoonbench.visualization import (
    create_model_comparison_table,
)

# Figure 1 year ranges
YEAR_RANGES = {
    "AIFS": np.arange(2019, 2025),
    "IFS": np.arange(2019, 2024),
    "FuXi": np.arange(2019, 2025),
    "Graphcast": np.arange(2019, 2025),
    "GenCast": np.arange(2019, 2025),
    "FuXi-S2S": np.arange(2019, 2022),
    "NGCM": np.arange(2019, 2025),
}

EXTENDED_YEARS = {
    "AIFS": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
    "IFS": np.arange(2019, 2024),
    "FuXi": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
    "Graphcast": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
    "GenCast": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
    "FuXi-S2S": np.arange(2019, 2022),
    "NGCM": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
}

# Figure 4 year ranges
YEAR_RANGES_COM = {
    "AIFS": np.arange(2004, 2022),
    "IFS": np.arange(2004, 2022),
    "FuXi": np.arange(2004, 2022),
    "Graphcast": np.arange(2004, 2022),
    "FuXi-S2S": np.arange(2004, 2022),
    "NGCM": np.arange(2004, 2022),
}

def get_model_dfs(
    model_paths: dict[str, str],
    year_ranges: dict[str, list[int]],
    config: dict[str, str],
    days: int = 15,
) -> tuple[dict[str, pd.DataFrame], dict[str, xr.DataArray]]:
    """Get model dataframes and onset data arrays for a given set of model paths, year ranges, and forecast period.

    Args:
        model_paths: Dictionary of model names and their file paths.
        year_ranges: Dictionary of model names and their year ranges.
        config: Dictionary of configuration parameters.
        days: Number of days to forecast (15 or 30).
    """
    p_metrics = ProbabilisticOnsetMetrics()
    d_metrics = DeterministicOnsetMetrics()

    model_dfs = {}
    model_onsets = {}

    # Validate verification window and tolerance days
    if days == 15:
        tol_days = 3
        ver_window = 1
    elif days == 30:
        tol_days = 5
        ver_window = 16
    else:
        raise ValueError(f"Invalid forecast period: {days}. Must be 15 or 30")

    # Compute metrics for each model
    for model_name, model_fp in model_paths.items():
        if model_name not in year_ranges:
            continue
        if model_name.lower() == "fuxi-s2s":
            date_filter_year=2022
        else:
            date_filter_year=2024
        try:
            model_df, onset_da_dict = p_metrics.compute_metrics_multiple_years(
                years=year_ranges[model_name],
                imd_folder=config["imd_folder"],
                thres_file=config["thres_file"],
                model_forecast_dir=model_fp,
                tolerance_days=tol_days,
                verification_window=ver_window,
                forecast_days=days,
                max_forecast_day=days,
                mok=True,
                onset_window=5,
                mok_month=6,
                mok_day=2,
                date_filter_year=date_filter_year,
            )
        except Exception:
            model_df, onset_da_dict = d_metrics.compute_metrics_multiple_years(
                years=year_ranges[model_name],
                imd_folder=config["imd_folder"],
                thres_file=config["thres_file"],
                model_forecast_dir=model_fp,
                tolerance_days=tol_days,
                verification_window=ver_window,
                forecast_days=days,
                max_forecast_day=days,
                mok=True,
                onset_window=5,
                mok_month=6,
                mok_day=2,
                date_filter_year=date_filter_year,
            )

        model_dfs[model_name] = model_df
        model_onsets[model_name] = onset_da_dict

    return model_dfs, model_onsets

def get_plot_metrics(
    model_dfs: dict[str, pd.DataFrame],
    model_onsets: dict[str, xr.DataArray],
    metrics_df_clim: pd.DataFrame,
    onset_da_clim: xr.DataArray,
    year_range: list[int],
    day: int = 15,
) -> dict[str, np.ndarray]:
    """Get Figure 1 & 4 plot metrics for a given set of model

    dataframes, onset data arrays, climatological data, and configuration.

    Args:
        model_dfs: Dictionary of model names and their individual metric dataframes.
        model_onsets: Dictionary of model names and their onset data arrays.
        metrics_df_clim: Climatological metrics dataframe.
        onset_da_clim: Climatological onset data array.
        year_range: Range of years to get data for.
        day: Number of days to forecast (15 or 30).

    Returns:
        Dictionary of plot metrics.
    """
    plot_metrics = {}
    c_metrics = ClimatologyOnsetMetrics()

    for model_name in model_dfs.keys():
        probabilistic_df = model_dfs[model_name]
        onset_da_dict = model_onsets[model_name]
        plot_probabilistic_metrics = c_metrics.create_spatial_far_mr_mae(
            probabilistic_df, onset_da_dict
        )
        plot_metrics[model_name] = plot_probabilistic_metrics

    clim_plot_data = c_metrics.create_spatial_far_mr_mae(
        metrics_df_clim, dict.fromkeys(year_range, onset_da_clim)
    )

    mean_mae = plot_metrics[model_name]["mean_mae"]

    # Get coordinates
    lats = mean_mae.lat.to_numpy()
    lons = mean_mae.lon.to_numpy()

    false_alarm_15 = [clim_plot_data["false_alarm_rate"]]
    miss_rate_15 = [clim_plot_data["miss_rate"]]
    mae_yr_15 = [clim_plot_data["mean_mae"]]

    for model_name in plot_metrics.keys():
        for stat in plot_metrics[model_name].keys():
            if "miss_rate" in stat:
                miss_rate_15.append(plot_metrics[model_name][stat])
            if "mean_mae" in stat:
                mae_yr_15.append(plot_metrics[model_name][stat])
            if "false" in stat:
                false_alarm_15.append(plot_metrics[model_name][stat])

    mae_yr_15 = xr.concat(mae_yr_15, dim="stack").transpose()
    miss_rate_15 = xr.concat(miss_rate_15, dim="stack").transpose()
    false_alarm_15 = xr.concat(false_alarm_15, dim="stack").transpose()
    plot_metrics["clim"] = clim_plot_data

    cmz_metrics = create_model_comparison_table(plot_metrics)

    cmz_metrics = pd.concat([cmz_metrics.tail(1), cmz_metrics.iloc[:-1]])

    # Format output dictionary
    mat_dict = {
        f"false_alarm_{str(day)}": false_alarm_15.values,
        f"far_cmz_mean_{str(day)}": np.array(cmz_metrics["cmz_far_pct"].values),
        "lat": lats,
        "lon": lons,
        f"mae_avg_{str(day)}": mae_yr_15.values,
        "mae_cmz_fixed_clim": np.array(
            [[7.18333333]]
        ),  # Fixed climatological MAE for 15-day and 30-day forecasts
        f"mae_cmz_mean_{str(day)}": np.array(cmz_metrics["cmz_mae_mean_days"].values),
        f"mae_yr_{str(day)}": np.array(cmz_metrics["overall_mae_mean_days"].values),
        f"miss_rate_{str(day)}": miss_rate_15.values,
        f"mr_cmz_mean_{str(day)}": np.array(cmz_metrics["cmz_mr_pct"].values),
        f"std_er_{str(day)}": np.array(cmz_metrics["cmz_mae_se_days"].values),
        "std_er_fixed_clim": np.array(
            [[0.9686474]]
        ),  # Fixed climatological standard error for 15-day and 30-day forecasts
    }

    return mat_dict


def get_climatological_dfs(
    config: dict[str, str],
    date_filter_year: int = 2024,
) -> dict[str, tuple[pd.DataFrame, xr.DataArray]]:
    """Get climatological dataframes and onset data arrays for all year ranges.

    Args:
        config: Dictionary of configuration parameters.
        date_filter_year: Year to filter initialization dates by 

    Returns:
        Dictionary of climatological dataframes and onset data arrays.
    """
    c_metrics = ClimatologyOnsetMetrics()

    # Compute 15-day forecast data
    clim_df_15, clim_onset_15 = c_metrics.compute_climatology_baseline_multiple_years(
        years=config["years"],
        imd_folder=config["imd_folder"],
        thres_file=config["thres_file"],
        tolerance_days=3,
        verification_window=1,
        forecast_days=15,
        max_forecast_day=15,
        mok=True,
        onset_window=5,
        mok_month=6,
        mok_day=2,
        date_filter_year=date_filter_year,
    )

    # Compute 15-day forecast data for the extended period
    clim_df_15_ex, clim_onset_15_ex = (
        c_metrics.compute_climatology_baseline_multiple_years(
            years=np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
            imd_folder=config["imd_folder"],
            thres_file=config["thres_file"],
            tolerance_days=3,
            verification_window=1,
            forecast_days=15,
            max_forecast_day=15,
            mok=True,
            onset_window=5,
            mok_month=6,
            mok_day=2,
            date_filter_year=date_filter_year,
        )
    )

    # Compute 30-day forecast data
    clim_df_30, clim_onset_30 = c_metrics.compute_climatology_baseline_multiple_years(
        years=config["years"],
        imd_folder=config["imd_folder"],
        thres_file=config["thres_file"],
        tolerance_days=5,
        verification_window=16,
        forecast_days=30,
        max_forecast_day=30,
        mok=True,
        onset_window=5,
        mok_month=6,
        mok_day=2,
        date_filter_year=date_filter_year,
    )

    # Compute 30-day forecast data for the extended period
    clim_df_30_ex, clim_onset_30_ex = (
        c_metrics.compute_climatology_baseline_multiple_years(
            years=np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
            imd_folder=config["imd_folder"],
            thres_file=config["thres_file"],
            tolerance_days=5,
            verification_window=16,
            forecast_days=30,
            max_forecast_day=30,
            mok=True,
            onset_window=5,
            mok_month=6,
            mok_day=2,
            date_filter_year=date_filter_year,
        )
    )

    # Compute 15-day forecast data for the common period
    clim_df_15_cm, clim_onset_15_cm = (
        c_metrics.compute_climatology_baseline_multiple_years(
            years=np.arange(2004,2022),
            imd_folder=config["imd_folder"],
            thres_file=config["thres_file"],
            tolerance_days=3,
            verification_window=1,
            forecast_days=15,
            max_forecast_day=15,
            mok=True,
            onset_window=5,
            mok_month=6,
            mok_day=2,
            date_filter_year=date_filter_year,
        )
    )

    # Compute 30-day forecast data for the common period
    clim_df_30_cm, clim_onset_30_cm = (
        c_metrics.compute_climatology_baseline_multiple_years(
            years=np.arange(2004,2022),
            imd_folder=config["imd_folder"],
            thres_file=config["thres_file"],
            tolerance_days=5,
            verification_window=16,
            forecast_days=30,
            max_forecast_day=30,
            mok=True,
            onset_window=5,
            mok_month=6,
            mok_day=2,
            date_filter_year=date_filter_year,
        )
    )

    output = {
        "15_day": (clim_df_15, clim_onset_15),
        "15_day_ex": (clim_df_15_ex, clim_onset_15_ex),
        "15_day_cm": (clim_df_15_cm, clim_onset_15_cm),
        "30_day": (clim_df_30, clim_onset_30),
        "30_day_ex": (clim_df_30_ex, clim_onset_30_ex),
        "30_day_cm": (clim_df_30_cm, clim_onset_30_cm),
    }

    return output


def save_data(mat_dict: dict[str, np.ndarray], output_dir: str, save_path: str) -> None:
    """Save data to a .mat file.

    Args:
        mat_dict: Dictionary of data to save.
        output_dir: Directory to save the data.
        save_path: Path to save the data.
    """
    out_path = f"{output_dir}/{save_path}.mat"
    savemat(out_path, mat_dict)
    print("Saved to:", out_path)
    return

def load_fig_1_4_data(model_paths: dict[str, str], config: dict[str, str]) -> None:
    """Load data for figures 1 and 4.

    Args:
        model_paths: Dictionary of model paths.
        config: Dictionary of configuration parameters.
    """
    print("Loading Climatoligcal Data")
    clim_data = get_climatological_dfs(config=config)
    clim_df_15, clim_onset_15 = clim_data["15_day"]
    clim_df_15_ex, clim_onset_15_ex = clim_data["15_day_ex"]
    clim_df_30, clim_onset_30 = clim_data["30_day"]
    clim_df_30_ex, clim_onset_30_ex = clim_data["30_day_ex"]
    clim_df_15_cm, clim_onset_15_cm = clim_data["15_day_cm"]
    clim_df_30_cm, clim_onset_30_cm = clim_data["30_day_cm"]

    print("Loading Model Data (2019-2024)...")
    model_dfs_15, model_onsets_15 = get_model_dfs(
        model_paths, year_ranges=YEAR_RANGES, config=config, days=15
    )
    md_15 = get_plot_metrics(
        model_dfs_15, model_onsets_15, clim_df_15,
        clim_onset_15, config["years"], 15
    )
    save_data(md_15, config["output_dir"], "deterministic_scores_15_day_2019_2024")

    # Compute and save 30-day forecast metrics for 2019-2024
    model_dfs_30, model_onsets_30 = get_model_dfs(
        model_paths, year_ranges=YEAR_RANGES, config=config, days=30
    )
    md_30 = get_plot_metrics(
        model_dfs_30, model_onsets_30, clim_df_30,
        clim_onset_30, config["years"], 30
    )
    save_data(md_30, config["output_dir"], "deterministic_scores_30_day_2019_2024")

    print("Loading Model Data (extended period)...")
    # Compute and save 15-day forecast metrics for extended period
    model_dfs_15_ex, model_onsets_15_ex = get_model_dfs(
        model_paths, year_ranges=EXTENDED_YEARS, config=config, days=15
    )
    md_15_ex = get_plot_metrics(
        model_dfs_15_ex, model_onsets_15_ex, clim_df_15_ex,
        clim_onset_15_ex, config["extended_years"], 15
    )
    save_data(
        md_15_ex,
        config["output_dir"],
        "deterministic_scores_15_day_1965_1978_2019_2024_with_gencast",
    )

    # Compute and save 30-day forecast metrics for extended period
    model_dfs_30_ex, model_onsets_30_ex = get_model_dfs(
        model_paths, year_ranges=EXTENDED_YEARS, config=config, days=30
    )
    md_30_ex = get_plot_metrics(
        model_dfs_30_ex, model_onsets_30_ex, clim_df_30_ex,
        clim_onset_30_ex, config["extended_years"], 30
    )
    save_data(
        md_30_ex,
        config["output_dir"],
        "deterministic_scores_30_day_1965_1978_2019_2024_with_gencast",
    )

    print("Loading Model Data (2004-2021)...")
    # Compute and save 15-day forecast metrics for common period
    model_dfs_15_cm, model_onsets_15_cm = get_model_dfs(
        model_paths, year_ranges=YEAR_RANGES_COM, config=config, days=15
    )
    md_15_cm = get_plot_metrics(
        model_dfs_15_cm, model_onsets_15_cm, clim_df_15_cm,
        clim_onset_15_cm, config["common_years"], 15
    )
    save_data(md_15_cm, config["output_dir"], "deterministic_scores_15_day_2004_2021")
    
    # Compute and save 30-day forecast metrics for common period
    model_dfs_30_cm, model_onsets_30_cm = get_model_dfs(
        model_paths, year_ranges=YEAR_RANGES_COM, config=config, days=30
    )
    md_30_cm = get_plot_metrics(
        model_dfs_30_cm, model_onsets_30_cm, clim_df_30_cm,
        clim_onset_30_cm, config["common_years"], 30
    )    
    save_data(md_30_cm, config["output_dir"], "deterministic_scores_30_day_2004_2021")
    return
