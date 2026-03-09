"""Utility functions for paper figures 1 & 4."""

import numpy as np
import pandas as pd
from scipy.io import savemat
import xarray as xr

from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    ProbabilisticOnsetMetrics,
)
from monsoonbench.visualization import (
    create_model_comparison_table,
)


def get_model_dfs(
    model_paths: dict[str, str],
    year_ranges: dict[str, list[int]],
    config: dict[str, str],
    days: int = 15,
) -> tuple[dict[str, pd.DataFrame], dict[str, xr.DataArray]]:
    """
    Get model dataframes and onset data arrays for a given set of model paths, year ranges, and forecast period.

    Args:
        model_paths: Dictionary of model names and their file paths.
        year_ranges: Dictionary of model names and their year ranges.
        config: Dictionary of configuration parameters.
        days: Number of days to forecast (15 or 30).
    """
    metrics = ProbabilisticOnsetMetrics()
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
        try:
            probabilistic_df, onset_da_dict = metrics.compute_metrics_multiple_years(
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
            )
        except Exception:
            probabilistic_df, onset_da_dict = d_metrics.compute_metrics_multiple_years(
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
            )

        model_dfs[model_name] = probabilistic_df
        model_onsets[model_name] = onset_da_dict

    return model_dfs, model_onsets


def get_plot_metrics(
    model_dfs: dict[str, pd.DataFrame],
    model_onsets: dict[str, xr.DataArray],
    metrics_df_clim: pd.DataFrame,
    onset_da_clim: xr.DataArray,
    config: dict[str, str],
    day: int = 15,
) -> dict[str, np.ndarray]:
    """
    Get Figure 1 & 4 plot metrics for a given set of model
    dataframes, onset data arrays, climatological data, and configuration.

    Args:
        model_dfs: Dictionary of model names and their individual metric dataframes.
        model_onsets: Dictionary of model names and their onset data arrays.
        metrics_df_clim: Climatological metrics dataframe.
        onset_da_clim: Climatological onset data array.
        config: Dictionary of configuration parameters.
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
        metrics_df_clim, dict.fromkeys(config["years"], onset_da_clim)
    )

    mean_mae = plot_metrics["AIFS"]["mean_mae"]

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
        "mae_cmz_fixed_clim": np.array(cmz_metrics.iloc[0]["overall_mae_mean_days"]),
        f"mae_cmz_mean_{str(day)}": np.array(cmz_metrics["cmz_mae_mean_days"].values),
        f"mae_yr_{str(day)}": np.array(cmz_metrics["overall_mae_mean_days"].values),
        f"miss_rate_{str(day)}": miss_rate_15.values,
        f"mr_cmz_mean_{str(day)}": np.array(cmz_metrics["cmz_mr_pct"].values),
        f"std_er_{str(day)}": np.array(cmz_metrics["cmz_mae_se_days"].values),
        "std_er_fixed_clim": np.array(cmz_metrics.iloc[0]["cmz_mae_se_days"]),
    }

    return mat_dict


def get_climatological_dfs(
    config: dict[str, str],
) -> dict[str, tuple[pd.DataFrame, xr.DataArray]]:
    """
    Get climatological dataframes and onset data arrays for all year ranges.

    Args:
        config: Dictionary of configuration parameters.

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
    )

    # Compute 15-day forecast data for the extended period
    clim_df_15_ex, clim_onset_15_ex = c_metrics.compute_climatology_baseline_multiple_years(
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
    )

    # Compute 30-day forecast data
    clim_df_30, clim_onset_30 = c_metrics.compute_climatology_baseline_multiple_years(
        years=config["years"],
        imd_folder=config["imd_folder"],
        thres_file=config["thres_file"],
        tolerance_days=3,
        verification_window=1,
        forecast_days=30,
        max_forecast_day=30,
        mok=True,
        onset_window=5,
        mok_month=6,
        mok_day=2,
    )

    # Compute 30-day forecast data for the extended period
    clim_df_30_ex, clim_onset_30_ex = c_metrics.compute_climatology_baseline_multiple_years(
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
    )

    output = {
        "15_day": (clim_df_15, clim_onset_15),
        "15_day_ex": (clim_df_15_ex, clim_onset_15_ex),
        "30_day": (clim_df_30, clim_onset_30),
        "30_day_ex": (clim_df_30_ex, clim_onset_30_ex),
    }

    return output

def save_data(
    mat_dict: dict[str, np.ndarray],
    output_dir: str,
    save_path
):
    """
    Save data to a .mat file.

    Args:
        mat_dict: Dictionary of data to save.
        output_dir: Directory to save the data.
        save_path: Path to save the data.
    """
    out_path = f"{output_dir}/{save_path}.mat"
    savemat(out_path, mat_dict)
    print("Saved to:", out_path)