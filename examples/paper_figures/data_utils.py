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
    "GenCast": np.arange(2019, 2025),
    "FuXi-S2S": np.arange(2019, 2022),
    "NGCM": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
}

# Figure 4 year ranges
YEAR_RANGES_COM = {
    "AIFS": np.arange(2004, 2022),
    "IFS": np.arange(2004, 2022),
    "FuXi": np.arange(2004, 2022),
    "Graphcast": np.arange(2004, 2022),
    "GenCast": np.arange(2019, 2022),
    "FuXi-S2S": np.arange(2004, 2022),
    "NGCM": np.arange(2004, 2022),
}

DETERMINISTIC_MODELS = ['IFS', 'AIFS', 'FuXi', 'Graphcast']
PROBABILISTIC_MODELS = ['GenCast', 'FuXi-S2S', 'NGCM']

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
        if model_name in PROBABILISTIC_MODELS:
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
            )
        else:
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
) -> dict[str, tuple[pd.DataFrame, xr.DataArray]]:
    """Get climatological dataframes and onset data arrays for all year ranges.

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


def load_wyi(output_dir: str) -> None:
    """Load WYI data from a .mat file.

    Args:
        output_dir: Directory to save the WYI data to.
    """
    model_str = np.array(
        [["clim"], ["ifs"], ["aifs"], ["fuxi"], ["graphcast"], ["gencast"], ["ngcm51"]]
    )

    alt_model_str = np.array(
        [["clim"], ["ifs"], ["aifs"], ["fuxi"], ["graphcast"], ["ngcm51"]]
    )

    wyi_15day = {
        "false_alarm": np.array(
            [[0.5434, 0.1481, 0.2333, 0.3030, 0.2000, 0.1666, 0.1612]]
        ),
        "mae_avg": np.array([[6.1666, 1.8000, 2.3435, 4.7546, 2.1712, 1.9861, 1.6202]]),
        "miss_rate": np.array(
            [[0.3333, 0.1739, 0.0740, 0.1851, 0.1851, 0.1481, 0.1111]]
        ),
        "model_str": model_str,
        "std_er": np.array([[0.8724, 1.0588, 1.3824, 1.9608, 1.1938, 1.2578, 0.9984]]),
    }

    wyi_30day = {
        "false_alarm": np.array(
            [[1.0000, 0.5000, 0.8571, 0.7500, 0.5000, 0.4285, 0.7000]]
        ),
        "mae_avg": np.array([[6.1666, 4.3766, 5.1111, 5.9666, 4.6694, 4.1500, 4.0666]]),
        "miss_rate": np.array(
            [[0.1224, 0.0476, 0.0408, 0.0204, 0.0408, 0.0816, 0.0204]]
        ),
        "model_str": model_str,
        "std_er": np.array([[0.8724, 1.6206, 1.9510, 2.1287, 2.0323, 1.6568, 0.8027]]),
    }

    wyi_15day_extended = {
        "false_alarm": np.array(
            [[0.4112, 0.1481, 0.2870, 0.4298, 0.3425, 0.2056, 0.2545]]
        ),
        "mae_avg": np.array([[4.7000, 1.8000, 3.1356, 4.7063, 3.3155, 2.6904, 2.6446]]),
        "miss_rate": np.array(
            [[0.2413, 0.1739, 0.1264, 0.1034, 0.1264, 0.0919, 0.0689]]
        ),
        "model_str": model_str,
        "std_er": np.array([[0.7884, 1.0588, 0.6182, 0.8258, 0.6061, 0.5977, 0.5668]]),
    }

    wyi_30day_extended = {
        "false_alarm": np.array(
            [[1.0000, 0.5000, 0.8709, 0.9210, 0.8484, 0.6923, 0.7777]]
        ),
        "mae_avg": np.array([[4.7000, 4.3766, 5.1447, 7.1400, 6.6508, 4.1158, 4.1641]]),
        "miss_rate": np.array(
            [[0.0848, 0.0476, 0.0606, 0.0060, 0.0121, 0.0484, 0.0181]]
        ),
        "model_str": model_str,
        "std_er": np.array([[0.7884, 1.6206, 0.7717, 1.0998, 1.2267, 0.7091, 0.8559]]),
    }

    wyi_15day_common = {
        "false_alarm": np.array([[0.3894, 0.1250, 0.1621, 0.3589, 0.1126, 0.1600]]),
        "mae_avg": np.array([[4.4705, 1.9068, 1.6950, 3.2616, 1.3737, 1.7058]]),
        "miss_rate": np.array([[0.2531, 0.2151, 0.1265, 0.0886, 0.1645, 0.1898]]),
        "model_str": alt_model_str,
        "std_er": np.array([[0.6812, 0.5090, 0.4903, 0.7105, 0.4258, 0.3777]]),
    }

    wyi_30day_common = {
        "false_alarm": np.array([[1.0000, 0.6666, 0.7692, 0.8500, 0.9000, 0.8888]]),
        "mae_avg": np.array([[5.2222, 4.1462, 3.4617, 6.4148, 4.3156, 4.5166]]),
        "miss_rate": np.array([[0.0507, 0.0652, 0.0434, 0.0507, 0.0144, 0.0434]]),
        "model_str": alt_model_str,
        "std_er": np.array([[0.9886, 0.7309, 0.7843, 1.0687, 0.8023, 0.6104]]),
    }

    wyi_data = [
        wyi_15day,
        wyi_30day,
        wyi_15day_extended,
        wyi_30day_extended,
        wyi_15day_common,
        wyi_30day_common,
    ]

    wyi_paths = [
        "wyi_onset_deterministic_metrics_15day_2019_2024",
        "wyi_onset_deterministic_metrics_30day_2019_2024",
        "wyi_onset_deterministic_metrics_15day_1965_1978_2019_2024_with_gencast",
        "wyi_onset_deterministic_metrics_30day_1965_1978_2019_2024_with_gencast",
        "wyi_onset_deterministic_metrics_15day_2004_2021",
        "wyi_onset_deterministic_metrics_30day_2004_2021",
    ]

    for wyi_dct, fp in zip(wyi_data, wyi_paths):
        save_data(wyi_dct, output_dir, fp)
    return


def save_all_data(model_paths: dict[str, str], config: dict[str, str]) -> None:
    """Save all data for figures 1 and 4.

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

    print("Saving WYI Data...")
    load_wyi(config["output_dir"])
    return
