"""Utilities for generating WYI onset figure data.

The output ``.mat`` files match the historical WYI files used by the paper
figures: ``false_alarm``, ``mae_avg``, ``miss_rate``, ``model_str``, and
``std_er``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from monsoonbench.onset.wyi_onset import (
    DATA_DIR,
    ENSEMBLE_ONSET_THRESHOLD_PERCENT,
    load_wyi_forecast,
    load_wyi_ground_truth_onsets,
    score_wyi_model_years,
    score_wyi_onset_year,
    wyi_results_to_mat_dict,
)

from examples.paper_figures.utils.data_utils import (
    YEAR_RANGES, YEAR_RANGES_COM, EXTENDED_YEARS
)


DEFAULT_MODEL_ORDER = ["clim", "ifs", "aifs", "fuxi", "graphcast", "gencast", "ngcm51"]

MODEL_NAME_ALIASES = {
    "ifs": "ifs",
    "ifs_s2s": "ifs",
    "ifss2s": "ifs",
    "aifs": "aifs",
    "fuxi": "fuxi",
    "graphcast": "graphcast",
    "graphcast_operational": "graphcast",
    "gencast": "gencast",
    "ngcm": "ngcm51",
    "ngcm51": "ngcm51",
    "neuralgcm": "ngcm51",
    "clim": "clim",
    "climatology": "clim",
}

PROBABILISTIC_MODELS = {"ifs", "gencast", "ngcm51"}
DETERMINISTIC_MODELS = {"aifs", "fuxi", "graphcast"}
DEFAULT_FORECAST_STEP_MODES = {
    "ifs": "stored_index",
}


def get_wyi_model_label(model_name: str) -> str:
    """Normalize model names to the labels used by the WYI ``.mat`` files."""
    key = model_name.lower().replace("-", "_")
    return MODEL_NAME_ALIASES.get(key, key)


def get_wyi_model_type(model_name: str) -> str:
    """Return deterministic, probabilistic, or climatology for a model label."""
    label = get_wyi_model_label(model_name)
    if label == "clim":
        return "climatology"
    if label in PROBABILISTIC_MODELS:
        return "probabilistic"
    if label in DETERMINISTIC_MODELS:
        return "deterministic"
    return "auto"


def get_wyi_forecast_step_mode(
    model_name: str,
    forecast_step_mode: str | dict[str, str],
) -> str:
    """Resolve model-specific forecast step conventions."""
    label = get_wyi_model_label(model_name)
    if isinstance(forecast_step_mode, dict):
        return forecast_step_mode.get(
            model_name,
            forecast_step_mode.get(label, DEFAULT_FORECAST_STEP_MODES.get(label, "valid_24h")),
        )
    if forecast_step_mode == "auto":
        return DEFAULT_FORECAST_STEP_MODES.get(label, "valid_24h")
    return forecast_step_mode


def get_wyi_observed_dir(config: dict) -> str:
    """Read the observed/full-year WYI directory from common config keys."""
    for key in ("wyi_era5_folder", "wyi_observed_dir", "wyi_ground_truth_dir", "wyi_data_dir", "wyi_folder"):
        if key in config:
            return config[key]
    default_path = DATA_DIR / "wyi_era5"
    if default_path.exists():
        return str(default_path)
    raise KeyError(
        "config must contain an observed full-year WYI directory under one of: "
        "wyi_era5_folder, wyi_observed_dir, wyi_ground_truth_dir, "
        "wyi_data_dir, or wyi_folder."
    )


def get_horizon_settings(horizon_days: int) -> tuple[int, int]:
    """Return ``verification_window`` and tolerance for a WYI forecast horizon."""
    if horizon_days == 15:
        return 1, 3
    if horizon_days == 30:
        return 16, 5
    raise ValueError("WYI horizon_days must be 15 or 30.")


def _reference_model_path(model_paths: dict[str, str]) -> str:
    """Choose a model path to provide initialization dates for climatology."""
    for preferred in ("NGCM", "ngcm", "NeuralGCM", "ngcm51"):
        if preferred in model_paths:
            return model_paths[preferred]
    for preferred in ("AIFS", "aifs", "Graphcast", "GraphCast", "graphcast", "FuXi", "fuxi"):
        if preferred in model_paths:
            return model_paths[preferred]
    for name, path in model_paths.items():
        if get_wyi_model_label(name) != "clim":
            return path
    raise ValueError("At least one non-climatology model path is needed for init dates.")


def _get_years_for_model(
    model_name: str,
    year_ranges: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Find a model's years using exact or normalized label matching."""
    if model_name in year_ranges:
        return np.asarray(year_ranges[model_name], dtype=int)

    label = get_wyi_model_label(model_name)
    for candidate_name, candidate_years in year_ranges.items():
        if get_wyi_model_label(candidate_name) == label:
            return np.asarray(candidate_years, dtype=int)
    return None


def _score_wyi_climatology_years(
    reference_model_path: str,
    years: np.ndarray,
    ground_truth_onsets: dict[int, pd.Timestamp | str],
    verification_window: int,
    tol: int,
) -> dict:
    """Score the fixed May-31 WYI climatology baseline."""
    year_metrics = []
    TP = FP = FN = TN = num_onset = 0
    maes = []

    for year in np.asarray(years, dtype=int):
        if int(year) not in ground_truth_onsets:
            continue

        _, init_dates, _ = load_wyi_forecast(reference_model_path, int(year))
        clim_onset = pd.Timestamp(int(year), 5, 31)
        ground_truth_onset = pd.Timestamp(ground_truth_onsets[int(year)])

        year_TP = year_FP = year_FN = year_TN = year_num_onset = 0
        abs_errors = []
        for init_date in init_dates[init_dates < ground_truth_onset]:
            valid_window_start = init_date + pd.Timedelta(days=verification_window)
            valid_window_end = valid_window_start + pd.Timedelta(days=14)
            whole_window_start = init_date + pd.Timedelta(days=1)
            whole_window_end = init_date + pd.Timedelta(days=verification_window + 14)

            true_in_whole_window = whole_window_start <= ground_truth_onset <= whole_window_end
            if true_in_whole_window:
                year_num_onset += 1

            if valid_window_start <= clim_onset <= valid_window_end:
                error_days = abs((clim_onset - ground_truth_onset).days)
                abs_errors.append(error_days)
                if error_days <= tol:
                    year_TP += 1
                else:
                    year_FP += 1

            if not (whole_window_start <= clim_onset <= whole_window_end):
                if true_in_whole_window:
                    year_FN += 1
                else:
                    year_TN += 1

        metrics = {
            "TP": year_TP,
            "FP": year_FP,
            "FN": year_FN,
            "TN": year_TN,
            "num_onset": year_num_onset,
            "MAE": np.nan if len(abs_errors) == 0 else float(np.mean(abs_errors)),
            "MR": np.nan if year_num_onset == 0 else year_FN / year_num_onset,
            "FAR_code_definition": np.nan if (year_FP + year_TN) == 0 else year_FP / (year_FP + year_TN),
            "FAR_common_ratio": np.nan if (year_TP + year_FP) == 0 else year_FP / (year_TP + year_FP),
        }
        year_metrics.append(metrics)
        TP += metrics["TP"]
        FP += metrics["FP"]
        FN += metrics["FN"]
        TN += metrics["TN"]
        num_onset += metrics["num_onset"]
        maes.append(metrics["MAE"])

    mae_yr = np.asarray(maes, dtype=float)
    n_valid_mae = int(np.sum(~np.isnan(mae_yr)))
    mae = np.nan if n_valid_mae == 0 else float(np.nanmean(mae_yr))
    std_er = (
        np.nan
        if n_valid_mae == 0
        else float(np.nanstd(mae_yr, ddof=1) / np.sqrt(n_valid_mae))
        if n_valid_mae > 1
        else 0.0
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "num_onset": num_onset,
        "MAE": mae,
        "std_er": std_er,
        "mae_yr": mae_yr,
        "MR": np.nan if num_onset == 0 else FN / num_onset,
        "FAR_code_definition": np.nan if (FP + TN) == 0 else FP / (FP + TN),
        "FAR_common_ratio": np.nan if (TP + FP) == 0 else FP / (TP + FP),
        "year_metrics": year_metrics,
    }


def compute_wyi_figure_results(
    config: dict,
    year_ranges: dict[str, np.ndarray],
    model_paths: dict[str, str],
    horizon_days: int,
    model_types: dict[str, str] | None = None,
    ens_pct_thres: float = ENSEMBLE_ONSET_THRESHOLD_PERCENT,
    forecast_step_mode: str | dict[str, str] = "auto",
    include_climatology: bool = True,
) -> tuple[dict[str, dict], list[str]]:
    """Compute WYI onset metrics for one horizon.

    Args:
        config: Must contain the observed full-year WYI directory.
        year_ranges: Mapping from model name to target years.
        model_paths: Mapping from model name to forecast WYI directory.
        horizon_days: Either 15 or 30.
        model_types: Optional mapping overriding deterministic/probabilistic
            inference for individual model names.
        ens_pct_thres: Percent of ensemble members required for onset.
        forecast_step_mode: How to map forecast step samples to lead days.
            ``"auto"`` uses the known model-specific conventions: IFS uses
            stored columns, while the current deterministic files use 24-hour
            valid steps.
        include_climatology: If True, add the fixed May-31 climatology row.

    Returns:
        ``(results_by_model, model_order)``.
    """
    verification_window, tol = get_horizon_settings(horizon_days)
    observed_dir = get_wyi_observed_dir(config)
    model_types = model_types or {}

    results_by_model = {}
    model_order = []
    clim_years = None

    if include_climatology:
        matched_clim_years = _get_years_for_model("clim", year_ranges)
        fallback_years = (
            config.get("years", [])
            if len(config.get("years", [])) > 0
            else np.unique(np.concatenate([np.asarray(years, dtype=int) for years in year_ranges.values()]))
        )
        clim_years = np.asarray(
            matched_clim_years if matched_clim_years is not None else fallback_years,
            dtype=int,
        )

    requested_year_sets = [np.asarray(years, dtype=int) for years in year_ranges.values()]
    if clim_years is not None and len(clim_years) > 0:
        requested_year_sets.append(clim_years)
    all_years = np.unique(np.concatenate(requested_year_sets))
    ground_truth_onsets = load_wyi_ground_truth_onsets(all_years, observed_dir)

    if include_climatology:
        results_by_model["clim"] = _score_wyi_climatology_years(
            reference_model_path=_reference_model_path(model_paths),
            years=clim_years,
            ground_truth_onsets=ground_truth_onsets,
            verification_window=verification_window,
            tol=tol,
        )
        model_order.append("clim")

    for model_name, model_path in model_paths.items():
        label = get_wyi_model_label(model_name)
        model_years = _get_years_for_model(model_name, year_ranges)
        if label == "clim" or model_years is None:
            continue

        model_type = model_types.get(model_name, model_types.get(label, get_wyi_model_type(model_name)))
        model_step_mode = get_wyi_forecast_step_mode(model_name, forecast_step_mode)
        results_by_model[label] = score_wyi_model_years(
            model_path=model_path,
            years=model_years,
            ground_truth_onsets=ground_truth_onsets,
            verification_window=verification_window,
            tol=tol,
            model_type=model_type,
            ens_pct_thres=ens_pct_thres,
            forecast_step_mode=model_step_mode,
        )
        model_order.append(label)

    ordered = [model for model in DEFAULT_MODEL_ORDER if model in results_by_model]
    ordered += [model for model in model_order if model not in ordered]
    return results_by_model, ordered


def compute_wyi_figure_mat_dict(
    config: dict,
    year_ranges: dict[str, np.ndarray],
    model_paths: dict[str, str],
    horizon_days: int,
    model_types: dict[str, str] | None = None,
    ens_pct_thres: float = ENSEMBLE_ONSET_THRESHOLD_PERCENT,
    forecast_step_mode: str | dict[str, str] = "auto",
    include_climatology: bool = True,
) -> dict:
    """Compute one WYI figure ``.mat`` dictionary."""
    results_by_model, model_order = compute_wyi_figure_results(
        config=config,
        year_ranges=year_ranges,
        model_paths=model_paths,
        horizon_days=horizon_days,
        model_types=model_types,
        ens_pct_thres=ens_pct_thres,
        forecast_step_mode=forecast_step_mode,
        include_climatology=include_climatology,
    )
    return wyi_results_to_mat_dict(results_by_model, model_order)


def save_wyi_figure_mat(
    mat_dict: dict,
    output_dir: str | Path,
    filename: str,
) -> Path:
    """Save a WYI figure dictionary as a MATLAB ``.mat`` file."""
    output_path = Path(output_dir) / filename
    if output_path.suffix != ".mat":
        output_path = output_path.with_suffix(".mat")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_path, mat_dict)
    print("Saved to:", output_path)
    return output_path


def load_wyi_figure_data(
    config: dict,
    model_paths: dict[str, str],
    horizons: tuple[int, ...] = (15, 30),
    filename_prefix: str = "wyi_onset_deterministic_metrics",
    ens_pct_thres: float = ENSEMBLE_ONSET_THRESHOLD_PERCENT,
    forecast_step_mode: str | dict[str, str] = "auto",
    include_climatology: bool = True,
) -> dict[int, Path]:
    """Compute and save WYI ``.mat`` files for one or more horizons.

    This mirrors the paper utility pattern: pass ``config``, ``year_ranges``,
    and ``model_paths``; generated files are written to ``config['output_dir']``
    unless ``output_dir`` is provided.  If ``period_label`` is supplied,
    filenames follow ``{filename_prefix}_{horizon}day_{period_label}.mat``.
    """
    output_root =config["output_dir"]
    saved = {}

    year_dict = {
        "2019_2024": YEAR_RANGES,
        "2004_2021": YEAR_RANGES_COM,
        "1965_1978_2019_2024": EXTENDED_YEARS,
    }

    for period_label, years in year_dict.items():
        for horizon_days in horizons:
            mat_dict = compute_wyi_figure_mat_dict(
                config=config,
                year_ranges=years,
                model_paths=model_paths,
                horizon_days=horizon_days,
                ens_pct_thres=ens_pct_thres,
                forecast_step_mode=forecast_step_mode,
                include_climatology=include_climatology,
            )
            if period_label is None:
                filename = f"{filename_prefix}_{horizon_days}day.mat"
            else:
                filename = f"{filename_prefix}_{horizon_days}day_{period_label}.mat"
            saved[f"{horizon_days}_{period_label}"] = save_wyi_figure_mat(mat_dict, output_root, filename)

    return saved
