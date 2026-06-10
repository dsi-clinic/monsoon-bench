"""Webster-Yang Index (WYI) loading and onset helpers.

The forecast WYI files in ``data/wyi/webster_yang_index`` contain a
precomputed WYI value at each forecast step.  The current NetCDF files store
6-hour steps, so the scoring path defaults to using the 24-hour valid steps
as daily lead times.  Literal stored-column indexing is still available for
diagnostics via ``forecast_step_mode="stored_index"``.
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import xarray as xr

WYI_THRESHOLD = -14.8
WYI_VARIABLE = "webster_yang_index"
WYI_CLIMATOLOGY_YEARS = np.arange(1979, 2025)
IDX_JUN2_365DAY = 152  # zero-indexed June 2 after removing Feb 29
FORECAST_DAYS = 14
ENSEMBLE_ONSET_THRESHOLD_PERCENT = 50.0

def matlab_like_movmean_7(x: np.ndarray) -> np.ndarray:
    """Return MATLAB ``movmean(x, 7)`` equivalent for a 1-D series."""
    return (
        pd.Series(np.asarray(x, dtype=float))
        .rolling(window=7, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def _resolve_wyi_year_file(data_dir: Path, year: int) -> Path:
    """Resolve either MATLAB-style or local WYI yearly filenames."""
    candidates = [
        data_dir / f"wyi_daily_{year}.nc",
        data_dir / f"{year}.nc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No WYI file found for {year}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_observed_wyi_year(year: int, data_dir: Path) -> np.ndarray:
    """Load one full-calendar observed WYI series and remove Feb 29."""
    path = _resolve_wyi_year_file(data_dir, year)
    with xr.open_dataset(path) as ds:
        wyi = ds[WYI_VARIABLE].values.astype(float)

    wyi = np.squeeze(wyi)
    if wyi.ndim != 1:
        raise ValueError(
            f"Observed WYI file {path} must contain a 1-D full-year series; "
            f"got shape {wyi.shape}. Forecast WYI files cannot be used as "
            "ground truth for get_wyi_onset()."
        )

    if len(wyi) == 366:
        wyi = np.delete(wyi, 59)
    if len(wyi) != 365:
        raise ValueError(
            f"Observed WYI file {path} has length {len(wyi)} after leap-day "
            "handling; expected 365."
        )
    return wyi


def _build_365day_dates(year: int) -> pd.DatetimeIndex:
    """Return all dates in *year* with Feb 29 removed."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    if len(dates) == 366:
        dates = dates[dates != pd.Timestamp(year, 2, 29)]
    return dates


def get_wyi_onset(
    year: int,
    data_dir: str | Path,
    climatology_years: np.ndarray = WYI_CLIMATOLOGY_YEARS,
) -> tuple[pd.Timestamp | None, np.ndarray, np.ndarray]:
    """Detect observed WYI onset from full-year daily WYI files.

    This ports ``get_wyi_onset.m``.  ``data_dir`` must point to the ERA5 or
    observed daily WYI directory, not a forecast-model directory.
    """
    data_path = Path(data_dir)

    wyi_yr = np.full((365, len(climatology_years)), np.nan)
    for idx, clim_year in enumerate(climatology_years):
        wyi_yr[:, idx] = _load_observed_wyi_year(int(clim_year), data_path)

    wyi_clim = np.nanmean(wyi_yr, axis=1)
    wyi_threshold = wyi_clim[IDX_JUN2_365DAY]

    wyi_year = _load_observed_wyi_year(year, data_path)
    wyi_smoothed = matlab_like_movmean_7(wyi_year)
    onset_dates = _build_365day_dates(year)[wyi_smoothed < wyi_threshold]

    if len(onset_dates) == 0:
        return None, wyi_smoothed, wyi_clim

    wyi_onset = onset_dates[0]
    if wyi_onset < pd.Timestamp(year, 5, 20):
        warnings.warn(
            f"The WYI onset date ({wyi_onset.strftime('%Y-%m-%d')}) is earlier than May 20.",
            stacklevel=2,
        )
    return wyi_onset, wyi_smoothed, wyi_clim


def _decode_init_dates(time_coord: xr.DataArray) -> pd.DatetimeIndex:
    """Decode an xarray time coordinate into pandas timestamps."""
    if np.issubdtype(time_coord.dtype, np.datetime64):
        return pd.DatetimeIndex(time_coord.values)

    units = time_coord.attrs.get("units", "")
    if "since" not in units:
        raise ValueError("Numeric WYI time coordinate must define '<unit> since <date>'.")
    unit_name, base_text = units.split("since", maxsplit=1)
    unit_name = unit_name.strip().lower()
    unit_map = {
        "day": "D",
        "days": "D",
        "hour": "h",
        "hours": "h",
        "minute": "m",
        "minutes": "m",
        "second": "s",
        "seconds": "s",
    }
    if unit_name not in unit_map:
        raise ValueError(f"Unsupported WYI time units '{unit_name}'.")
    base_date = pd.Timestamp(base_text.strip())
    return pd.DatetimeIndex(
        base_date + pd.to_timedelta(time_coord.values, unit=unit_map[unit_name])
    )


def _normalize_forecast_array(ds: xr.Dataset) -> np.ndarray:
    """Return forecast WYI as ``(time, step)`` or ``(member, time, step)``."""
    da = ds[WYI_VARIABLE]
    dims = set(da.dims)
    if {"number", "time", "step"}.issubset(dims):
        return da.transpose("number", "time", "step").values
    if {"time", "step"}.issubset(dims):
        return da.transpose("time", "step").values
    raise ValueError(
        f"Unsupported WYI dimensions {da.dims}; expected time/step with optional number."
    )


def _align_init_dates_to_year(init_dates: pd.DatetimeIndex, year: int) -> pd.DatetimeIndex:
    """Return initialization dates with month/day preserved in ``year``."""
    if np.all(init_dates.year == year):
        return init_dates
    mismatched_years = sorted(set(int(date.year) for date in init_dates))
    warnings.warn(
        f"Aligning WYI initialization dates from coordinate year(s) "
        f"{mismatched_years} to file year {year}.",
        stacklevel=3,
    )
    return pd.DatetimeIndex([date.replace(year=year) for date in init_dates])


def load_wyi_forecast(
    model_path: str | Path,
    year: int,
    align_init_dates: bool = True,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load a precomputed forecast WYI file.

    Returns
    -------
    wyi:
        ``(n_init, n_step)`` for deterministic models or
        ``(n_member, n_init, n_step)`` for ensembles.
    init_dates:
        Initialization dates.  Some FuXi files carry a previous-year time
        coordinate even though the file is named for the target year; by
        default the month/day template is aligned to ``year``.
    step_hours:
        Stored step coordinate in hours.  The default scoring path uses the
        24-hour valid steps as daily lead times.
    """
    path = Path(model_path) / f"{year}.nc"
    with xr.open_dataset(path) as ds:
        wyi = _normalize_forecast_array(ds)
        init_dates = _decode_init_dates(ds["time"])
        step_hours = ds["step"].values.astype(int)
    if align_init_dates:
        init_dates = _align_init_dates_to_year(init_dates, year)
    return wyi, init_dates, step_hours


def make_wyi_loader(model_path: str | Path):
    """Create a ``compute_wyi_metrics`` data loader for a forecast directory."""

    def data_loader(year: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
        wyi, init_dates, _ = load_wyi_forecast(model_path, year)
        return init_dates, wyi

    return data_loader


def convert_6hourly_to_daily(wyi_6h: np.ndarray, step_hours: np.ndarray, method: str = "daily_mean"):
    """Convert stored 6-hourly values to daily values.

    This helper is kept for diagnostics; scoring code should call
    ``select_wyi_forecast_steps`` so deterministic and ensemble arrays are
    handled with the same conventions.
    """
    step_hours = np.asarray(step_hours)

    if method == "valid_24h":
        mask = step_hours % 24 == 0
        lead_days = (step_hours[mask] // 24).astype(int)
        return wyi_6h[..., mask], lead_days

    if method == "daily_mean":
        max_day = int(step_hours.max() // 24)
        daily_values = []
        lead_days = []
        for day in range(1, max_day + 1):
            mask = (step_hours > 24 * (day - 1)) & (step_hours <= 24 * day)
            if mask.any():
                daily_values.append(np.nanmean(wyi_6h[..., mask], axis=-1))
                lead_days.append(day)
        return np.stack(daily_values, axis=-1), np.array(lead_days)

    raise ValueError("method must be either 'daily_mean' or 'valid_24h'")


def select_wyi_forecast_steps(
    wyi: np.ndarray,
    step_hours: np.ndarray,
    mode: str = "valid_24h",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Select forecast samples for WYI onset scoring.

    Args:
        wyi: Forecast WYI array, either ``(init, step)`` or
            ``(member, init, step)``.
        step_hours: Forecast step coordinate in hours.
        mode: ``"valid_24h"`` uses only 24-hour valid steps and maps them to
            lead days from the step coordinate. ``"stored_index"`` uses stored
            columns as day 1, day 2, ... matching the literal MATLAB indexing.
            ``"daily_mean"`` averages all samples within each 24-hour bin.

    Returns:
        ``(wyi_selected, lead_days)``. ``lead_days`` is ``None`` for
        ``stored_index`` because column index maps directly to day number.
    """
    wyi = np.asarray(wyi)
    step_hours = np.asarray(step_hours)
    normalized_mode = mode.lower()

    if normalized_mode in {"stored_index", "index", "matlab_index"}:
        return wyi, None

    if normalized_mode == "valid_24h":
        mask = step_hours % 24 == 0
        lead_days = (step_hours[mask] // 24).astype(int)
        return wyi[..., mask], lead_days

    if normalized_mode == "daily_mean":
        max_day = int(step_hours.max() // 24)
        daily_values = []
        lead_days = []
        for day in range(1, max_day + 1):
            mask = (step_hours > 24 * (day - 1)) & (step_hours <= 24 * day)
            if mask.any():
                daily_values.append(np.nanmean(wyi[..., mask], axis=-1))
                lead_days.append(day)
        return np.stack(daily_values, axis=-1), np.array(lead_days)

    raise ValueError(
        "forecast_step_mode must be one of 'valid_24h', 'stored_index', or 'daily_mean'."
    )


def predicted_onsets_from_wyi(
    wyi_steps: np.ndarray,
    init_dates: pd.DatetimeIndex,
    lead_days: np.ndarray | None = None,
    threshold: float = WYI_THRESHOLD,
    max_steps: int | None = None,
) -> pd.DatetimeIndex:
    """Return one predicted onset date per initialization.

    By default, ``lead_days`` is ``1, 2, ...`` matching the MATLAB code's
    ``t_init(i) + wyi_onset_idx(1)`` convention.
    """
    wyi_steps = np.asarray(wyi_steps)
    if max_steps is not None:
        wyi_steps = wyi_steps[:, :max_steps]

    if lead_days is None:
        lead_days = np.arange(1, wyi_steps.shape[1] + 1)
    elif max_steps is not None:
        lead_days = np.asarray(lead_days)[:max_steps]

    predicted_onsets = []
    for init_date, wyi_ts in zip(init_dates, wyi_steps):
        smoothed = matlab_like_movmean_7(wyi_ts)
        onset_indices = np.flatnonzero(smoothed < threshold)
        if len(onset_indices) == 0:
            predicted_onsets.append(pd.NaT)
        else:
            lead_day = int(lead_days[onset_indices[0]])
            predicted_onsets.append(init_date + pd.Timedelta(days=lead_day))
    return pd.DatetimeIndex(predicted_onsets)


def _doy_to_timestamp(year: int, doy: float) -> pd.Timestamp:
    """Convert day-of-year to timestamp using MATLAB-style rounded DOY."""
    rounded_doy = int(np.floor(float(doy) + 0.5))
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=rounded_doy - 1)


def predicted_onsets_from_wyi_ensemble(
    wyi_steps: np.ndarray,
    init_dates: pd.DatetimeIndex,
    year: int,
    lead_days: np.ndarray | None = None,
    threshold: float = WYI_THRESHOLD,
    max_steps: int | None = None,
    ens_pct_thres: float = ENSEMBLE_ONSET_THRESHOLD_PERCENT,
) -> pd.DatetimeIndex:
    """Return ensemble-mean onset dates for probabilistic/ensemble WYI forecasts.

    Each member is first converted to an onset date using the same stored-step
    convention as deterministic forecasts.  An initialization is accepted only
    when at least ``ens_pct_thres`` percent of members detect onset; the final
    onset is the rounded mean day-of-year across those members.
    """
    wyi_steps = np.asarray(wyi_steps)
    if wyi_steps.ndim != 3:
        raise ValueError(
            "Ensemble WYI forecasts must have shape (member, init, step); "
            f"got {wyi_steps.shape}."
        )

    if max_steps is not None:
        wyi_steps = wyi_steps[:, :, :max_steps]

    if lead_days is None:
        lead_days = np.arange(1, wyi_steps.shape[2] + 1)
    elif max_steps is not None:
        lead_days = np.asarray(lead_days)[:max_steps]

    n_members, n_init, _ = wyi_steps.shape
    member_threshold = math.ceil((ens_pct_thres / 100.0) * n_members)
    onset_doy = np.full((n_init, n_members), np.nan)

    for member_idx in range(n_members):
        member_onsets = predicted_onsets_from_wyi(
            wyi_steps[member_idx],
            init_dates,
            lead_days=lead_days,
            threshold=threshold,
        )
        for init_idx, onset in enumerate(member_onsets):
            if pd.notna(onset):
                onset_doy[init_idx, member_idx] = pd.Timestamp(onset).dayofyear

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_onset_doy = np.nanmean(onset_doy, axis=1)
    nonnull_members = np.sum(~np.isnan(onset_doy), axis=1)

    ensemble_onsets = []
    for init_idx in range(n_init):
        if nonnull_members[init_idx] >= member_threshold and not np.isnan(mean_onset_doy[init_idx]):
            ensemble_onsets.append(_doy_to_timestamp(year, mean_onset_doy[init_idx]))
        else:
            ensemble_onsets.append(pd.NaT)
    return pd.DatetimeIndex(ensemble_onsets)


def score_wyi_onset_year(
    init_dates: pd.DatetimeIndex,
    predicted_onsets: pd.DatetimeIndex,
    ground_truth_onset: pd.Timestamp | str,
    verification_window: int = 1,
    tol: int = 3,
    horizon_days: int = 14,
) -> dict:
    """Compute MATLAB-compatible WYI onset contingency metrics."""
    ground_truth_onset = pd.Timestamp(ground_truth_onset)
    init_dates = pd.DatetimeIndex(init_dates)

    TP = FP = FN = TN = 0
    num_onset = 0
    num_no_onset = 0
    abs_errors = []

    keep = init_dates < ground_truth_onset
    init_dates = init_dates[keep]
    predicted_onsets = predicted_onsets[keep]

    for init_date, pred_onset in zip(init_dates, predicted_onsets):
        valid_window_start = init_date + pd.Timedelta(days=verification_window)
        valid_window_end = valid_window_start + pd.Timedelta(days=horizon_days)
        whole_window_start = init_date + pd.Timedelta(days=1)
        whole_window_end = init_date + pd.Timedelta(days=verification_window + horizon_days)

        true_in_whole_window = whole_window_start <= ground_truth_onset <= whole_window_end
        if true_in_whole_window:
            num_onset += 1
        else:
            num_no_onset += 1

        if pd.notna(pred_onset):
            if valid_window_start <= pred_onset <= valid_window_end:
                error_days = abs((pred_onset - ground_truth_onset).days)
                abs_errors.append(error_days)
                if error_days <= tol:
                    TP += 1
                else:
                    FP += 1
        elif true_in_whole_window:
            FN += 1
        else:
            TN += 1

    mae = np.nan if len(abs_errors) == 0 else float(np.mean(abs_errors))
    miss_rate = np.nan if num_onset == 0 else FN / num_onset
    false_alarm = np.nan if (FP + TN) == 0 else FP / (FP + TN)
    false_alarm_ratio = np.nan if (TP + FP) == 0 else FP / (TP + FP)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "num_onset": num_onset,
        "num_no_onset": num_no_onset,
        "MAE": mae,
        "MR": miss_rate,
        "FAR_code_definition": false_alarm,
        "FAR_common_ratio": false_alarm_ratio,
    }


def wyi_one_year(
    model_path: str | Path,
    year: int,
    ground_truth_onset: pd.Timestamp | str,
    verification_window: int = 1,
    tol: int = 3,
    model_type: str = "deterministic",
    ens_pct_thres: float = ENSEMBLE_ONSET_THRESHOLD_PERCENT,
    forecast_step_mode: str = "valid_24h",
) -> dict:
    """Score one model year using stored WYI steps."""
    wyi, init_dates, step_hours = load_wyi_forecast(model_path, year)
    normalized_type = normalize_wyi_model_type(model_type, wyi)
    wyi, lead_days = select_wyi_forecast_steps(wyi, step_hours, mode=forecast_step_mode)

    max_steps = verification_window + FORECAST_DAYS
    if normalized_type == "deterministic":
        if wyi.ndim != 2:
            raise ValueError("Deterministic WYI forecasts must have shape (init, step).")
        predicted_onsets = predicted_onsets_from_wyi(
            wyi,
            init_dates,
            lead_days=lead_days,
            threshold=WYI_THRESHOLD,
            max_steps=max_steps,
        )
    else:
        predicted_onsets = predicted_onsets_from_wyi_ensemble(
            wyi,
            init_dates,
            year=year,
            lead_days=lead_days,
            threshold=WYI_THRESHOLD,
            max_steps=max_steps,
            ens_pct_thres=ens_pct_thres,
        )

    return score_wyi_onset_year(
        init_dates=init_dates,
        predicted_onsets=predicted_onsets,
        ground_truth_onset=ground_truth_onset,
        verification_window=verification_window,
        tol=tol,
    )


def wyi_multiple_years(
    model_path: str | Path,
    years: np.ndarray,
    ground_truth_onsets: dict[int, pd.Timestamp | str],
    verification_window: int = 1,
    tol: int = 3,
    model_type: str = "deterministic",
    ens_pct_thres: float = ENSEMBLE_ONSET_THRESHOLD_PERCENT,
    forecast_step_mode: str = "valid_24h",
) -> dict:
    """Aggregate WYI metrics across years."""
    year_metrics = []
    TP = FP = FN = TN = num_onset = 0
    maes = []

    for year in years:
        metrics = wyi_one_year(
            model_path,
            int(year),
            ground_truth_onsets[int(year)],
            verification_window=verification_window,
            tol=tol,
            model_type=model_type,
            ens_pct_thres=ens_pct_thres,
            forecast_step_mode=forecast_step_mode,
        )
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
    miss_rate = np.nan if num_onset == 0 else FN / num_onset
    false_alarm = np.nan if (FP + TN) == 0 else FP / (FP + TN)
    false_alarm_ratio = np.nan if (TP + FP) == 0 else FP / (TP + FP)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "num_onset": num_onset,
        "MAE": mae,
        "std_er": std_er,
        "mae_yr": mae_yr,
        "MR": miss_rate,
        "FAR_code_definition": false_alarm,
        "FAR_common_ratio": false_alarm_ratio,
        "year_metrics": year_metrics,
    }


def _filter_available_forecast_years(model_path: str | Path, years: np.ndarray) -> np.ndarray:
    """Keep only years with forecast files in ``model_path``."""
    model_path = Path(model_path)
    available_years = []
    missing_years = []
    for year in np.asarray(years, dtype=int):
        if (model_path / f"{int(year)}.nc").exists():
            available_years.append(int(year))
        else:
            missing_years.append(int(year))
    if missing_years:
        warnings.warn(
            f"Skipping WYI forecast years with no file in {model_path}: {missing_years}",
            stacklevel=2,
        )
    return np.asarray(available_years, dtype=int)


def normalize_wyi_model_type(model_type: str | None, wyi: np.ndarray | None = None) -> str:
    """Normalize user-facing model-type names to deterministic/probabilistic."""
    if model_type is None or str(model_type).lower() == "auto":
        if wyi is None:
            raise ValueError("model_type='auto' requires a loaded WYI array.")
        return "probabilistic" if np.asarray(wyi).ndim == 3 else "deterministic"

    lowered = str(model_type).lower()
    if lowered in {"deterministic", "single", "single_member"}:
        return "deterministic"
    if lowered in {"probabilistic", "ensemble", "prob"}:
        return "probabilistic"
    raise ValueError(
        f"Unsupported WYI model_type '{model_type}'. "
        "Expected deterministic, probabilistic, ensemble, or auto."
    )


def infer_wyi_model_type(model_path: str | Path, year: int) -> str:
    """Infer deterministic vs probabilistic from one forecast file."""
    wyi, _, _ = load_wyi_forecast(model_path, year)
    return normalize_wyi_model_type("auto", wyi)


def load_wyi_ground_truth_onsets(
    years: np.ndarray,
    observed_wyi_dir: str | Path,
    climatology_years: np.ndarray = WYI_CLIMATOLOGY_YEARS,
) -> dict[int, pd.Timestamp]:
    """Load observed WYI onset dates for a set of years."""
    onsets = {}
    for year in years:
        onset, _, _ = get_wyi_onset(
            int(year),
            observed_wyi_dir,
            climatology_years=climatology_years,
        )
        if onset is not None:
            onsets[int(year)] = onset
    return onsets


def score_wyi_model_years(
    model_path: str | Path,
    years: np.ndarray,
    ground_truth_onsets: dict[int, pd.Timestamp | str],
    verification_window: int,
    tol: int,
    model_type: str = "auto",
    ens_pct_thres: float = ENSEMBLE_ONSET_THRESHOLD_PERCENT,
    forecast_step_mode: str = "valid_24h",
) -> dict:
    """Score a deterministic or probabilistic WYI model over multiple years."""
    years = np.asarray(years, dtype=int)
    if len(years) == 0:
        raise ValueError("years must contain at least one year.")

    available_years = _filter_available_forecast_years(model_path, years)
    if len(available_years) == 0:
        raise ValueError(f"No requested years have forecast files in {model_path}.")

    if model_type == "auto":
        normalized_type = infer_wyi_model_type(model_path, int(available_years[0]))
    else:
        normalized_type = normalize_wyi_model_type(model_type)

    scored_years = np.array(
        [year for year in available_years if int(year) in ground_truth_onsets],
        dtype=int,
    )
    if len(scored_years) == 0:
        raise ValueError("No requested years have WYI ground-truth onset dates.")

    return wyi_multiple_years(
        model_path=model_path,
        years=scored_years,
        ground_truth_onsets=ground_truth_onsets,
        verification_window=verification_window,
        tol=tol,
        model_type=normalized_type,
        ens_pct_thres=ens_pct_thres,
        forecast_step_mode=forecast_step_mode,
    )


def wyi_results_to_mat_dict(results_by_model: dict[str, dict], model_order: list[str]) -> dict:
    """Convert model result dictionaries to the original WYI ``.mat`` layout."""
    return {
        "false_alarm": np.array(
            [results_by_model[model]["FAR_code_definition"] for model in model_order],
            dtype=float,
        ),
        "mae_avg": np.array(
            [results_by_model[model]["MAE"] for model in model_order],
            dtype=float,
        ),
        "miss_rate": np.array(
            [results_by_model[model]["MR"] for model in model_order],
            dtype=float,
        ),
        "model_str": np.array([str(model).lower() for model in model_order], dtype=object),
        "std_er": np.array(
            [results_by_model[model]["std_er"] for model in model_order],
            dtype=float,
        ),
    }
