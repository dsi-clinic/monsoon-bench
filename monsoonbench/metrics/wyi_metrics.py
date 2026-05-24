"""WYI-based monsoon onset deterministic metrics.

Ported from deterministic_metrics_for_wyi_onset.m.

Supported model types
---------------------
``"deterministic"``
    Single-member models (AIFS, GraphCast, FuXi).  WYI array shape:
    ``(n_init, n_steps)``.

``"ensemble"``
    Multi-member models (IFS, NGCM51, GenCast).  WYI array shape:
    ``(n_members, n_init, n_steps)``.

``"climatology"``
    No WYI data needed.  The predicted onset is always May 31 of the
    target year.  Pass ``wyi_arr=None`` and ``t_init`` from any
    reference model that provides initialization times.

Usage example
-------------
>>> from monsoonbench.metrics.wyi_metrics import WYIOnsetMetrics, compute_wyi_metrics
>>> # deterministic model
>>> results = WYIOnsetMetrics.compute_year_metrics(
...     year=2020,
...     gt_onset=gt,          # pd.Timestamp or None
...     t_init=t_init_arr,    # array of pd.Timestamps
...     wyi_arr=wyi,          # shape (n_init, n_steps)
...     model_type="deterministic",
...     verification_window=1,
...     tol=3,
... )
"""

import math
import warnings
from typing import Callable

import numpy as np
import pandas as pd


# Climatological WYI value on June 2 (from MATLAB code comment)
_WYI_CLIM_THRESHOLD = -14.8
_CLIM_ONSET_MONTH = 5
_CLIM_ONSET_DAY = 31
_FORECAST_DAYS = 14  # window is verification_window + 14 total steps


def _apply_movmean7(arr: np.ndarray) -> np.ndarray:
    """Centred 7-day moving average matching MATLAB's movmean(arr, 7)."""
    return (
        pd.Series(arr)
        .rolling(window=7, center=True, min_periods=7)
        .mean()
        .to_numpy()
    )


def _get_ic_stop_idx(t_init: list[pd.Timestamp], gt_onset: pd.Timestamp) -> int:
    """Return the index of the last IC that is strictly before gt_onset.

    Returns 0 if no IC is before the onset (no valid ICs to evaluate).
    """
    diff_days = np.array([(t - gt_onset).days for t in t_init])

    exact = np.where(diff_days == 0)[0]
    if len(exact) > 0:
        return int(exact[0])  # stop before the onset-date IC

    before = np.where(diff_days < 0)[0]
    if len(before) == 0:
        return 0
    # Largest negative diff = closest IC before onset
    return int(before[np.argmax(diff_days[before])]) + 1


def _doy_to_timestamp(year: int, doy: float) -> pd.Timestamp:
    """Convert a (possibly fractional) day-of-year to a pd.Timestamp."""
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=round(doy) - 1)


class WYIOnsetMetrics:
    """Compute WYI-based monsoon onset metrics.

    All methods are static; the class acts as a namespace consistent with
    the existing ``DeterministicOnsetMetrics`` / ``ProbabilisticOnsetMetrics``
    pattern in this package.
    """

    WYI_CLIM_THRESHOLD: float = _WYI_CLIM_THRESHOLD

    # ------------------------------------------------------------------
    # Onset detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_onset_deterministic(
        t_init: list[pd.Timestamp],
        wyi_arr: np.ndarray,
        verification_window: int,
        ic_stop_idx: int,
        wyi_clim_threshold: float = _WYI_CLIM_THRESHOLD,
    ) -> list[pd.Timestamp | None]:
        """Detect WYI onset for each IC of a deterministic model.

        Parameters
        ----------
        t_init:
            Sequence of initialization datetimes (length n_init).
        wyi_arr:
            WYI forecasts, shape ``(n_init, n_steps)``.
        verification_window:
            Lead-time offset (days) before the verification window opens.
        ic_stop_idx:
            Only process ICs 0 … ic_stop_idx-1.
        wyi_clim_threshold:
            WYI value below which onset is declared.

        Returns
        -------
        List of onset ``pd.Timestamp`` (or ``None``) for each IC.
        """
        n_steps = verification_window + _FORECAST_DAYS
        onset_days: list[pd.Timestamp | None] = []

        for i in range(ic_stop_idx):
            wyi_ts = wyi_arr[i, :n_steps].astype(float)
            wyi_movmean = _apply_movmean7(wyi_ts)
            onset_idx = np.where(wyi_movmean < wyi_clim_threshold)[0]
            if len(onset_idx) > 0:
                # MATLAB step array is 1-indexed so step 1 = 1 day after init
                onset_days.append(t_init[i] + pd.Timedelta(days=int(onset_idx[0]) + 1))
            else:
                onset_days.append(None)

        return onset_days

    @staticmethod
    def compute_onset_ensemble(
        t_init: list[pd.Timestamp],
        wyi_arr: np.ndarray,
        verification_window: int,
        ic_stop_idx: int,
        year: int,
        wyi_clim_threshold: float = _WYI_CLIM_THRESHOLD,
        ens_pct_thres: float = 50.0,
    ) -> list[pd.Timestamp | None]:
        """Detect WYI onset for each IC of an ensemble model.

        The ensemble onset is the mean DOY across members that show onset,
        rounded to the nearest day and converted back to a calendar date.
        An IC is considered to show onset only if at least ``ens_pct_thres``
        percent of members detect onset.

        Parameters
        ----------
        t_init:
            Sequence of initialization datetimes (length n_init).
        wyi_arr:
            WYI forecasts, shape ``(n_members, n_init, n_steps)``.
        verification_window:
            Lead-time offset (days) before the verification window opens.
        ic_stop_idx:
            Only process ICs 0 … ic_stop_idx-1.
        year:
            Target year (used for DOY → date conversion).
        wyi_clim_threshold:
            WYI value below which onset is declared.
        ens_pct_thres:
            Percentage (0-100) of members that must show onset for the
            ensemble mean onset to be accepted.

        Returns
        -------
        List of onset ``pd.Timestamp`` (or ``None``) for each IC.
        """
        n_members, _, _ = wyi_arr.shape
        n_steps = verification_window + _FORECAST_DAYS
        mem_thres = math.ceil((ens_pct_thres / 100.0) * n_members)

        # onset_day_doy[i, m] = DOY of onset for IC i, member m (NaN if none)
        onset_day_doy = np.full((ic_stop_idx, n_members), np.nan)

        for m in range(n_members):
            for i in range(ic_stop_idx):
                wyi_ts = wyi_arr[m, i, :n_steps].astype(float)
                wyi_movmean = _apply_movmean7(wyi_ts)
                onset_idx = np.where(wyi_movmean < wyi_clim_threshold)[0]
                if len(onset_idx) > 0:
                    onset_dt = t_init[i] + pd.Timedelta(days=int(onset_idx[0]) + 1)
                    onset_day_doy[i, m] = onset_dt.dayofyear

        ensavg_doy = np.nanmean(onset_day_doy, axis=1)  # shape (ic_stop_idx,)
        n_nonnull = np.sum(~np.isnan(onset_day_doy), axis=1)

        onset_days: list[pd.Timestamp | None] = []
        for i in range(ic_stop_idx):
            if n_nonnull[i] >= mem_thres and not np.isnan(ensavg_doy[i]):
                onset_days.append(_doy_to_timestamp(year, ensavg_doy[i]))
            else:
                onset_days.append(None)

        return onset_days

    # ------------------------------------------------------------------
    # Contingency computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_contingency(
        onset_days: list[pd.Timestamp | None],
        t_init: list[pd.Timestamp],
        gt_onset: pd.Timestamp,
        verification_window: int,
        tol: int,
    ) -> dict:
        """Compute TP/FP/FN/TN and per-IC MAE values.

        Parameters
        ----------
        onset_days:
            Predicted onset per IC (``None`` means no onset detected).
        t_init:
            Initialization datetimes corresponding to *onset_days*.
        gt_onset:
            Ground-truth onset date.
        verification_window:
            Days after init when the validation window starts.
        tol:
            Tolerance in days for a prediction to count as a true positive.

        Returns
        -------
        dict with keys: TP, FP, FN, TN, num_onset, num_no_onset,
        mae_tp (list), mae_fp (list).
        """
        TP = FP = FN = TN = num_onset = num_no_onset = 0
        mae_tp: list[float] = []
        mae_fp: list[float] = []

        for i, onset_day in enumerate(onset_days):
            valid_start = t_init[i] + pd.Timedelta(days=verification_window)
            valid_end = valid_start + pd.Timedelta(days=_FORECAST_DAYS)

            whole_start = t_init[i] + pd.Timedelta(days=1)
            whole_end = t_init[i] + pd.Timedelta(days=verification_window + _FORECAST_DAYS)

            gt_in_whole_window = whole_start <= gt_onset <= whole_end
            if gt_in_whole_window:
                num_onset += 1
            else:
                num_no_onset += 1

            if onset_day is not None:
                if valid_start <= onset_day <= valid_end:
                    abs_diff = abs((onset_day - gt_onset).days)
                    if abs_diff <= tol:
                        TP += 1
                        mae_tp.append(float(abs_diff))
                    else:
                        FP += 1
                        mae_fp.append(float(abs_diff))
            else:
                if gt_in_whole_window:
                    FN += 1
                else:
                    TN += 1

        return dict(
            TP=TP, FP=FP, FN=FN, TN=TN,
            num_onset=num_onset, num_no_onset=num_no_onset,
            mae_tp=mae_tp, mae_fp=mae_fp,
        )

    # ------------------------------------------------------------------
    # Per-year metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_year_metrics(
        year: int,
        gt_onset: pd.Timestamp | None,
        t_init: list[pd.Timestamp],
        wyi_arr: np.ndarray | None,
        model_type: str,
        verification_window: int,
        tol: int,
        ens_pct_thres: float = 50.0,
        wyi_clim_threshold: float = _WYI_CLIM_THRESHOLD,
    ) -> dict:
        """Compute WYI onset metrics for a single year.

        Parameters
        ----------
        year:
            Target year.
        gt_onset:
            Observed WYI onset (``None`` → all NaN outputs).
        t_init:
            Initialization datetimes, already filtered to >= May 1.
        wyi_arr:
            WYI forecast array.  Shape ``(n_init, n_steps)`` for
            ``"deterministic"``, ``(n_members, n_init, n_steps)`` for
            ``"ensemble"``, or ``None`` for ``"climatology"``.
        model_type:
            One of ``"deterministic"``, ``"ensemble"``, ``"climatology"``.
        verification_window:
            Days after init before the validation window opens.
        tol:
            Tolerance in days for TP classification.
        ens_pct_thres:
            Ensemble member threshold percentage (ensemble models only).
        wyi_clim_threshold:
            WYI threshold value for onset detection.

        Returns
        -------
        dict with keys: TP, FP, FN, TN, num_onset, num_no_onset, mae (scalar).
        All values are NaN if *gt_onset* is ``None`` or no valid ICs exist.
        """
        _nan_result = dict(TP=np.nan, FP=np.nan, FN=np.nan, TN=np.nan,
                           num_onset=np.nan, num_no_onset=np.nan, mae=np.nan)

        if gt_onset is None:
            return _nan_result

        ic_stop_idx = _get_ic_stop_idx(list(t_init), gt_onset)
        if ic_stop_idx == 0:
            return _nan_result

        # --- Detect onset per IC ---
        if model_type == "climatology":
            clim_onset = pd.Timestamp(year, _CLIM_ONSET_MONTH, _CLIM_ONSET_DAY)
            onset_days = [clim_onset] * ic_stop_idx
        elif model_type == "deterministic":
            onset_days = WYIOnsetMetrics.compute_onset_deterministic(
                t_init, wyi_arr, verification_window, ic_stop_idx, wyi_clim_threshold
            )
        elif model_type == "ensemble":
            onset_days = WYIOnsetMetrics.compute_onset_ensemble(
                t_init, wyi_arr, verification_window, ic_stop_idx, year,
                wyi_clim_threshold, ens_pct_thres
            )
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                "Expected 'deterministic', 'ensemble', or 'climatology'."
            )

        # --- Contingency metrics ---
        stats = WYIOnsetMetrics._compute_contingency(
            onset_days, list(t_init)[:ic_stop_idx], gt_onset, verification_window, tol
        )

        mae_vals = stats["mae_tp"] + stats["mae_fp"]
        mae = float(np.mean(mae_vals)) if mae_vals else np.nan

        return dict(
            TP=stats["TP"], FP=stats["FP"], FN=stats["FN"], TN=stats["TN"],
            num_onset=stats["num_onset"], num_no_onset=stats["num_no_onset"],
            mae=mae,
        )

    # ------------------------------------------------------------------
    # Multi-year aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_metrics(year_results: list[dict]) -> dict:
        """Aggregate per-year result dicts into summary statistics.

        Parameters
        ----------
        year_results:
            List of dicts returned by :meth:`compute_year_metrics`, one
            per year.

        Returns
        -------
        dict with keys:
            mae_avg, std_er, mae_yr (array), false_alarm, miss_rate
        """
        mae_yr = np.array([r["mae"] for r in year_results], dtype=float)
        tp_arr = np.array([r["TP"] for r in year_results], dtype=float)
        fp_arr = np.array([r["FP"] for r in year_results], dtype=float)
        fn_arr = np.array([r["FN"] for r in year_results], dtype=float)
        tn_arr = np.array([r["TN"] for r in year_results], dtype=float)
        num_onset_arr = np.array([r["num_onset"] for r in year_results], dtype=float)

        mae_avg = float(np.nanmean(mae_yr))
        n_valid = int(np.sum(~np.isnan(mae_yr)))
        std_er = (
            float(np.nanstd(mae_yr, ddof=1) / np.sqrt(n_valid))
            if n_valid > 1
            else 0.0
        )

        denom_far = np.nansum(fp_arr) + np.nansum(tn_arr)
        false_alarm = (
            float(np.nansum(fp_arr) / denom_far) if denom_far > 0 else np.nan
        )

        denom_mr = np.nansum(num_onset_arr)
        miss_rate = (
            float(np.nansum(fn_arr) / denom_mr) if denom_mr > 0 else np.nan
        )

        return dict(
            mae_avg=mae_avg,
            std_er=std_er,
            mae_yr=mae_yr,
            false_alarm=false_alarm,
            miss_rate=miss_rate,
        )


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------

def compute_wyi_metrics(
    yr_ar: np.ndarray,
    verification_window: int,
    tol: int,
    model_type: str,
    data_loader: Callable[[int], tuple],
    wyi_data_dir: str,
    ens_pct_thres: float = 50.0,
    wyi_clim_threshold: float = _WYI_CLIM_THRESHOLD,
) -> dict:
    """Compute WYI onset metrics across multiple years.

    Parameters
    ----------
    yr_ar:
        Array of years to evaluate.
    verification_window:
        Days after IC when the validation window opens.
    tol:
        Tolerance in days for a TP classification.
    model_type:
        One of ``"deterministic"``, ``"ensemble"``, ``"climatology"``.
    data_loader:
        Callable ``data_loader(year) -> (t_init, wyi_arr)`` where
        *t_init* is a sequence of ``pd.Timestamp`` (already filtered to
        >= May 1) and *wyi_arr* has the shape expected for *model_type*.
        For ``"climatology"`` the return value of *wyi_arr* is ignored.
    wyi_data_dir:
        Directory containing ``wyi_daily_{year}.nc`` files.
    ens_pct_thres:
        Ensemble member threshold percentage (ensemble models only).
    wyi_clim_threshold:
        WYI value below which onset is declared.

    Returns
    -------
    dict with keys: mae_avg, std_er, mae_yr, false_alarm, miss_rate.

    Notes
    -----
    Prints a summary to stdout matching the MATLAB script output format.
    """
    from monsoonbench.onset.wyi_onset import get_wyi_onset  # local import to avoid circularity

    year_results = []
    for year in yr_ar:
        gt_onset, _, _ = get_wyi_onset(int(year), wyi_data_dir)
        t_init, wyi_arr = data_loader(int(year))

        # Filter ICs to >= May 1
        may1 = pd.Timestamp(int(year), 5, 1)
        t_init_arr = np.asarray(t_init)
        keep = np.array([t >= may1 for t in t_init_arr])
        t_init_filt = list(t_init_arr[keep])

        if wyi_arr is not None and model_type == "ensemble":
            wyi_arr_filt = wyi_arr[:, keep, :]
        elif wyi_arr is not None:
            wyi_arr_filt = wyi_arr[keep, :]
        else:
            wyi_arr_filt = None

        result = WYIOnsetMetrics.compute_year_metrics(
            year=int(year),
            gt_onset=gt_onset,
            t_init=t_init_filt,
            wyi_arr=wyi_arr_filt,
            model_type=model_type,
            verification_window=verification_window,
            tol=tol,
            ens_pct_thres=ens_pct_thres,
            wyi_clim_threshold=wyi_clim_threshold,
        )
        year_results.append(result)
        print(f"  Year {year} done.")

    summary = WYIOnsetMetrics.aggregate_metrics(year_results)

    model_label = model_type
    print(f"\nWYI onset results for {model_label}")
    print(f"MAE is : {summary['mae_avg']:.4f} \\pm {summary['std_er']:.4f}")
    print(f"FAR is : {summary['false_alarm']:.4f}")
    print(f"Miss rate is : {summary['miss_rate']:.4f}")

    return summary
