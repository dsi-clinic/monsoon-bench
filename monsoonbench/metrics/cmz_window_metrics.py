"""CMZ forecast-window scoring utilities.

This module isolates forecast-window contingency scoring logic that was
previously embedded in notebooks. The functions here operate on onset-pair
tables and return MAE/FAR/MR metrics together with TP/FP/FN/TN counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "WindowScoringConfig",
    "compute_cmz_window_metrics",
    "score_window_contingency",
]


@dataclass(frozen=True)
class WindowScoringConfig:
    """Behavior controls for window-based contingency classification.

    Attributes:
        obs_window_mode: Observation window used when deciding if a prediction
            is in-window (``"bin"`` or ``"cumulative"``).
        classification_mode: Rule for handling predictions outside the current
            window.
        fn_tn_obs_mode: Observation window used for FN/TN decisions when there
            is no valid in-window prediction.
        mr_denom_mode: Denominator definition for miss rate.
        aggregate_mode: Spatial aggregation mode (``"pooled"`` or ``"gridmean"``).
    """

    obs_window_mode: str = "bin"
    classification_mode: str = "strict_window"
    fn_tn_obs_mode: str = "bin"
    mr_denom_mode: str = "obs_in_bin"
    aggregate_mode: str = "gridmean"


def score_window_contingency(
    onset_df: pd.DataFrame,
    start_day: int,
    end_day: int,
    tolerance_days: int,
    scoring: WindowScoringConfig | None = None,
) -> dict[str, float]:
    """Score one forecast window on one onset table.

    Args:
        onset_df: Onset pair table. Expected columns include ``init_time``,
            ``obs_onset_date``, and ``onset_date``.
        start_day: First lead day in the forecast bin.
        end_day: Last lead day in the forecast bin.
        tolerance_days: Allowed absolute day error to count as TP.
        scoring: Optional scoring configuration. Defaults to
            :class:`WindowScoringConfig`.

    Returns:
        Dictionary containing MAE/FAR/MR, contingency counts, and row counts.
    """
    cfg = scoring or WindowScoringConfig()
    if onset_df.empty:
        return {
            "mae": np.nan,
            "far": np.nan,
            "mr": np.nan,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "n_rows": 0,
            "n_obs_in": 0,
            "n_obs_in_bin": 0,
            "n_obs_in_cumulative_to_end": 0,
            "n_ignored": 0,
        }

    onset_table = onset_df.copy()
    onset_table["init_dt"] = pd.to_datetime(onset_table["init_time"], errors="coerce")
    onset_table["obs_dt"] = pd.to_datetime(
        onset_table["obs_onset_date"], errors="coerce"
    )
    onset_table["pred_dt"] = pd.to_datetime(onset_table["onset_date"], errors="coerce")
    onset_table = onset_table.dropna(subset=["init_dt", "obs_dt"])

    onset_table["obs_lead"] = (onset_table["obs_dt"] - onset_table["init_dt"]).dt.days
    onset_table["pred_lead"] = (onset_table["pred_dt"] - onset_table["init_dt"]).dt.days
    onset_table["obs_in_bin"] = onset_table["obs_lead"].between(
        start_day, end_day, inclusive="both"
    )
    onset_table["obs_in_cumulative_to_end"] = onset_table["obs_lead"].between(
        1, end_day, inclusive="both"
    )

    if cfg.obs_window_mode == "bin":
        obs_for_pred = "obs_in_bin"
        pred_in_series = pd.notna(onset_table["pred_dt"]) & onset_table[
            "pred_lead"
        ].between(start_day, end_day, inclusive="both")
    elif cfg.obs_window_mode == "cumulative":
        obs_for_pred = "obs_in_cumulative_to_end"
        pred_in_series = pd.notna(onset_table["pred_dt"]) & onset_table[
            "pred_lead"
        ].between(1, end_day, inclusive="both")
    else:
        raise ValueError(f"Unsupported obs_window_mode: {cfg.obs_window_mode}")

    if cfg.fn_tn_obs_mode not in ("bin", "cumulative_to_end"):
        raise ValueError(f"Unsupported fn_tn_obs_mode: {cfg.fn_tn_obs_mode}")
    obs_for_no_pred = (
        "obs_in_bin" if cfg.fn_tn_obs_mode == "bin" else "obs_in_cumulative_to_end"
    )

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    abs_errors: list[int] = []

    for idx, row in onset_table.iterrows():
        has_pred = pd.notna(row["pred_dt"])
        pred_in = bool(pred_in_series.loc[idx])

        if pred_in:
            abs_err = abs((row["pred_dt"] - row["obs_dt"]).days)
            if abs_err <= tolerance_days:
                tp += 1
            else:
                fp += 1
            abs_errors.append(abs_err)
            continue

        if cfg.classification_mode == "strict_window":
            if bool(row[obs_for_pred]):
                fn += 1
            else:
                tn += 1
            continue

        if cfg.classification_mode == "legacy_skip_pred_out":
            if not has_pred:
                if bool(row[obs_for_no_pred]):
                    fn += 1
                else:
                    tn += 1
            continue

        if cfg.classification_mode == "legacy_plus_tn_noobs":
            if not has_pred:
                if bool(row[obs_for_no_pred]):
                    fn += 1
                else:
                    tn += 1
            elif not bool(row[obs_for_no_pred]):
                tn += 1
            continue

        raise ValueError(f"Unsupported classification_mode: {cfg.classification_mode}")

    mae = float(np.mean(abs_errors)) if abs_errors else np.nan
    far = float(fp / (fp + tn) * 100.0) if (fp + tn) > 0 else np.nan

    n_obs_in_bin = int(onset_table["obs_in_bin"].sum())
    n_obs_in_cumulative = int(onset_table["obs_in_cumulative_to_end"].sum())

    if cfg.mr_denom_mode == "obs_in_bin":
        mr_denom = n_obs_in_bin
    elif cfg.mr_denom_mode == "obs_in_cumulative_to_end":
        mr_denom = n_obs_in_cumulative
    elif cfg.mr_denom_mode == "tp_fn":
        mr_denom = tp + fn
    else:
        raise ValueError(f"Unsupported mr_denom_mode: {cfg.mr_denom_mode}")

    mr = float(fn / mr_denom * 100.0) if mr_denom > 0 else np.nan

    return {
        "mae": mae,
        "far": far,
        "mr": mr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_rows": int(len(onset_table)),
        "n_obs_in": int(onset_table[obs_for_pred].sum()),
        "n_obs_in_bin": n_obs_in_bin,
        "n_obs_in_cumulative_to_end": n_obs_in_cumulative,
        "n_ignored": int(len(onset_table) - (tp + fp + fn + tn)),
    }


def compute_cmz_window_metrics(
    onset_df: pd.DataFrame,
    start_day: int,
    end_day: int,
    tolerance_days: int,
    scoring: WindowScoringConfig | None = None,
) -> dict[str, float]:
    """Compute CMZ metrics for one window using pooled or grid-mean aggregation.

    Args:
        onset_df: Onset pair table with ``lat`` and ``lon`` columns.
        start_day: First lead day in the forecast bin.
        end_day: Last lead day in the forecast bin.
        tolerance_days: Allowed absolute day error for TP.
        scoring: Optional scoring configuration. Defaults to
            :class:`WindowScoringConfig`.

    Returns:
        Dictionary of metrics and contingency counts. Includes ``n_grids`` when
        ``aggregate_mode="gridmean"``.
    """
    cfg = scoring or WindowScoringConfig()
    pooled = score_window_contingency(
        onset_df, start_day, end_day, tolerance_days, scoring=cfg
    )

    if cfg.aggregate_mode == "pooled":
        return pooled
    if cfg.aggregate_mode != "gridmean":
        raise ValueError(f"Unsupported aggregate_mode: {cfg.aggregate_mode}")
    if onset_df.empty:
        pooled["n_grids"] = 0
        return pooled

    per_grid: list[dict[str, float]] = []
    for _, group_df in onset_df.groupby(["lat", "lon"], dropna=False):
        gm = score_window_contingency(
            group_df,
            start_day,
            end_day,
            tolerance_days,
            scoring=cfg,
        )
        per_grid.append(gm)

    gdf = pd.DataFrame(per_grid)
    pooled["mae"] = float(gdf["mae"].mean(skipna=True))
    pooled["far"] = float(gdf["far"].mean(skipna=True))
    pooled["mr"] = float(gdf["mr"].mean(skipna=True))
    pooled["n_grids"] = int(len(gdf))
    return pooled
