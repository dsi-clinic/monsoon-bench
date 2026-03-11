"""Utility modules used across notebooks and evaluation workflows."""

from monsoonbench.utils.onset_timeseries import (
    build_cross_type_onset_timeseries,
    compute_onset_diagnostics,
    default_cross_type_model_configs,
    find_repo_root,
    load_threshold_with_fix,
    plot_cross_type_onset_comparison,
    plot_onset_timeseries,
    standardize_rainfall_dims,
    summarize_error_metrics,
)

__all__ = [
    "build_cross_type_onset_timeseries",
    "compute_onset_diagnostics",
    "default_cross_type_model_configs",
    "find_repo_root",
    "load_threshold_with_fix",
    "plot_cross_type_onset_comparison",
    "plot_onset_timeseries",
    "standardize_rainfall_dims",
    "summarize_error_metrics",
]
