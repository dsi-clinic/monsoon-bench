# Utility Modules

This directory contains reusable helper modules that are shared by notebooks and analysis scripts.

## Files

- `onset_timeseries.py`
  - Utilities for exploratory onset diagnostics and cross-type comparison plots.
  - Provides notebook-facing functions such as:
    - `find_repo_root`
    - `default_cross_type_model_configs`
    - `compute_onset_diagnostics`
    - `plot_onset_timeseries`
    - `build_cross_type_onset_timeseries`
    - `summarize_error_metrics`
    - `plot_cross_type_onset_comparison`
- `__init__.py`
  - Re-exports selected utility functions for convenient imports.
