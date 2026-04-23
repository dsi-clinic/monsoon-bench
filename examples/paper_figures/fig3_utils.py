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

from examples.paper_figures.data_utils import (
    YEAR_RANGES, PROBABILISTIC_MODELS
)

DEFAULT_WINDOW_BINS: list[tuple[int, int]] = [
    (1, 6),
    (6, 11),
    (11, 16),
    (16, 21),
    (21, 26),
    (26, 31),
]

def get_clim_window_data(
    config: dict[str, str],
):
    c_metrics = ClimatologyOnsetMetrics()

    clim_data = []

    for (lower, upper) in DEFAULT_WINDOW_BINS:
        if lower < 11:
            tol_days = 2
        elif lower < 21:
            tol_days = 3
        else:
            tol_days = 5
        
        multi_yr_metrics, multi_onset_dy = c_metrics.compute_climatology_baseline_multiple_years(        
            years=config["years"],
            imd_folder=config["imd_folder"],
            thres_file=config["thres_file"],
            tolerance_days=tol_days, #Tolerance window
            verification_window=lower, 
            forecast_days=upper,
            max_forecast_day=upper,
            mok=True,
            onset_window=5,
            mok_month=6,
            mok_day=2,
        )
        clim_plot_data = c_metrics.create_spatial_far_mr_mae(
            multi_yr_metrics, dict.fromkeys(config["years"], multi_onset_dy)
        )

        clim_data.append(clim_plot_data)
        
    return clim_data

