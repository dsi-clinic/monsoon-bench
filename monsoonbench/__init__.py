"""MonsoonBench: Benchmark Framework for Indian Monsoon Onset Prediction.

A Python package for evaluating AI-driven and traditional weather models
on the task of predicting the spatio-temporal onset of the Indian monsoon.

Basic Usage:
    >>> from monsoonbench import DataLoader, DeterministicOnsetMetrics
    >>>
    >>> # Load data
    >>> loader = DataLoader.from_config("config.yaml")
    >>> obs = loader.load_observations(years=[2020, 2021])
    >>>
    >>> # Compute metrics
    >>> metrics = DeterministicOnsetMetrics(loader.config)
    >>> results = metrics.compute_metrics()
    >>> results.to_netcdf("output/metrics.nc")

Modules:
    - data: Data loading and validation
    - metrics: Onset metrics computation (MAE, FAR, MR, Brier, RPS, AUC)
    - onset: Onset detection algorithms
    - visualization: Plotting and scorecards
    - config: Configuration management
"""

from monsoonbench._version import __version__

# Public API
from monsoonbench.config import load_config

# from monsoonbench.data import DataLoader  # Zhenfei Commented this line out and add the following line
from .data import load, get_registered, register_loader

from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    ProbabilisticOnsetMetrics,
)
from monsoonbench.visualization.spatial import plot_spatial_metrics
from monsoonbench.cli.main import main
from monsoonbench.data import load
# from monsoonbench.onset import detect_onset  # TODO: Implement if needed
# from monsoonbench.visualization import create_scorecard  # TODO: Implement if needed

__all__ = [
    "__version__",
    "load",
    "get_registered",
    "register_loader",
    "DeterministicOnsetMetrics",
    "ProbabilisticOnsetMetrics",
    "ClimatologyOnsetMetrics",
    "load_config",
    "plot_spatial_metrics",
    # "detect_onset",  # TODO: Implement if needed
    # "create_scorecard",  # TODO: Implement if needed
]
