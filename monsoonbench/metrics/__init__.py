"""Onset metrics computation module."""

from monsoonbench.metrics.base import OnsetMetricsBase
from monsoonbench.metrics.climatology import ClimatologyOnsetMetrics
from monsoonbench.metrics.cmz_window_metrics import (
    WindowScoringConfig,
    compute_cmz_window_metrics,
    score_window_contingency,
)
from monsoonbench.metrics.deterministic import DeterministicOnsetMetrics
from monsoonbench.metrics.probabilistic import ProbabilisticOnsetMetrics
from monsoonbench.metrics.wyi_metrics import WYIOnsetMetrics, compute_wyi_metrics

__all__ = [
    "OnsetMetricsBase",
    "DeterministicOnsetMetrics",
    "ProbabilisticOnsetMetrics",
    "ClimatologyOnsetMetrics",
    "WindowScoringConfig",
    "score_window_contingency",
    "compute_cmz_window_metrics",
    "WYIOnsetMetrics",
    "compute_wyi_metrics",
]
