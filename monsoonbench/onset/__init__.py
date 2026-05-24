"""Monsoon onset detection algorithms."""

from monsoonbench.onset.detection import detect_observed_onset
from monsoonbench.onset.wyi_onset import get_wyi_onset

__all__ = [
    "detect_observed_onset",
    "get_wyi_onset",
]
