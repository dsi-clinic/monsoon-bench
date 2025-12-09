"""Data loading and validation module."""
# ruff: noqa: F401

# Import concrete loaders so their decorators execute at import time:
from .loaders import deterministic as _det
from .loaders import imd as _imd
from .loaders import probabilistic as _prob
from .loaders import threshold as _thr
from .registry import get_registered, load, register_loader
