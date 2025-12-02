"""Data loading and validation module."""
# ruff: noqa: F401

# ------The following is going to be implemented in the next step
from .loaders import deterministic as _det

# Import concrete loaders so their decorators execute at import time:
from .loaders import imd as _imd
from .loaders import probabilistic as _prob
from .loaders import threshold as _thr
from .registry import get_registered, load, register_loader
# from .loaders import shapefile as _shp
