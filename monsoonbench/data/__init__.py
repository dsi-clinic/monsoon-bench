"""Data loading and validation module."""
# ruff: noqa: F401

# Import concrete loaders so their decorators execute at import time:
from .loaders import imd as _imd
from .registry import get_registered, load, register_loader

# ------The following is going to be implemented in the next step
# from .loaders import deterministic as _det
# from .loaders import probabilistic as _prob
# from .loaders import threshold as _thr
# from .loaders import shapefile as _shp
