"""Registry for data loaders.

This module provides a registry system for registering and loading data loaders.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import xarray as xr

from .base import BaseLoader

_REGISTRY: dict[str, type[BaseLoader]] = {}


def register_loader(name: str) -> Callable[[type[BaseLoader]], type[BaseLoader]]:
    """Decorator to register a loader class under a string key."""

    def deco(cls: type[BaseLoader]) -> type[BaseLoader]:
        if name in _REGISTRY:
            raise ValueError(f"Loader '{name}' already registered")
        _REGISTRY[name] = cls
        return cls

    return deco


def get_registered() -> dict[str, type[BaseLoader]]:
    """Return a copy of the registry (for help/diagnostics)."""
    return dict(_REGISTRY)


def load(name: str, **kwargs) -> xr.Dataset | xr.DataArray:
    """Instantiate the registered loader by name and call .load()."""
    try:
        loader_cls = _REGISTRY[name]
    except KeyError as err:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown loader '{name}'. Available: [{available}]") from err
    return loader_cls.from_kwargs(**kwargs).load()
