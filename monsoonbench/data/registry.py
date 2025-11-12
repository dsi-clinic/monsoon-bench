# src/monsoonbench/data/registry.py
from __future__ import annotations
from typing import Dict, Type, Any
from .base import BaseLoader

_REGISTRY: Dict[str, Type[BaseLoader]] = {}

def register_loader(name: str):
    """Decorator to register a loader class under a string key."""
    def deco(cls: Type[BaseLoader]):
        if name in _REGISTRY:
            raise ValueError(f"Loader '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return deco

def get_registered() -> Dict[str, Type[BaseLoader]]:
    """Return a copy of the registry (for help/diagnostics)."""
    return dict(_REGISTRY)

def load(name: str, **kwargs) -> Any:
    """Instantiate the registered loader by name and call .load()."""
    try:
        loader_cls = _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown loader '{name}'. Available: [{available}]")
    return loader_cls.from_kwargs(**kwargs).load()
