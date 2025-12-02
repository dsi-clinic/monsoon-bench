# monsoonbench/data/loaders/__init__.py
from .deterministic import DeterministicForecastLoader  # noqa: F401
from .imd import IMDRainLoader  # noqa: F401
from .probabilistic import ProbabilisticForecastLoader  # noqa: F401
from .threshold import ThresholdLoader  # noqa: F401
