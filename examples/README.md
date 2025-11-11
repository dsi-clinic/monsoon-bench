# MonsoonBench Examples

This directory contains example configurations, scripts, and notebooks demonstrating how to use MonsoonBench.

## Directory Structure

- **`configs/`** - Example YAML configuration files
  - `deterministic_graphcast.yaml` - Deterministic forecast evaluation
  - `probabilistic_ecmwf.yaml` - Ensemble forecast evaluation
  - `climatology_baseline.yaml` - Climatology baseline

- **`scripts/`** - Example Python scripts
  - `run_deterministic_eval.py` - Evaluate deterministic forecasts
  - `run_probabilistic_eval.py` - Evaluate ensemble forecasts
  - `generate_scorecard.py` - Create multi-model comparison scorecard

- **`notebooks/`** - Jupyter notebook tutorials
  - `01_basic_usage.ipynb` - Getting started
  - `02_deterministic_forecast.ipynb` - Deterministic example
  - `03_probabilistic_forecast.ipynb` - Probabilistic example

## Quick Start

### Using Configuration Files

```bash
# Run with example config
monsoonbench --config examples/configs/deterministic_graphcast.yaml
```

### Using Python API

```python
from monsoonbench import DataLoader, DeterministicOnsetMetrics

# Load example configuration
loader = DataLoader.from_config("examples/configs/deterministic_graphcast.yaml")

# Compute metrics
metrics = DeterministicOnsetMetrics(loader.config)
results = metrics.compute_metrics()
```

## Data Requirements

These examples assume you have:
- IMD rainfall data in the path specified in the config
- Model forecast data in the appropriate format

See `docs/user_guide/data_preparation.md` for data format requirements.

## Status

🚧 **Under Construction** - Examples are being developed alongside the package.
