# MonsoonBench
*A unified, reproducible benchmarking framework for Indian monsoon onset prediction.*

MonsoonBench provides a standardized workflow for loading rainfall and forecast datasets, computing monsoon onset, and evaluating forecasting skill across space and time.  
It is designed for climate researchers, forecasters, and data scientists aiming to compare deterministic, probabilistic, and climatology-based onset models using consistent methods.

The framework follows WeatherBench-style principles: **clean APIs, reproducible configuration, modular components, and shareable outputs.**

---

## Documentation Overview

MonsoonBench includes detailed module-specific guides. Use the links below to navigate the documentation.

### Core Package Overview & Pipeline
High-level explanation of the evaluation pipeline, CLI interface, onset metrics, and NetCDF outputs.  
**Path:** `monsoonbench/README.md`  
[Open Metrics & Pipeline README](monsoonbench/README.md)

---

### Data Loading Guide
How to load IMD rainfall, deterministic/probabilistic forecasts, and threshold datasets using the unified API.  
**Path:** `monsoonbench/data/dataloader_quickstart.md`  
[Open DataLoader QuickStart](monsoonbench/data/dataloader_quickstart.md)

---

### Visualization & Metric Export Tools
How to generate spatial scorecards and export skill metrics in NetCDF, CSV, Parquet, or JSON formats.  
**Path:** `monsoonbench/visualization/README.md`  
[Open Visualization README](monsoonbench/visualization/README.md)

---

### Examples (Configs, Scripts, Notebooks)
Example YAML configs, runnable scripts, and tutorial notebooks demonstrating end-to-end usage.  
**Path:** `examples/README.md`  
[Open Examples README](examples/README.md)

---

## Installation

MonsoonBench is available on TestPyPI for pre-release testing:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    monsoonbench==0.1.0
```

### Verify installation:

```bash
monsoonbench --help
```

## Minimal Example (Python)

```python
from monsoonbench.data import load
from monsoonbench.metrics import DeterministicOnsetMetrics

# Load IMD rainfall
imd = load(
    "imd_rain",
    root="/path/to/IMD/2p0",
    years=range(2015, 2020),
)

# Run deterministic onset evaluation
metrics = DeterministicOnsetMetrics()
results = metrics.compute_metrics_multiple_years(
    rainfall=imd,
    forecast=imd,           # placeholder example
    years=range(2015, 2020),
    tolerance_days=3,
)

results.to_netcdf("results.nc")
```

This illustrates the core workflow: load → evaluate → save.

## Repository Structure

```
monsoon-bench/
│
├── monsoonbench/ # Core package
│ ├── data/ # Dataloaders
│ │ └── dataloader_quickstart.md
│ ├── metrics/ # Onset detection + evaluation pipeline
│ ├── visualization/ # Scorecards + metric downloaders
│ │ └── README.md
│ ├── README.md # Module-level pipeline documentation
│ └── ...
│
├── examples/ # Configs, scripts, tutorial notebooks
│ └── README.md
│
├── tests/ # Unit tests
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Development Process with branches

Each team member created their own branch to implement specific fixes or features, such as the data loader, data downloader, and visualizations. We regularly merged these branches during TA meetings to ensure that the codebase stayed consistent and that everyone remained aligned on progress and design decisions.
