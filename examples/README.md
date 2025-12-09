# MonsoonBench Examples

This directory contains example notebooks demonstrating how to use MonsoonBench for monsoon onset prediction evaluation.

## Directory Structure

- **`demo_notebooks/`** - Interactive Jupyter notebook demonstrations
  - `package_walkthrough.ipynb` - Basic package usage and workflow
  - `researcher_use_case_benchmark.ipynb` - Comprehensive benchmarking example
  - `compare_models.ipynb` - Multi-model comparison
  - `outputs/` - Generated visualizations and results

- **`exploratory_notebooks/`** - Development and experimental notebooks

## Quick Start

### 1. Package Walkthrough (Recommended Starting Point)

Open `demo_notebooks/package_walkthrough.ipynb` to learn:
- Loading IMD rainfall data
- Computing onset metrics for deterministic models
- Creating spatial visualizations
- Comparing multiple models

### 2. Researcher Use Case Benchmark

For a complete benchmarking workflow, see `demo_notebooks/researcher_use_case_benchmark.ipynb`:
- Benchmark FuXi model against AIFS and Graphcast baselines
- MOK date sensitivity analysis
- Forecast lead time degradation analysis
- Model agreement/disagreement visualization
- Comprehensive spatial and temporal metrics

This notebook demonstrates production-ready analysis suitable for research papers and presentations.

## Using the Notebooks

### In Docker (Recommended)

```bash
# Start Jupyter from the Docker container
docker run -it --rm \
  -v $(pwd):/project \
  -p 8888:8888 \
  monsoonbench \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

Then open the provided URL in your browser and navigate to `examples/demo_notebooks/`.

### Locally with uv

```bash
# From the project root
uv run jupyter lab examples/demo_notebooks/
```

## Data Requirements

These examples require:

1. **IMD Rainfall Data** (`data/imd_rainfall_data/4p0/`)
   - Gridded daily rainfall observations
   - Format: NetCDF with dimensions (time, lat, lon)

2. **Model Forecast Data** (`data/model_forecast_data/`)
   - FuXi: `fuxi/output_daily_paper_0z_4p0/tp_lsm/`
   - AIFS: `aifs/daily_0z/tp_4p0_lsm/`
   - Graphcast: `graphcast37/output_twice_weekly_paper_0z_4p0/tp_lsm/`
   - Format: NetCDF with dimensions (init_time, step, lat, lon) for deterministic
   - Format: NetCDF with dimensions (init_time, step, lat, lon, member) for ensemble

3. **Onset Threshold** (`data/imd_onset_threshold/`)
   - Pre-computed climatological thresholds
   - File: `mwset4x4.nc4`

4. **India Shapefile** (`data/ind_map_shpfile/`)
   - For spatial visualization boundaries
   - File: `india_shapefile.shp`

## Python API Example

```python
from monsoonbench.metrics import DeterministicOnsetMetrics
from monsoonbench.visualization import create_model_comparison_table

# Initialize metrics calculator
metrics = DeterministicOnsetMetrics()

# Compute metrics for multiple years
df, onset_data = metrics.compute_metrics_multiple_years(
    years=[2019, 2020, 2021, 2022],
    model_forecast_dir="data/model_forecast_data/fuxi/...",
    imd_folder="data/imd_rainfall_data/4p0",
    thres_file="data/imd_onset_threshold/mwset4x4.nc4",
    tolerance_days=3,
    verification_window=1,
    forecast_days=15,
)

# Create spatial metrics
spatial = metrics.create_spatial_far_mr_mae(df, onset_data)

# Generate comparison table
comparison = create_model_comparison_table({"FuXi": spatial})
print(comparison)
```

## Key Metrics

The package computes three primary onset prediction metrics:

- **MAE (Mean Absolute Error)**: Average error in onset date prediction (days)
- **FAR (False Alarm Rate)**: Percentage of false onset predictions
- **MR (Miss Rate)**: Percentage of missed onset events

Lower values indicate better performance for all three metrics.

## Output Examples

The notebooks generate:
- Spatial heatmaps of model performance
- Model comparison charts (MAE, FAR, MR)
- Sensitivity analysis plots (MOK dates, verification windows)
- Model agreement/disagreement visualizations

All outputs are saved to `demo_notebooks/outputs/` as high-resolution PNG files suitable for presentations and publications.
