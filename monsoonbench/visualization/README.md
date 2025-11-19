# Visualization Utilities

This directory contains the visualization helpers that ship with
MonsoonBench. They currently cover two core use cases:

1. Rendering publication-quality spatial scorecards via
   `plot_spatial_metrics`.
2. Exporting the underlying scorecard data for downstream analysis with the
   data downloader introduced in `data_downloader.py`.

## Data Downloader Overview

`VisualizationDataDownloader` packages the `spatial_metrics` dictionary
produced by `OnsetMetricsBase.create_spatial_far_mr_mae` into annotated
artifacts. It can emit the full dataset or specific metrics in any of the
following formats: NetCDF, CSV, Parquet, or JSON. Auto-generated metadata
captures the grid definition (lat/lon range and resolution), the generation
timestamp, and any user-supplied tags.

```python
from monsoonbench.visualization import VisualizationDataDownloader

downloader = VisualizationDataDownloader.from_spatial_metrics(spatial_metrics)
downloader.save("artifacts/metrics.nc")  # default NetCDF export
```

The convenience wrapper `download_spatial_metrics_data` performs multi-format
exports in one call and is used by the CLI when `--download_dir` is provided:

```python
from monsoonbench.visualization import download_spatial_metrics_data

download_spatial_metrics_data(
    spatial_metrics,
    output_dir="artifacts",
    filename="exp_2019",
    formats=("netcdf", "csv"),
    metadata={"experiment": "graphcast_v1"},
)
```

Both the API and CLI support limiting the exported metrics (e.g.,
`metrics=["mean_mae", "miss_rate"]`) and controlling whether rows that are all
NaN should be dropped (`dropna` vs. `--download_keep_nans`).

## Related CLI Flags

The main CLI exposes download functionality without requiring users to write
additional code. Set the following flags to control behavior:

- `--download_dir PATH`: enables exports and selects the destination folder.
- `--download_formats netcdf csv ...`: optional list of formats (defaults to
  NetCDF). Accepts any subset of `netcdf`, `csv`, `parquet`, `json`.
- `--download_metrics mean_mae false_alarm_rate`: optional subset of metrics to
  persist.
- `--download_keep_nans`: keep rows where all metrics are NaN when writing
  tabular formats.
