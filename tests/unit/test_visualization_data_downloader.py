"""Tests for the visualization data downloader utilities."""

from __future__ import annotations

import numpy as np
import xarray as xr

from monsoonbench.visualization.data_downloader import (
    VisualizationDataDownloader,
    download_spatial_metrics_data,
)


def _sample_spatial_metrics() -> dict[str, xr.DataArray]:
    """Create a minimal set of spatial metrics for testing."""
    lats = np.array([10.0, 12.0], dtype=float)
    lons = np.array([70.0, 72.0], dtype=float)
    mae = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, np.nan]], dtype=float),
        coords=[("lat", lats), ("lon", lons)],
        name="mean_mae",
    )
    far = xr.DataArray(
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        coords=[("lat", lats), ("lon", lons)],
        name="false_alarm_rate",
    )
    miss_rate = xr.DataArray(
        np.array([[0.5, 0.6], [0.7, 0.8]], dtype=float),
        coords=[("lat", lats), ("lon", lons)],
        name="miss_rate",
    )
    return {
        "mean_mae": mae,
        "false_alarm_rate": far,
        "miss_rate": miss_rate,
    }


def test_downloader_from_spatial_metrics_infers_metadata():
    """The downloader should build a dataset and metadata from spatial metrics."""
    downloader = VisualizationDataDownloader.from_spatial_metrics(
        _sample_spatial_metrics(),
        metadata={"source_model": "demo"},
    )

    assert set(downloader.available_metrics) == {
        "mean_mae",
        "false_alarm_rate",
        "miss_rate",
    }
    meta = downloader.metadata
    assert meta["source_model"] == "demo"
    assert meta["lat_range"] == [10.0, 12.0]
    assert meta["lon_range"] == [70.0, 72.0]


def test_to_dataframe_filters_metrics():
    """Dataframes should only include the requested metrics."""
    downloader = VisualizationDataDownloader.from_spatial_metrics(
        _sample_spatial_metrics(),
    )
    result_df = downloader.to_dataframe(metrics=["mean_mae"])

    assert "mean_mae" in result_df.columns
    assert "false_alarm_rate" not in result_df.columns
    # Two lat-lon cells contain non-null MAE values
    assert len(result_df) == 3


def test_save_creates_files(tmp_path):
    """Saving the dataset should create files for supported formats."""
    downloader = VisualizationDataDownloader.from_spatial_metrics(
        _sample_spatial_metrics(),
    )
    nc_path = tmp_path / "metrics.nc"
    csv_path = tmp_path / "metrics.csv"

    downloader.save(nc_path)
    downloader.save(csv_path, format="csv")

    assert nc_path.exists()
    assert csv_path.exists()


def test_download_spatial_metrics_data(tmp_path):
    """High-level helper should create artifacts for each requested format."""
    paths = download_spatial_metrics_data(
        _sample_spatial_metrics(),
        output_dir=tmp_path,
        filename="spatial",
        formats=("netcdf", "csv"),
        metadata={"experiment": "unit-test"},
        metrics=["mean_mae", "miss_rate"],
    )
    saved = {path.name for path in paths}
    assert saved == {"spatial.nc", "spatial.csv"}
    for path in paths:
        assert path.exists()
