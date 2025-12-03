"""Utilities for exporting visualization-ready data.

The plotting functions in :mod:`monsoonbench.visualization` typically operate on
the ``spatial_metrics`` dictionary returned by
``OnsetMetricsBase.create_spatial_far_mr_mae``.  While this is convenient for
rendering static figures, downstream tooling often needs direct access to the
values that were plotted.  This module provides a lightweight downloader that
standardizes how those intermediate results are packaged, annotated, and saved
to disk so they can be fed into other visualization or analysis workflows.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "VisualizationDataDownloader",
    "download_spatial_metrics_data",
]


def _normalize_format(fmt: str | Path | None) -> str:
    """Coerce user-specified output formats to one of the supported keys."""
    if fmt is None:
        return "netcdf"
    fmt_str = str(fmt).lower().strip()
    if fmt_str.startswith("."):
        fmt_str = fmt_str[1:]
    mapping = {
        "nc": "netcdf",
        "cdf": "netcdf",
    }
    return mapping.get(fmt_str, fmt_str)


def _infer_metadata(
    dataset: xr.Dataset, metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Build a metadata dictionary describing the dataset."""
    lat_range: list[float] | None = None
    lon_range: list[float] | None = None
    lat_resolution: float | None = None
    lon_resolution: float | None = None

    if "lat" in dataset.coords:
        lat_values = np.asarray(dataset["lat"].values, dtype=float)
        if lat_values.size:
            lat_range = [float(np.nanmin(lat_values)), float(np.nanmax(lat_values))]
            if lat_values.size > 1:
                lat_diff = np.diff(np.sort(lat_values))
                with np.errstate(invalid="ignore"):
                    lat_resolution_val = np.nanmin(np.abs(lat_diff))
                lat_resolution = (
                    float(lat_resolution_val)
                    if np.isfinite(lat_resolution_val)
                    else None
                )

    if "lon" in dataset.coords:
        lon_values = np.asarray(dataset["lon"].values, dtype=float)
        if lon_values.size:
            lon_range = [float(np.nanmin(lon_values)), float(np.nanmax(lon_values))]
            if lon_values.size > 1:
                lon_diff = np.diff(np.sort(lon_values))
                with np.errstate(invalid="ignore"):
                    lon_resolution_val = np.nanmin(np.abs(lon_diff))
                lon_resolution = (
                    float(lon_resolution_val)
                    if np.isfinite(lon_resolution_val)
                    else None
                )

    timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    base_metadata: dict[str, Any] = {
        "generated_at": timestamp,
        "generated_by": "monsoonbench.visualization.data_downloader",
        "variables": sorted(dataset.data_vars),
        "lat_range": lat_range,
        "lon_range": lon_range,
        "lat_resolution": lat_resolution,
        "lon_resolution": lon_resolution,
    }

    if metadata:
        base_metadata.update({key: metadata[key] for key in metadata})
    return base_metadata


class VisualizationDataDownloader:
    """Helper for exporting plotting data in common, downstream-friendly formats."""

    def __init__(
        self: Self,
        dataset: xr.Dataset,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the downloader with a dataset and optional metadata."""
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("dataset must be an xarray.Dataset")
        self._dataset = dataset
        self._metadata = _infer_metadata(dataset, metadata)

    @classmethod
    def from_spatial_metrics(
        cls: type[Self],
        spatial_metrics: Mapping[str, xr.DataArray] | xr.Dataset,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Construct a downloader from the outputs of ``create_spatial_far_mr_mae``."""
        if isinstance(spatial_metrics, xr.Dataset):
            dataset = spatial_metrics
        else:
            dataset = xr.Dataset(spatial_metrics)
        return cls(dataset=dataset, metadata=metadata)

    @property
    def available_metrics(self: Self) -> list[str]:
        """Return the names of the metrics contained in the dataset."""
        return list(self._dataset.data_vars)

    @property
    def metadata(self: Self) -> dict[str, Any]:
        """Metadata describing the visualization dataset."""
        return dict(self._metadata)

    def _select_metrics(self: Self, metrics: Sequence[str] | None) -> xr.Dataset:
        """Return a dataset with the requested metric subset."""
        if metrics is None:
            return self._dataset
        missing = [
            metric for metric in metrics if metric not in self._dataset.data_vars
        ]
        if missing:
            raise KeyError(f"Metrics not found in dataset: {missing}")
        return self._dataset[list(metrics)]

    def to_dataset(self: Self, metrics: Sequence[str] | None = None) -> xr.Dataset:
        """Return the dataset, annotated with metadata."""
        dataset = self._select_metrics(metrics)
        ds_copy = dataset.copy()
        ds_copy.attrs.update(self._metadata)
        return ds_copy

    def to_dataframe(
        self: Self,
        metrics: Sequence[str] | None = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """Return a tidy ``pandas.DataFrame`` representation of the dataset."""
        ds = self._select_metrics(metrics)
        dataframe = ds.to_dataframe().reset_index()
        metric_columns = list(ds.data_vars)
        if dropna and metric_columns:
            dataframe = dataframe.dropna(subset=metric_columns, how="all")
        return dataframe

    def save(
        self: Self,
        path: str | Path,
        *,
        format: str | Path | None = None,
        metrics: Sequence[str] | None = None,
        dropna: bool = True,
    ) -> Path:
        """Persist the dataset to disk in a supported format."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = _normalize_format(format or output_path.suffix)

        if fmt not in {"netcdf", "csv", "parquet", "json"}:
            raise ValueError(
                f"Unsupported format '{fmt}'. Expected one of 'netcdf', 'csv', 'parquet', 'json'."
            )

        if fmt == "netcdf":
            ds = self.to_dataset(metrics=metrics)
            ds.to_netcdf(output_path)
        elif fmt in {"csv", "parquet", "json"}:
            dataframe = self.to_dataframe(metrics=metrics, dropna=dropna)
            if fmt == "csv":
                dataframe.to_csv(output_path, index=False)
            elif fmt == "parquet":
                dataframe.to_parquet(output_path, index=False)
            else:
                payload = {
                    "metadata": self.metadata,
                    "data": dataframe.to_dict(orient="records"),
                }
                output_path.write_text(json.dumps(payload, indent=2))

        return output_path


def download_spatial_metrics_data(
    spatial_metrics: Mapping[str, xr.DataArray] | xr.Dataset,
    output_dir: str | Path,
    *,
    filename: str = "spatial_metrics",
    formats: Iterable[str | Path] = ("netcdf",),
    metadata: Mapping[str, Any] | None = None,
    metrics: Sequence[str] | None = None,
    dropna: bool = True,
) -> list[Path]:
    """Persist spatial metrics to disk in one or more formats."""
    downloader = VisualizationDataDownloader.from_spatial_metrics(
        spatial_metrics=spatial_metrics,
        metadata=metadata,
    )
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix_map = {
        "netcdf": ".nc",
        "csv": ".csv",
        "parquet": ".parquet",
        "json": ".json",
    }

    saved_paths: list[Path] = []
    for fmt in formats:
        normalized_fmt = _normalize_format(fmt)
        if normalized_fmt not in suffix_map:
            raise ValueError(
                f"Format '{fmt}' is not supported. Choose from {sorted(suffix_map)}."
            )
        suffix = suffix_map[normalized_fmt]
        path = target_dir / f"{filename}{suffix}"
        downloader.save(
            path,
            format=normalized_fmt,
            metrics=metrics,
            dropna=dropna,
        )
        saved_paths.append(path)

    return saved_paths
