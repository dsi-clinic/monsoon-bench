# monsoonbench/data/loaders/thresholds.py

from __future__ import annotations

import os
from dataclasses import dataclass, field

import xarray as xr

from ..base import BaseLoader
from ..registry import register_loader


@register_loader("onset_threshold")
@dataclass
class ThresholdLoader(BaseLoader):
    """Loader for onset threshold mean wet spell (MWmean)

    The data file to be loaded contains:
      - data_var:  MWmean  (mean wet threshold)
      - dims:      (lat, lon)
    """

    # Resolution string used in the filename, e.g. "1x1", "0p25x0p25"
    resolution: str = "1x1"

    # Filename pattern matching the script:
    filename_pattern: str = "mwset{resolution}.nc4"

    # We expect only spatial coords for thresholds
    ensure_coords: list[str] = field(default_factory=lambda: ["lat", "lon"])

    # We expect only the MWmean variable
    ensure_vars: list[str] = field(default_factory=lambda: ["MWmean"])

    # We want a DataArray back (single variable)
    to_dataarray: bool = True

    def _build_path(self) -> str:
        if not self.root:
            raise ValueError(
                "ThresholdLoader expects 'root' to point to the folder "
                "containing mwset*.nc4 files."
            )
        fname = self.filename_pattern.format(resolution=self.resolution)
        path = os.path.join(self.root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Threshold file not found: {path}")
        return path

    def load(self) -> xr.Dataset | xr.DataArray:
        path = self._build_path()

        ds = xr.open_dataset(
            path,
            engine=self.engine,
            chunks=self.chunks,
            decode_times=self.decode_times,
        )

        return self._postprocess(ds)
