# loaders/deterministic.py
"""Deterministic forecast data loader for the monsoon benchmark."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import xarray as xr

from ..base import BaseLoader
from ..registry import register_loader


@register_loader("deterministic_forecast")
@dataclass
class DeterministicForecastLoader(BaseLoader):
    """Loader for deterministic forecast rainfall (tp).

    Assumptions:
    - One file per year, with filename: "{year}.nc". The root directory points to the folder
        these "{year}.nc" files exist

    - Each file is an xarray.Dataset like:

        <xarray.Dataset>
        Dimensions:  (day: 46, time: 35, lat: 16, lon: 17)
        Coordinates:
          * day      (day) int64
          * time     (time) datetime64[ns]
          * lat      (lat) float64
          * lon      (lon) float64
        Data variables:
              tp     (day, time, lat, lon) float32
    """

    # First select data files by year
    years: Sequence[int] | int = ()

    # Fixed filename pattern: {year}.nc in tuple
    file_patterns: tuple[str, ...] = ("{year}.nc",)

    # Use h5netcdf by default
    engine: str | None = "h5netcdf"

    # Coordinate names are expect to be day/time + lat/lon
    ensure_coords: list[str] = field(default_factory=lambda: ["day", "time"])

    # Require tp as the main data variable
    ensure_vars: list[str] = field(default_factory=lambda: ["tp"])

    to_dataarray: bool = True

    # ---------- helpers ----------

    def _seq(self, x) -> list[int]:
        """Normalize an int or sequence of int to a list[int]."""
        if isinstance(x, Sequence) and not isinstance(x, str | bytes):
            return list(x)
        return [int(x)]

    def _resolve_paths(self, folder: str, years: int | Iterable[int]) -> list[str]:
        """Find all existing {year}.nc files for this deterministic model.

        We keep a friendly message listing missing years but don't hard-fail
        as long as at least one file is found.
        """
        # Normalize years to a concrete list so we can safely iterate & reuse
        if isinstance(years, int):
            year_list = [years]
        else:
            year_list = list(years)

        paths: list[str] = []
        missing: list[int] = []

        for y in year_list:
            found_for_year = False
            for pat in self.file_patterns:
                p = os.path.join(folder, pat.format(year=y))
                if os.path.exists(p):
                    paths.append(p)
                    found_for_year = True
                    break
            if not found_for_year:
                missing.append(y)

        if not paths:
            raise FileNotFoundError(
                f"No deterministic forecast files found under {folder} "
                f"for years {year_list}"
            )

        if missing:
            print(f"[deterministic_forecast] Missing years (skipped): {missing}")

        # Defense line. Remove duplicate input
        seen: set[str] = set()
        unique_paths: list[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        return unique_paths

    # ---------- main entry point ----------

    def load(self) -> xr.Dataset | xr.DataArray:
        """Load deterministic forecast data and apply common post-processing."""
        if not self.root:
            raise ValueError("DeterministicForecastLoader expects 'root' directory.")

        years = self._seq(self.years)
        if not years:
            raise ValueError("'years' must be provided (int or sequence of int).")

        paths = self._resolve_paths(self.root, years)

        ds = xr.open_mfdataset(
            paths,
            combine="by_coords",
            engine=self.engine,
            chunks=self.chunks,
            decode_times=self.decode_times,
            parallel=True,
        )

        return self._postprocess(ds)
