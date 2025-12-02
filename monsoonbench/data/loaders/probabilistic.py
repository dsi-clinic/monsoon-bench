# monsoonbench/data/loaders/probabilistic.py
from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import xarray as xr

from ..base import BaseLoader
from ..registry import register_loader


@register_loader("probabilistic_forecast")
@dataclass
class ProbabilisticForecastLoader(BaseLoader):
    """Loader for probabilistic (ensemble) forecast rainfall (tp).

    Assumptions:
    - One file per year, with filename: "{year}.nc". The root directory points to the folder
        these "{year}.nc" files exist

    - Each file is an xarray.Dataset like:

        <xarray.Dataset>
        Dimensions:  (day: 46, time: 35, lat: 16, lon: 17)
        Coordinates:
          * member   (member) int64.        (Ensemble members along a 'member' dimension.)
          * day      (day) int64
          * time     (time) datetime64[ns]
          * lat      (lat) float64
          * lon      (lon) float64
        Data variables:
              tp     (member, day, time, lat, lon) float32
    """

    # Probabilitic models: e.g. "aifs", "fuxi", "graphcast"
    model: str = "aifs"

    # Accept single year or a sequence of years
    years: Sequence[int] | int = 1964

    # Fixed filename pattern: {year}.nc in tuple
    file_patterns: tuple[str, ...] = ("{year}.nc",)

    # Coordinate names are expect to be day/time + lat/lon + member
    ensure_coords: list[str] = field(default_factory=lambda: ["time", "member"])

    # Require tp as the main data variable
    ensure_vars: list[str] = field(default_factory=lambda: ["tp"])

    to_dataarray: bool = True

    def _resolve_paths(
        self, folder: str, years: Union[int, Iterable[int]]
    ) -> list[str]:
        """Find all existing {year}.nc files for this probabilistic model.

        We keep a friendly message listing missing years but don't hard-fail
        as long as at least one file is found.
        """
        # Normalize years to a concrete list so we can safely iterate and reuse
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
                f"No probabilistic forecast files found under {folder} "
                f"for years {year_list}"
            )

        if missing:
            print(f"[probabilistic_forecast] Missing years (skipped): {missing}")

        # Defense line. Remove duplicate input.
        seen: set[str] = set()
        unique_paths: list[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        return unique_paths

    def load(self) -> xr.DataArray:
        """Open and combine all requested years into one Dataset/DataArray,
        then run the common BaseLoader post-processing.
        """
        if not self.root:
            raise ValueError(
                "ProbabilisticForecastLoader expects 'root' to point to the "
                "forecast data directory."
            )

        # Normalize years to a list
        if isinstance(self.years, Sequence) and not isinstance(
            self.years, (str, bytes)
        ):
            years = list(self.years)
        else:
            years = [int(self.years)]

        paths = self._resolve_paths(self.root, years)

        ds = xr.open_mfdataset(
            paths,
            combine="by_coords",
            engine=self.engine,
            chunks=self.chunks,
            decode_times=self.decode_times,
            parallel=True,
        )

        # Let BaseLoader handle subset / coord checks / DataArray conversion, etc.
        return self._postprocess(ds)
