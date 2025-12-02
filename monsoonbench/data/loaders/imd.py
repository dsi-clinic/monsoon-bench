# loaders/imd.py
from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import xarray as xr

from ..base import BaseLoader
from ..registry import register_loader


@register_loader("imd_rain")
@dataclass
class IMDRainLoader(BaseLoader):
    # Accept one year (int) or many (Sequence[int])
    years: Sequence[int] | int = 2021
    file_patterns: tuple[str, ...] = ("data_{year}.nc",)
    ensure_coords: list[str] = field(default_factory=lambda: ["time"])
    ensure_vars: list[str] = field(default_factory=lambda: ["tp"])
    to_dataarray: bool = True

    def _resolve_paths(self, folder: str, years: int | Iterable[int]) -> list[str]:
        # Normalize years
        if isinstance(years, int):
            year_list = [years]
        else:
            # works for range, list, tuple, generator, etc.
            year_list = list(years)

        paths: list[str] = []
        missing: list[int] = []

        for y in year_list:
            found = None
            for pat in self.file_patterns:
                p = os.path.join(folder, pat.format(year=y))
                if os.path.exists(p):
                    found = p
                    break
            if found:
                paths.append(found)
            else:
                missing.append(y)

        if not paths:
            raise FileNotFoundError(
                f"No IMD files found in {folder} for years {year_list}"
            )
        if missing:
            print(f"[imd_rain] Missing years (skipped): {missing}")

        return paths

    def load(self) -> xr.DataArray:
        if not self.root:
            raise ValueError("IMDRainLoader expects 'root' to point to the IMD folder.")

        years = (
            self.years
            if isinstance(self.years, Sequence)
            and not isinstance(self.years, (str, bytes))
            else [self.years]
        )
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
