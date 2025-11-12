# loaders/imd.py
from __future__ import annotations
from dataclasses import dataclass, field
import os, xarray as xr
from typing import Iterable, Sequence
from ..registry import register_loader
from ..base import BaseLoader

@register_loader("imd_rain")
@dataclass
class IMDRainLoader(BaseLoader):
    # Accept one year (int) or many (Sequence[int])
    years: Sequence[int] | int = 2021
    file_patterns: tuple[str, ...] = ("data_{year}.nc",)
    ensure_coords: list[str] = field(default_factory=lambda: ["time"])
    ensure_vars:   list[str] = field(default_factory=lambda: ["tp"])
    to_dataarray: bool = True

    def _resolve_paths(self, folder: str, years: Iterable[int]) -> list[str]:
        paths, missing = [], []
        for y in years:
            found = None
            for pat in self.file_patterns:
                p = os.path.join(folder, pat.format(year=y))
                if os.path.exists(p):
                    found = p; break
            if found: paths.append(found)
            else:     missing.append(y)
        if not paths:
            raise FileNotFoundError(f"No IMD files found in {folder} for years {list(years)}")
        if missing:
            print(f"[imd_rain] Missing years (skipped): {missing}")
        return paths

    def load(self) -> xr.DataArray:
        if not self.root:
            raise ValueError("IMDRainLoader expects 'root' to point to the IMD folder.")

        years = self.years if isinstance(self.years, Sequence) and not isinstance(self.years, (str, bytes)) else [self.years]
        paths = self._resolve_paths(self.root, years)

        ds = xr.open_mfdataset(
            paths, combine="by_coords",
            engine=self.engine, chunks=self.chunks, decode_times=self.decode_times, parallel=True
        )

        # Ensure main var is 'tp'
        if "tp" not in ds.data_vars:
            for cand in ("RAINFALL", "rain", "precip", "pr"):
                if cand in ds.data_vars:
                    ds = ds.rename({cand: "tp"}); break

        # returns DataArray('tp') since to_dataarray=True
        return self._postprocess(ds)
