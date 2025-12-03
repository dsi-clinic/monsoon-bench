"""Base loader class for dataset loading.

This module provides the BaseLoader class that all dataset loaders inherit from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    from typing import Self

DEFAULT_VAR_ALIASES = {
    "RAINFALL": "tp",
    "rainfall": "tp",
    "Rainfall": "tp",
    "precip": "tp",
    "precipitation": "tp",
    "Precipitation": "tp",
    "PRECIPITATION": "tp",
    "tp": "tp",
    "TP": "tp",
    "total_precipitation": "tp",
    "rain": "tp",
    "Rain": "tp",
    "RAIN": "tp",
    "pr": "tp",
    "PR": "tp",
    "latitude": "lat",
    "Latitude": "lat",
    "LATITUDE": "lat",
    "LAT": "lat",
    "longitude": "lon",
    "Longitude": "lon",
    "LONGITUDE": "lon",
    "LON": "lon",
    "time": "time",
    "Time": "time",
    "TIME": "time",
    "date": "time",
    "Date": "time",
    "DATE": "time",
}

REQUIRED_COORDS_COMMON = ["lat", "lon"]


@dataclass
class BaseLoader:
    """Base class for all dataset loaders.

    Subclasses should implement .load() to return an xr.Dataset or xr.DataArray,
    then call self._postprocess(...) before returning.

    Common knobs:
      - root:       dataset root directory
      - chunks:     Dask chunk dict, e.g. {'time': 64, 'lat': 180, 'lon': 180}
      - engine:     xarray engine ('netcdf4', 'h5netcdf', 'cfgrib', 'zarr', ...)
      - decode_times: let xarray decode CF times
      - rename:     explicit rename map (overrides DEFAULT_VAR_ALIASES if key collides)
      - drop_variables: variables to drop from Dataset
      - subset:     simple .sel(...) filter, e.g. {'lat': slice(5, 40), 'lon': slice(60, 100)}
      - ensure_vars: list of data_vars to keep (error if missing), optional
      - ensure_coords: list of coords to require, optional
      - to_dataarray: if True and Dataset has a single var, return DataArray
    """

    # Options:
    root: str | None = None
    chunks: dict[str, int] | None = None
    engine: str | None = None
    decode_times: bool = True

    rename: dict[str, str] = field(default_factory=dict)
    drop_variables: list[str] | None = None
    subset: dict[str, Any] = field(default_factory=dict)

    ensure_vars: list[str] | None = None
    ensure_coords: list[str] | None = None
    to_dataarray: bool = False

    @classmethod
    def from_kwargs(cls: type[Self], **kwargs) -> Self:  # type: ignore[no-untyped-def]
        """Create a loader instance from keyword arguments."""
        return cls(**kwargs)

    # ------ Helpers ------
    def _apply_alias_renames(
        self: Self, ds: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        # Apply default aliases if present
        for src, tgt in DEFAULT_VAR_ALIASES.items():
            if (
                src in ds.dims
                or src in ds.coords
                or (hasattr(ds, "data_vars") and src in ds.data_vars)
            ):
                try:
                    ds = ds.rename({src: tgt})
                except (ValueError, KeyError):
                    # If rename fails, ignore and let explicit map handle it
                    continue
        # Apply explicit user-provided rename (takes precedence)
        if self.rename:
            ds = ds.rename(self.rename)
        return ds

    def _subset(self: Self, ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        if not self.subset:
            return ds
        # Only select coords that exist
        sel_kwargs = {k: v for k, v in self.subset.items() if k in ds.coords}
        return ds.sel(**sel_kwargs) if sel_kwargs else ds

    def _drop_vars(
        self: Self, ds: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        if not self.drop_variables or not isinstance(ds, xr.Dataset):
            return ds
        keep = [v for v in ds.data_vars if v not in self.drop_variables]
        return ds[keep]

    def _ensure(self: Self, ds: xr.Dataset | xr.DataArray) -> None:
        # coords
        coords_req = set(REQUIRED_COORDS_COMMON)
        if self.ensure_coords:
            coords_req |= set(self.ensure_coords)
        missing_coords = [c for c in coords_req if c not in ds.coords]
        if missing_coords:
            raise ValueError(f"Missing required coords: {missing_coords}")

        # variables
        if self.ensure_vars and isinstance(ds, xr.Dataset):
            missing_vars = [v for v in self.ensure_vars if v not in ds.data_vars]
            if missing_vars:
                raise ValueError(f"Missing required variables: {missing_vars}")

    def _finalize(
        self: Self, ds: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        # Optionally coerce Dataset to DataArray if there is a single variable
        if self.to_dataarray and isinstance(ds, xr.Dataset):
            if len(ds.data_vars) == 1:
                ds = next(iter(ds.data_vars.values()))
            else:
                raise ValueError(
                    "to_dataarray=True but dataset has multiple variables."
                )
        return ds

    def _postprocess(
        self: Self, ds: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        """Run after the subclass has opened/assembled the dataset.

        Order matters: rename -> subset -> drop -> ensure -> finalize
        """
        ds = self._apply_alias_renames(ds)
        ds = self._subset(ds)
        ds = self._drop_vars(ds)
        self._ensure(ds)
        ds = self._finalize(ds)
        return ds

    # ------ What subclasses must implement ------
    def load(self: Self) -> xr.Dataset | xr.DataArray:
        """Return xr.Dataset or xr.DataArray. Must call self._postprocess(...) before returning."""
        raise NotImplementedError("Implement in subclasses")
