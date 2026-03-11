"""Subgrid onset-variability utilities for multi-resolution onset analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from monsoonbench.metrics.base import OnsetMetricsBase
from monsoonbench.utils.onset_timeseries import (
    load_threshold_with_fix,
    standardize_rainfall_dims,
)

AUTHOR_025_XFIRST = 66.625
AUTHOR_025_YFIRST = 6.625
AUTHOR_025_INC = 0.25
AUTHOR_025_XSIZE = 135
AUTHOR_025_YSIZE = 129

__all__ = [
    "resolve_threshold_file",
    "summarize_sample_rainfall_grid",
    "compute_mean_subgrid_variability_maps",
]


def resolve_threshold_file(candidates: list[Path]) -> Path:
    """Return the first existing threshold file from a priority list.

    Args:
        candidates: Candidate threshold-file paths.

    Returns:
        First path that exists on disk.

    Raises:
        FileNotFoundError: If none of the candidate files exist.
    """
    found = next((path for path in candidates if path.exists()), None)
    if found is None:
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(
            "Native 0.25-degree threshold file not found. " f"Expected one of: {names}"
        )
    return found


def summarize_sample_rainfall_grid(folder: Path) -> dict[str, Any] | None:
    """Summarize one sample rainfall file to verify selected grid geometry.

    Args:
        folder: Directory containing yearly IMD rainfall NetCDF files.

    Returns:
        Summary dictionary for one sample file, or ``None`` if no files exist.
    """
    sample_files = sorted(folder.glob("*.nc"))
    if not sample_files:
        return None

    sample_file = sample_files[0]
    sample_ds = xr.open_dataset(sample_file)
    var_name = list(sample_ds.data_vars)[0]
    sample_da = standardize_rainfall_dims(sample_ds[var_name])

    return {
        "selected_folder": str(folder),
        "file_count": len(sample_files),
        "sample_file": sample_file.name,
        "lat_first": float(sample_da.lat.values[0]),
        "lat_last": float(sample_da.lat.values[-1]),
        "lat_step": float(np.median(np.diff(sample_da.lat.values))),
        "lon_first": float(sample_da.lon.values[0]),
        "lon_last": float(sample_da.lon.values[-1]),
        "lon_step": float(np.median(np.diff(sample_da.lon.values))),
    }


def _load_imd(year: int, folder: Path) -> xr.DataArray:
    rain = OnsetMetricsBase.load_imd_rainfall(year, str(folder))
    return standardize_rainfall_dims(rain)


def _author_025_coords() -> dict[str, xr.DataArray]:
    lon = AUTHOR_025_XFIRST + AUTHOR_025_INC * np.arange(AUTHOR_025_XSIZE)
    lat = AUTHOR_025_YFIRST + AUTHOR_025_INC * np.arange(AUTHOR_025_YSIZE)
    return {
        "lat": xr.DataArray(lat, dims=["lat"]),
        "lon": xr.DataArray(lon, dims=["lon"]),
    }


def _is_same_grid_2d(
    da: xr.DataArray,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
    tol: float = 1e-9,
) -> bool:
    if "lat" not in da.coords or "lon" not in da.coords:
        return False
    if da.sizes.get("lat") != target_lat.size or da.sizes.get("lon") != target_lon.size:
        return False
    return np.allclose(da["lat"].values, target_lat.values, atol=tol) and np.allclose(
        da["lon"].values, target_lon.values, atol=tol
    )


def _interp_to_grid_2d(
    da: xr.DataArray,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
    label: str = "",
) -> xr.DataArray:
    """Interpolate to target grid with linear first, then nearest edge fill."""
    out_lin = da.interp(lat=target_lat, lon=target_lon, method="linear")
    if np.isnan(out_lin.values).any():
        out_nn = da.interp(lat=target_lat, lon=target_lon, method="nearest")
        out_lin = out_lin.where(np.isfinite(out_lin), out_nn)
    if label:
        print(
            f"  Regridded {label} -> author 0.25 grid "
            f"({float(target_lon.values[0]):.3f}, {float(target_lat.values[0]):.3f})"
        )
    return out_lin


def _force_author_025_grid(
    rain_025: xr.DataArray, thres_025: xr.DataArray, year: int | None = None
) -> tuple[xr.DataArray, xr.DataArray]:
    """Force rainfall/threshold to the same author 0.25-degree grid."""
    target = _author_025_coords()
    target_lat = target["lat"]
    target_lon = target["lon"]

    rain_out = rain_025
    if not _is_same_grid_2d(rain_out, target_lat, target_lon):
        rain_out = _interp_to_grid_2d(
            rain_out,
            target_lat,
            target_lon,
            label=f"0.25 rainfall {year}" if year else "0.25 rainfall",
        )

    thres_out = thres_025
    if not _is_same_grid_2d(thres_out, target_lat, target_lon):
        thres_out = _interp_to_grid_2d(
            thres_out, target_lat, target_lon, label="0.25 threshold"
        )

    # Ensure strict coordinate identity before onset detection.
    rain_out = rain_out.assign_coords(lat=thres_out["lat"], lon=thres_out["lon"])
    return rain_out, thres_out


def _abs_day_diff(
    sub_onset: xr.DataArray, coarse_onset_on_sub: xr.DataArray
) -> xr.DataArray:
    """Compute absolute day difference between two datetime onset maps."""
    sub_vals = sub_onset.values
    coarse_vals = coarse_onset_on_sub.values
    valid = (~np.isnat(sub_vals)) & (~np.isnat(coarse_vals))

    out = np.full(sub_vals.shape, np.nan, dtype=float)
    out[valid] = np.abs((sub_vals[valid] - coarse_vals[valid]) / np.timedelta64(1, "D"))

    return xr.DataArray(
        out, coords=sub_onset.coords, dims=sub_onset.dims, name="abs_day_diff"
    )


def compute_mean_subgrid_variability_maps(
    years: list[int],
    imd_1deg_dir: Path,
    imd_025deg_dir: Path,
    imd_4deg_dir: Path,
    thres_1deg_file: Path,
    thres_4deg_file: Path,
    thres_025deg_file: Path,
    cache_nc: Path | None = None,
    mok: bool = True,
    use_author_025_grid: bool = True,
    force_recompute: bool = False,
) -> xr.Dataset:
    """Compute Figure-14 subgrid variability maps averaged across years.

    Args:
        years: Years included in the multi-year average.
        imd_1deg_dir: Directory with 1-degree IMD rainfall.
        imd_025deg_dir: Directory with 0.25-degree IMD rainfall.
        imd_4deg_dir: Directory with 4-degree IMD rainfall.
        thres_1deg_file: 1-degree threshold file.
        thres_4deg_file: 4-degree threshold file.
        thres_025deg_file: 0.25-degree threshold file.
        cache_nc: Optional NetCDF cache path.
        mok: Whether to enforce MOK onset search start date.
        use_author_025_grid: Whether to force 0.25 data onto author grid.
        force_recompute: If ``True``, ignore cache and recompute from raw data.

    Returns:
        Dataset with variables:
        - ``panel_a_1deg`` on dims ``(lat1, lon1)``
        - ``panel_b_025deg`` on dims ``(lat025, lon025)``
    """
    if not years:
        raise ValueError("`years` must contain at least one year.")

    if cache_nc is not None and cache_nc.exists() and not force_recompute:
        print("Loading cached results from", cache_nc)
        cached = xr.open_dataset(cache_nc)
        cache_ok = (
            "panel_a_1deg" in cached.data_vars
            and "panel_b_025deg" in cached.data_vars
            and "lat1" in cached["panel_a_1deg"].dims
            and "lon1" in cached["panel_a_1deg"].dims
        )
        if cache_ok:
            return cached
        print("Cache format is old/incompatible. Recomputing...")

    thres_1deg = load_threshold_with_fix(thres_1deg_file)
    thres_4deg = load_threshold_with_fix(thres_4deg_file)
    thres_025deg_native = load_threshold_with_fix(thres_025deg_file)

    yearly_a_1deg: list[xr.DataArray] = []
    yearly_b_025deg: list[xr.DataArray] = []

    for idx, year in enumerate(years, start=1):
        print(f"[{idx}/{len(years)}] Processing year {year} ...")

        rain_1deg = _load_imd(year, imd_1deg_dir)
        rain_025deg = _load_imd(year, imd_025deg_dir)
        rain_4deg = _load_imd(year, imd_4deg_dir)

        onset_1deg = OnsetMetricsBase.detect_observed_onset(
            rain_1deg, thres_1deg, year, mok=mok
        )
        onset_4deg = OnsetMetricsBase.detect_observed_onset(
            rain_4deg, thres_4deg, year, mok=mok
        )

        onset_4_on_1 = onset_4deg.sel(
            lat=onset_1deg.lat, lon=onset_1deg.lon, method="nearest"
        )
        var_a_1deg = _abs_day_diff(onset_1deg, onset_4_on_1)

        if use_author_025_grid:
            rain_025deg_this_year, thres_025deg_this_year = _force_author_025_grid(
                rain_025deg, thres_025deg_native, year=year
            )
        else:
            rain_025deg_this_year = rain_025deg
            thres_025deg_this_year = thres_025deg_native

        onset_025deg = OnsetMetricsBase.detect_observed_onset(
            rain_025deg_this_year, thres_025deg_this_year, year, mok=mok
        )
        onset_4_on_025 = onset_4deg.sel(
            lat=onset_025deg.lat,
            lon=onset_025deg.lon,
            method="nearest",
        )
        var_b_025deg = _abs_day_diff(onset_025deg, onset_4_on_025)

        yearly_a_1deg.append(var_a_1deg.expand_dims(year=[year]))
        yearly_b_025deg.append(var_b_025deg.expand_dims(year=[year]))

    panel_a = (
        xr.concat(yearly_a_1deg, dim="year")
        .mean("year", skipna=True)
        .rename("panel_a_1deg")
    )
    panel_b = (
        xr.concat(yearly_b_025deg, dim="year")
        .mean("year", skipna=True)
        .rename("panel_b_025deg")
    )

    panel_a = panel_a.rename({"lat": "lat1", "lon": "lon1"})
    panel_b = panel_b.rename({"lat": "lat025", "lon": "lon025"})

    result = xr.Dataset(
        {
            "panel_a_1deg": panel_a,
            "panel_b_025deg": panel_b,
        }
    )
    result.attrs["years"] = f"{years[0]}-{years[-1]}"
    result.attrs["mok"] = str(mok)
    result.attrs["use_author_025_grid"] = str(use_author_025_grid)

    if cache_nc is not None:
        cache_nc.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(cache_nc)
        print("Saved cache to", cache_nc)

    return result
