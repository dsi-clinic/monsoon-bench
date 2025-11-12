# Dataloader User Guide

---

## A. Quick Start

### 1) Clone and create a virtual environment
```bash
git clone <YOUR_FORK_OR_REPO_URL> monsoon-bench
cd monsoon-bench

python3 -m venv .venv
source .venv/bin/activate  
```

### 2) Install the package in editable mode
```bash
python -m pip install -e .
```

### 4) Smoke test
Start Python or a notebook in this venv:
```python
from monsoonbench.data import get_registered, load
print("registered loaders:", sorted(get_registered().keys()))
```
You should see `['imd_rain']` at minimum.

---

## B. About NetCDF4 backend (can safely ignore if you don't need it)

NetCDF4 wheels conflicts with `h5netcdf` on my local machine. If you **must** use NetCDF4, install the **extra**:

```bash
python -m pip install -e ".[netcdf4]"
```
If you later hit crashes, try removing NetCDF4 and use `h5netcdf`:
```bash
python -m pip uninstall -y netCDF4 cftime
```

---

## C. Using the loader

### Core call
```python
from monsoonbench.data import load

da = load(
    "imd_rain",
    root="/path/to/imd_rainfall_data/2p0",
    years=range(2012, 2015),                    # int or iterable of years
    subset={"time": slice("2012-01-01", "2014-12-31")},  # optional .sel on coords (time, lat, lon)
    chunks={"time": 64},                        # optional dask chunking
    # engine="h5netcdf",                        # can be overrided
    # decode_times=True,                        # default True
    # rename={"RAINFALL": "tp"},                # optional explicit renames
    # drop_variables=["unneeded_var"],          # optional drop
)
print(da)           # xarray.DataArray 'tp' (...)
print(da.dims)      # e.g., ('time', 'lat', 'lon')
print(da.shape)     # e.g., (Ntime, Nlat, Nlon)
```

### What `load(...)` accepts
- **name**: loader key (currently `"imd_rain"`, will be expanded to include other forecast data in the next step).
- **root**: folder containing the IMD files. Loader expects `data_{year}.nc` files inside `root`.
- **years**: `int` (single) or iterable of `int` (e.g., `range(2012, 2020)`).
- **subset** (`dict`): coordinates to pass to `.sel(...)`, e.g. `{"time": slice("2013", "2014-12-31")}` or `{"lat": slice(5, 35), "lon": slice(60, 100)}`.
- **chunks** (`dict`): optional dask chunk sizes, e.g. `{"time": 64, "lat": 180, "lon": 180}`.
- **engine** (`str`, optional): `"h5netcdf"` (default pathway).
- **decode_times** (`bool`, default `True`): let xarray decode CF times.
- **rename** (`dict`, optional): explicit renames if file variable names differ. Common aliases are applied automatically. Your explicit map wins on conflicts.
- **drop_variables** (`list[str]`, optional): drop unneeded variables to keep memory low.

### File naming
The IMD loader looks for `data_{year}.nc` inside `root` (e.g., `root/data_2012.nc`, `root/data_2013.nc`, …). Missing years are logged and skipped; if nothing is found, it raises an error.

### Returned type
By default the IMD loader returns a `DataArray` named **`"tp"`** (total precipitation) with dimensions `(time, lat, lon)`. If your source files use other names, either rely on aliases or pass `rename={...}`.
