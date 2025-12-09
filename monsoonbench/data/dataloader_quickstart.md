# MonsoonBench DataLoader — QuickStart Guide

A fast guide to loading IMD rainfall, deterministic forecasts, probabilistic forecasts, 
and onset thresholds using the unified dataloader API.

---

## 1. Import + List Available Loaders

```python
from monsoonbench.data import load, get_registered

print(get_registered())
```
Typical output:

```css
['imd_rain', 'deterministic_forecast', 'probabilistic_forecast', 'onset_threshold']
```
You don't need to import "get_registered" to use the dataloader. I did it here
just to demonstrate the types of data loaders currently available to you.

## 2. Basic Usage
Every dataset is loaded using the same API. 

It's best practice to rename the variable and coordinates in your datasets as:
- "time" for initialization date
- "lon" for longitude
- "lat" for latitute
- "tp" for rainfall/precipitation 

before using the dataloader.

Now, let's go through how to use the **"imd_rainfall"** loader as an example.

```python
from monsoonbench.data import load

da = load(
    "imd_rain",                    
    root="/path/to/imd_rainfall_data/2p0",
    years=range(2012, 2015),                    
    subset={"time": slice("2012-01-01", "2014-12-31")
            "lat":  slice(15, 30),
            "lon":  slice(75, 95),
        },                                      
    # chunks={"time": 64},                        # optional dask chunking
    # engine="h5netcdf",                          # can be overrided
    # decode_times=True,                          # default True
    # rename={"RAINFALL": "tp"},                  # optional explicit renames
    # drop_variables=["unneeded_var"],            # optional drop
)
print(da)           # xarray.DataArray 'tp' (...)
print(da.dims)      # e.g., ('time', 'lat', 'lon')
```

### What `load(...)` accepts
- **name**: loader key. In this example, we are using `"imd_rain"`. If you are
working on a different type of dataset, adjust accordingly using `"deterministic_forecast"`, `"probabilistic_forecast"`, or `"onset_threshold"` as your loader key.

- **root**: folder containing the data files. Loader expects files with name 
    - *data_{year}.nc* for "imd_rain"
    - *{year}.nc* for "deterministic_forecast" and "probabilistic_forecast"
    - *mwset{resolution}x{resolution}.nc* (e.g., *mwset1x1.nc*) for "onset_threshold"

    inside **root**. Missing years are logged and skipped; if nothing is found, it raises an error.

- **years**: int (single), iterable of int (e.g., *range(2012, 2020)* and this covers
year 2012 to 2019), or a list of non-consecutive integers if you only want specific
years and they are not consecutive (e.g., *[2019, 2022, 2024]*). You select the
data files you're interested in in this step and subset the yearly data into monthly,
weekly or daily data by (lat, lon) based on your needs using the following **subset** option.

- **subset** (dict, optional): offers a variety of ways to subset the original yearly datasets.
There are three patterns you can use to subset by time:
    1. Single time interval: `subset={"time": slice("2019-05-01", "2019-06-01")}`.
    This will give your the data with init date from 2019-05-01 to 2019-06-01.

    2. Multiple time intervals:
        `subset={
            "time": [
                slice("2019-05-01", "2019-06-30"),
                slice("2020-05-01", "2020-06-30"),
            ]
        }`
        This will give you two periods of data from 2019-05-01 to 2019-06-30 and 
        from 2020-05-01 to 2020-06-30.
    
    3. Single dates: `subset={"time": ["2012-01-01", "2013-01-10"]}`. This will
    give you the data for two specific init dates-2012-01-01 and 2013-01-10.

    For spatial slicing: `subset{"lat": slice(15, 30), "lon": slice(75, 95)}`

- **chunks** (dict, optional): optional dask chunk sizes, e.g. `{"time": 64, "lat": 180, "lon": 180}`.

- **engine** (str, optional): `"h5netcdf"` (default pathway).

- **decode_times** (bool, default True): let xarray decode CF times.

- **rename** (dict, optional): explicit renames if file variable names differ. Common aliases are applied automatically. Your explicit map wins on conflicts.

- **drop_variables** (list[str], optional): drop unneeded variables to keep memory low.

### Returned type
By default all loaders returns a `DataArray` named **`"tp"`** (total precipitation)
with dimensions `(time, lat, lon)`. Probabilistic forecast data would have one more
dimension "member" representing ensemble members. If your source files use other names,
either rely on aliases or pass `rename={...}`.


## 3. Now you’re Ready to Load Any Dataset
With this API you can:

- Load any dataset in one line

- Combine multiple years or pick any number of single dates

- Slice time and space easily

- Load deterministic & probabilistic models identically

- Add new loaders with minimal code