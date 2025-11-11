"""Data loading and validation for MonsoonBench.

This module provides the DataLoader class for loading IMD observations,
model forecasts, and onset threshold data.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

# Constants for day-of-week filtering
MONDAY_WEEKDAY = 0
THURSDAY_WEEKDAY = 3


class DataLoader:
    """Handle loading of IMD observations, forecasts, and threshold data.

    This class provides a unified interface for loading all data needed for
    monsoon onset metrics computation.

    Args:
        imd_folder: Directory containing IMD rainfall NetCDF files
        thres_file: Path to onset threshold NetCDF file
        shpfile_path: Path to India shapefile for visualization
        model_forecast_dir: Optional directory containing model forecast data

    Example:
        >>> loader = DataLoader.from_config("config.yaml")
        >>> obs = loader.load_observations(years=[2020, 2021])
        >>> threshold = loader.load_threshold()
        >>> forecasts = loader.load_forecasts(year=2020, model_type="deterministic")
    """

    def __init__(
        self,
        imd_folder: str | Path,
        thres_file: str | Path,
        shpfile_path: str | Path,
        model_forecast_dir: str | Path | None = None,
    ):
        """Initialize DataLoader with data paths.

        Args:
            imd_folder: Directory containing IMD rainfall NetCDF files
            thres_file: Path to onset threshold NetCDF file
            shpfile_path: Path to India shapefile
            model_forecast_dir: Directory containing model forecast data (optional)
        """
        self.imd_folder = Path(imd_folder)
        self.thres_file = Path(thres_file)
        self.shpfile_path = Path(shpfile_path)
        self.model_forecast_dir = (
            Path(model_forecast_dir) if model_forecast_dir else None
        )

        # Validate paths exist
        if not self.imd_folder.exists():
            raise FileNotFoundError(f"IMD folder not found: {self.imd_folder}")
        if not self.thres_file.exists():
            raise FileNotFoundError(f"Threshold file not found: {self.thres_file}")
        if not self.shpfile_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {self.shpfile_path}")

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DataLoader":
        """Create DataLoader from YAML configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Initialized DataLoader instance

        Example:
            >>> loader = DataLoader.from_config("config.yaml")
        """
        config_path = Path(config_path)
        with config_path.open() as f:
            config = yaml.safe_load(f)

        return cls(
            imd_folder=config["imd_folder"],
            thres_file=config["thres_file"],
            shpfile_path=config["shpfile_path"],
            model_forecast_dir=config.get("model_forecast_dir"),
        )

    def load_observations(self, year: int) -> xr.DataArray:
        """Load IMD daily rainfall observations for a given year.

        Args:
            year: Year to load data for

        Returns:
            Rainfall DataArray with dimensions (time, lat, lon)
            Variable is 'RAINFALL' in mm/day

        Raises:
            FileNotFoundError: If no IMD file found for the year

        Example:
            >>> rainfall = loader.load_observations(year=2020)
            >>> print(rainfall.dims)
            ('time', 'lat', 'lon')
        """
        # Try common file naming patterns
        file_patterns = [f"data_{year}.nc", f"{year}.nc"]

        imd_file = None
        for pattern in file_patterns:
            test_path = self.imd_folder / pattern
            if test_path.exists():
                imd_file = test_path
                break

        if imd_file is None:
            available_files = [
                f.name for f in self.imd_folder.glob("*.nc") if f.is_file()
            ]
            raise FileNotFoundError(
                f"No IMD file found for year {year} in {self.imd_folder}. "
                f"Tried patterns: {file_patterns}. "
                f"Available files: {available_files}"
            )

        print(f"Loading IMD rainfall from: {imd_file}")

        ds = xr.open_dataset(imd_file)
        rainfall = ds["RAINFALL"]

        # Standardize dimension names
        dim_mapping = {}
        if "latitude" in rainfall.dims:
            dim_mapping["latitude"] = "lat"
        if "LATITUDE" in rainfall.dims:
            dim_mapping["LATITUDE"] = "lat"
        if "longitude" in rainfall.dims:
            dim_mapping["longitude"] = "lon"
        if "LONGITUDE" in rainfall.dims:
            dim_mapping["LONGITUDE"] = "lon"
        if "TIME" in rainfall.dims:
            dim_mapping["TIME"] = "time"

        if dim_mapping:
            rainfall = rainfall.rename(dim_mapping)
            print(f"Renamed dimensions: {dim_mapping}")

        return rainfall

    def load_threshold(self) -> xr.DataArray:
        """Load onset threshold data.

        Returns:
            Threshold DataArray for onset detection

        Example:
            >>> threshold = loader.load_threshold()
        """
        print(f"Loading threshold data from: {self.thres_file}")
        ds = xr.open_dataset(self.thres_file)

        # Find the threshold variable (common names)
        threshold_vars = ["threshold", "thresh", "THRESHOLD", "THRESH"]
        threshold = None

        for var in threshold_vars:
            if var in ds:
                threshold = ds[var]
                break

        if threshold is None:
            # If not found, use the first data variable
            data_vars = list(ds.data_vars)
            if data_vars:
                threshold = ds[data_vars[0]]
                print(f"Using variable '{data_vars[0]}' as threshold")
            else:
                raise ValueError(f"No threshold variable found in {self.thres_file}")

        return threshold

    def load_forecasts(
        self,
        year: int,
        model_type: str,
        forecast_dir: str | Path | None = None,
        twice_weekly: bool = True,
    ) -> xr.DataArray:
        """Load model forecast data for a given year.

        Args:
            year: Year to load forecasts for
            model_type: Type of forecast ("deterministic" or "probabilistic")
            forecast_dir: Optional override for forecast directory
            twice_weekly: If True, filter for Mon/Thu initializations (default: True)

        Returns:
            Forecast DataArray with dimensions:
            - deterministic: (init_time, lat, lon, step)
            - probabilistic: (init_time, lat, lon, step, member)

        Raises:
            FileNotFoundError: If forecast file not found
            ValueError: If model_type is invalid or no matching init times

        Example:
            >>> forecasts = loader.load_forecasts(year=2020, model_type="deterministic")
        """
        if model_type not in ["deterministic", "probabilistic"]:
            raise ValueError(
                f"model_type must be 'deterministic' or 'probabilistic', got: {model_type}"
            )

        # Use provided forecast_dir or fall back to instance attribute
        model_dir = Path(forecast_dir) if forecast_dir else self.model_forecast_dir
        if model_dir is None:
            raise ValueError(
                "model_forecast_dir must be provided either in __init__ or as argument"
            )

        fname = f"{year}.nc"
        file_path = model_dir / fname

        if not file_path.exists():
            raise FileNotFoundError(f"Forecast file not found: {file_path}")

        print(f"Loading {model_type} forecasts from: {file_path}")

        # Load data
        ds = xr.open_dataset(file_path)

        # Standardize dimension names
        if "time" in ds.dims:
            ds = ds.rename({"time": "init_time"})
        if "number" in ds.dims:  # Common for probabilistic forecasts
            ds = ds.rename({"number": "member"})
        if "day" in ds.dims:
            ds = ds.rename({"day": "step"})

        # Filter for twice-weekly initializations if requested
        if twice_weekly:
            ds = self._filter_twice_weekly(ds, year)

        # Remove step=0 if present (we want forecasts starting from day 1)
        if "step" in ds.dims:
            if ds["step"][0].to_numpy() == 0:
                ds = ds.sel(step=slice(1, None))

        # Get precipitation variable
        precip_vars = ["tp", "precip", "precipitation", "RAINFALL"]
        p_model = None
        for var in precip_vars:
            if var in ds:
                p_model = ds[var]
                break

        if p_model is None:
            raise ValueError(
                f"No precipitation variable found in {file_path}. "
                f"Expected one of: {precip_vars}"
            )

        ds.close()
        return p_model

    def _filter_twice_weekly(self, ds: xr.Dataset, year: int) -> xr.Dataset:
        """Filter dataset for Monday and Thursday initializations.

        Args:
            ds: Dataset with init_time dimension
            year: Year to filter for

        Returns:
            Filtered dataset with only Mon/Thu initializations
        """
        # Use 2024 as reference for day-of-week pattern
        start_date = datetime(2024, 5, 1)
        end_date = datetime(2024, 7, 31)
        date_range = pd.date_range(start_date, end_date, freq="D")

        # Find Mondays and Thursdays
        is_monday = date_range.weekday == MONDAY_WEEKDAY
        is_thursday = date_range.weekday == THURSDAY_WEEKDAY
        filtered_dates = date_range[is_monday | is_thursday]

        # Convert to target year
        filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f"{year}-%m-%d"))

        # Find matching times in dataset
        available_init_times = pd.to_datetime(ds.init_time.to_numpy())
        matching_times = available_init_times[
            available_init_times.isin(filtered_dates_yr)
        ]

        if len(matching_times) == 0:
            raise ValueError(
                f"No matching Mon/Thu initialization times found for year {year}"
            )

        print(f"Found {len(matching_times)} Mon/Thu initializations for {year}")

        # Select only matching times
        ds = ds.sel(init_time=matching_times)
        return ds
