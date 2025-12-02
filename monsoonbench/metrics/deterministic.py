import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from .base import OnsetMetricsBase


class DeterministicOnsetMetrics(OnsetMetricsBase):
    """Deterministic model specific onset metrics calculations."""

    @staticmethod
    def get_forecast_deterministic_twice_weekly(yr, model_forecast_dir):
        """Loads model precip data for twice-weekly initializations from May to July.
        Filters for Mondays and Thursdays in the specified year.
        The forecast file is expected to be named as '{year}.nc' in the model_forecast_dir with
        variable "tp" being daily accumulated rainfall with dimensions (init_time, lat, lon, step).

        Parameters:
        yr: int, year to load data for

        Returns:
        p_model: ndarray, precipitation data
        """
        fname = f"{yr}.nc"
        file_path = os.path.join(model_forecast_dir, fname)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Filter for twice weekly data from daily for the specified year based on 2024 Monday and Thursday dates (to match with IFS CY48R1 reforecasts)
        # Define date range from May 1 to July 31 of 2024
        start_date = datetime(2024, 5, 1)
        end_date = datetime(2024, 7, 31)
        date_range = pd.date_range(start_date, end_date, freq="D")

        # Find Mondays (weekday=0) and Thursdays (weekday=3) in pandas
        is_monday = date_range.weekday == 0
        is_thursday = date_range.weekday == 3
        filtered_dates = date_range[is_monday | is_thursday]

        filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f"{yr}-%m-%d"))
        # Load data using xarray
        ds = xr.open_dataset(file_path)
        if "time" in ds.dims:
            ds = ds.rename({"time": "init_time"})
        ds = ds.sel(init_time=filtered_dates_yr)
        # Find common dates between desired dates and available dates
        available_init_times = pd.to_datetime(ds.init_time.values)
        matching_times = available_init_times[
            available_init_times.isin(filtered_dates_yr)
        ]
        if len(matching_times) == 0:
            raise ValueError(f"No matching initialization times found for year {yr}")
        ds = ds.sel(init_time=matching_times)
        # Check if 'day' dimension exists and conditionally slice
        if "day" in ds.dims:
            # Check if the first value of 'day' is 0, then slice to exclude it
            if ds["day"][0].values == 0:
                ds = ds.sel(day=slice(1, None))
        # Check if 'step' dimension exists and conditionally slice
        if "step" in ds.dims:
            # Check if the first value of 'step' is 0, then slice to exclude it
            if ds["step"][0].values == 0:
                ds = ds.sel(step=slice(1, None))

        p_model = ds["tp"]
        # Only rename 'day' to 'step' if 'day' dimension exists
        if "day" in p_model.dims:
            p_model = p_model.rename({"day": "step"})
        # Close the dataset
        ds.close()
        return p_model

    @staticmethod
    def compute_onset_for_deterministic_model(
        p_model,
        thresh_slice,
        onset_da,
        max_forecast_day=15,
        mok=True,
        onset_window=5,
        mok_month=6,
        mok_day=2,
    ):
        """Compute onset dates for deterministic model forecast."""
        window = onset_window
        results_list = []

        init_times = p_model.init_time.values
        lats = p_model.lat.values
        lons = p_model.lon.values

        date_method = f"MOK ({mok_month}/{mok_day} filter)" if mok else "no date filter"
        print(
            f"Processing {len(init_times)} init times x {len(lats)} lats x {len(lons)} lons..."
        )
        print(f"Using {date_method} for onset detection")
        print("Only processing forecasts initialized before observed onset dates")

        max_steps_needed = max_forecast_day + window - 1

        total_potential_inits = 0
        valid_inits = 0
        skipped_no_obs = 0
        skipped_late_init = 0
        onsets_found = 0

        for t_idx, init_time in enumerate(init_times):
            if t_idx % 5 == 0:
                print(
                    f"Processing init time {t_idx+1}/{len(init_times)}: {pd.to_datetime(init_time).strftime('%Y-%m-%d')}"
                )

            init_date = pd.to_datetime(init_time)
            year = init_date.year
            mok_date = datetime(year, mok_month, mok_day)

            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    total_potential_inits += 1

                    try:
                        obs_onset = onset_da.isel(lat=i, lon=j).values
                    except:
                        skipped_no_obs += 1
                        continue

                    if pd.isna(obs_onset):
                        skipped_no_obs += 1
                        continue

                    obs_onset_dt = pd.to_datetime(obs_onset)

                    if init_date >= obs_onset_dt:
                        skipped_late_init += 1
                        continue

                    valid_inits += 1

                    thresh = thresh_slice.isel(lat=i, lon=j).values

                    try:
                        forecast_series = (
                            p_model.isel(
                                init_time=t_idx,
                                lat=i,
                                lon=j,
                            )
                            .sel(step=slice(1, max_steps_needed))
                            .values
                        )

                        if len(forecast_series) < max_steps_needed:
                            onset_day = None
                        else:
                            onset_day = None

                            for day in range(1, max_forecast_day + 1):
                                start_idx = day - 1
                                end_idx = start_idx + window

                                if end_idx <= len(forecast_series):
                                    window_series = forecast_series[start_idx:end_idx]

                                    if (
                                        window_series[0] > 1
                                        and np.nansum(window_series) > thresh
                                    ):
                                        forecast_date = init_date + pd.Timedelta(
                                            days=day
                                        )

                                        if mok:
                                            if forecast_date.date() > mok_date.date():
                                                onset_day = day
                                                break
                                        else:
                                            onset_day = day
                                            break

                    except Exception as e:
                        print(f"Error at init_time {t_idx}, lat {i}, lon {j}: {e}")
                        onset_day = None

                    onset_date = None
                    if onset_day is not None:
                        onsets_found += 1
                        onset_date = init_date + pd.Timedelta(days=onset_day)

                    result = {
                        "init_time": init_time,
                        "lat": lat,
                        "lon": lon,
                        "onset_day": onset_day,
                        "onset_date": onset_date.strftime("%Y-%m-%d")
                        if onset_date is not None
                        else None,
                        "obs_onset_date": obs_onset_dt.strftime("%Y-%m-%d"),
                    }
                    results_list.append(result)

        onset_df = pd.DataFrame(results_list)

        print("\nProcessing Summary:")
        print(f"Total potential initializations: {total_potential_inits}")
        print(f"Skipped (no observed onset): {skipped_no_obs}")
        print(f"Skipped (initialized after observed onset): {skipped_late_init}")
        print(f"Valid initializations processed: {valid_inits}")
        print(f"Onsets found: {onsets_found}")
        print(
            f"Onset rate: {onsets_found/valid_inits:.3f}"
            if valid_inits > 0
            else "Onset rate: 0.000"
        )

        return onset_df

    @staticmethod
    def compute_metrics_multiple_years(
        years,
        model_forecast_dir,
        imd_folder,
        thres_file,
        tolerance_days=3,
        verification_window=1,
        forecast_days=15,
        max_forecast_day=15,
        mok=True,
        onset_window=5,
        mok_month=6,
        mok_day=2,
    ):
        """Compute onset metrics for multiple years."""
        metrics_df_dict = {}
        onset_da_dict = {}

        thresh_ds = xr.open_dataset(thres_file)
        thres_da = thresh_ds["MWmean"]

        for year in years:
            print(f"\n{'='*50}")
            print(f"Processing year {year}")
            print(f"{'='*50}")

            p_model = DeterministicOnsetMetrics.get_forecast_deterministic_twice_weekly(
                year, model_forecast_dir
            )
            imd = OnsetMetricsBase.load_imd_rainfall(year, imd_folder)
            onset_da = OnsetMetricsBase.detect_observed_onset(
                imd, thres_da, year, mok=mok
            )

            onset_df = DeterministicOnsetMetrics.compute_onset_for_deterministic_model(
                p_model,
                thres_da,
                onset_da,
                max_forecast_day=max_forecast_day,
                mok=mok,
                onset_window=onset_window,
                mok_month=6,
                mok_day=2,
            )

            metrics_df, summary_stats = (
                OnsetMetricsBase.compute_onset_metrics_with_windows(
                    onset_df,
                    tolerance_days=tolerance_days,
                    verification_window=verification_window,
                    forecast_days=forecast_days,
                )
            )

            metrics_df_dict[year] = metrics_df
            onset_da_dict[year] = onset_da

            print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")

        return metrics_df_dict, onset_da_dict

    @staticmethod
    def compute_metrics_multiple_years_from_loaders(
        tp_forecast,  # (day, time, lat, lon)
        tp_imd,  # (time, lat, lon)
        thres_da,  # (lat, lon)
        years=None,
        tolerance_days=3,
        verification_window=1,
        forecast_days=15,
        max_forecast_day=15,
        mok=True,
        onset_window=5,
        mok_month=6,
        mok_day=2,
    ):
        """Loader-based version of "compute_onset_metrics_for_multiple_years" using
        three *loaded* DataArrays:
            - tp_forecast: model precip, dims ('day', 'time', 'lat', 'lon')
            - tp_imd: observed precip, dims ('time', 'lat', 'lon')
            - thres_da: threshold field, dims ('lat', 'lon')

        Years are inferred from tp_forecast['time'] if not provided.
        """
        # Infer which years to process from forecast init_time
        init_times_all = pd.to_datetime(tp_forecast["time"].values)
        if years is None:
            years = sorted(np.unique(init_times_all.year))
        else:
            years = sorted(int(y) for y in years)

        metrics_df_dict = {}
        onset_da_dict = {}

        for year in years:
            print(f"\n{'='*50}")
            print(f"Processing year {year}")
            print(f"{'='*50}")

            year_mask = init_times_all.year == year
            year_init_times = init_times_all[year_mask]
            if len(year_init_times) == 0:
                print(f"No init times for year {year} in forecast data, skipping.")
                continue

            tp_fc_year = tp_forecast.sel(time=year_init_times)

            p_model = tp_fc_year.rename({"time": "init_time", "day": "step"}).transpose(
                "init_time", "lat", "lon", "step"
            )

            # Drop day=0 if present
            if int(p_model.step[0]) == 0:
                p_model = p_model.sel(step=slice(1, None))

            # Slice IMD rainfall for this year
            tp_imd_year = tp_imd.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            tp_imd_year = tp_imd_year.load()

            # detect_observed_onset expects a rainfall DataArray with dims (time, lat, lon)
            onset_da = OnsetMetricsBase.detect_observed_onset(
                tp_imd_year, thres_da, year, mok=mok
            )

            onset_df = DeterministicOnsetMetrics.compute_onset_for_deterministic_model(
                p_model,
                thres_da,
                onset_da,
                max_forecast_day=max_forecast_day,
                mok=mok,
                onset_window=onset_window,
                mok_month=6,
                mok_day=2,
            )

            metrics_df, summary_stats = (
                OnsetMetricsBase.compute_onset_metrics_with_windows(
                    onset_df,
                    tolerance_days=tolerance_days,
                    verification_window=verification_window,
                    forecast_days=forecast_days,
                )
            )

            metrics_df_dict[year] = metrics_df
            onset_da_dict[year] = onset_da

            print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")

        return metrics_df_dict, onset_da_dict
