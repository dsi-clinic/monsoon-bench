import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
from monsoonbench.metrics.climatology import ClimatologyOnsetMetrics
from monsoonbench.metrics.probabilistic.probabilistic import ProbabilisticOnsetMetrics
import pandas as pd

@pytest.fixture
def mock_climatology_data():
    lats = np.array([10.0, 12.0])
    lons = np.array([70.0, 72.0])
    data = np.array([[150, 152], [151, 153]])
    return xr.DataArray(data, coords=[("lat", lats), ("lon", lons)], name="onset_doy")

@pytest.fixture
def mock_probabilistic_data():
    lats = np.array([10.0, 12.0])
    lons = np.array([70.0, 72.0])
    steps = np.arange(1, 16)
    members = np.arange(1, 5)
    init_times = pd.date_range("2020-05-01", periods=1)
    data = np.random.rand(len(init_times), len(steps), len(lats), len(lons), len(members))
    return xr.DataArray(
        data,
        coords=[
            ("init_time", init_times),
            ("step", steps),
            ("lat", lats),
            ("lon", lons),
            ("member", members),
        ],
        name="tp",
    )

@pytest.fixture
def mock_onset():
    lats = np.array([10.0, 12.0])
    lons = np.array([70.0, 72.0])
    data = np.array([[140, 142], [141, 143]])  # Adjusted to align with init_time
    return xr.DataArray(data, coords=[("lat", lats), ("lon", lons)], name="onset_date")

def test_climatology_workflow(mock_climatology_data):
    with patch("monsoonbench.metrics.climatology.ClimatologyOnsetMetrics.compute_climatological_onset", return_value=mock_climatology_data):
        result = ClimatologyOnsetMetrics.compute_climatological_onset("/fake/imd", "/fake/threshold.nc")
        assert result.equals(mock_climatology_data)

def test_probabilistic_workflow(mock_probabilistic_data):
    mock_thresh = xr.DataArray(np.random.rand(2, 2), coords=[("lat", [10.0, 12.0]), ("lon", [70.0, 72.0])])
    mock_onset = xr.DataArray(np.array([[150, 152], [151, 153]]), coords=[("lat", [10.0, 12.0]), ("lon", [70.0, 72.0])])

    with patch("monsoonbench.metrics.probabilistic.probabilistic.ProbabilisticOnsetMetrics.get_forecast_probabilistic_twice_weekly", return_value=mock_probabilistic_data):
        result = ProbabilisticOnsetMetrics.compute_mean_onset_for_all_members(
            mock_probabilistic_data, mock_thresh, mock_onset
        )
        assert isinstance(result, xr.DataArray)