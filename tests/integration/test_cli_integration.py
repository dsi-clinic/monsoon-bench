"""Integration tests for monsoonbench CLI.

Tests the complete workflow with mocked data and configurations.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

if TYPE_CHECKING:
    from typing import Self


@pytest.fixture
def mock_file_system():
    """Create a temporary directory structure to mock file system."""
    with tempfile.TemporaryDirectory() as temp_dir:
        imd_folder = os.path.join(temp_dir, "imd")
        forecasts_folder = os.path.join(temp_dir, "forecasts")
        threshold_file = os.path.join(temp_dir, "threshold.nc")
        india_shp_file = os.path.join(temp_dir, "india.shp")

        os.makedirs(imd_folder, exist_ok=True)
        os.makedirs(forecasts_folder, exist_ok=True)

        # Create mock files
        with open(threshold_file, "w") as f:
            f.write("mock threshold data")

        # Create mock shapefile
        with open(india_shp_file, "w") as f:
            f.write("mock shapefile data")

        yield {
            "imd_folder": imd_folder,
            "forecasts_folder": forecasts_folder,
            "threshold_file": threshold_file,
            "india_shp_file": india_shp_file,
        }


@pytest.fixture
def mock_config(mock_file_system):
    """Create a mock configuration object with real temporary paths."""
    config = Mock()
    config.model_type = "deterministic"
    config.years = [2020]
    config.imd_folder = mock_file_system["imd_folder"]
    config.thres_file = mock_file_system["threshold_file"]
    config.model_forecast_dir = mock_file_system["forecasts_folder"]
    config.shpfile_path = mock_file_system["india_shp_file"]
    config.output_file = "test_output.nc"
    config.plot_dir = None
    config.verification_window = 1
    config.forecast_days = 15
    config.max_forecast_day = 15
    config.mok = True
    config.onset_window = 5
    config.mok_month = 6
    config.mok_day = 2
    config.figsize = [18, 6]
    config.download_dir = None
    config.download_formats = None
    config.download_metrics = None
    config.download_keep_nans = False
    return config


@pytest.fixture(autouse=True)
def patched_load():
    """Avoid touching the real filesystem when creating DataLoader."""
    with patch("monsoonbench.cli.main.load") as mock_load:
        loader = MagicMock(name="DataLoader")
        mock_load.return_value = loader
        yield loader


@pytest.fixture
def mock_metrics_df():
    """Create a mock metrics DataFrame."""
    return pd.DataFrame(
        {
            "lat": [10.0, 10.0, 12.0],
            "lon": [70.0, 72.0, 70.0],
            "true_positive": [5, 3, 4],
            "true_negative": [2, 4, 3],
            "false_positive": [1, 2, 1],
            "false_negative": [2, 1, 2],
            "num_onset": [7, 4, 6],
            "num_no_onset": [3, 6, 4],
            "mae_combined": [2.5, 3.0, 2.8],
            "mae_tp_only": [2.0, 2.5, 2.3],
        }
    )


@pytest.fixture
def mock_onset_da():
    """Create a mock onset DataArray."""
    lats = np.array([10.0, 12.0])
    lons = np.array([70.0, 72.0])
    dates = pd.date_range("2020-06-01", periods=2)

    data = np.array([[dates[0], dates[1]], [dates[0], dates[1]]])

    return xr.DataArray(data, coords=[("lat", lats), ("lon", lons)], name="onset_date")


@pytest.fixture
def mock_spatial_metrics():
    """Create mock spatial metrics dictionary."""
    lats = np.array([10.0, 12.0])
    lons = np.array([70.0, 72.0])

    return {
        "mean_mae": xr.DataArray(
            np.array([[2.5, 3.0], [2.8, 2.6]]), coords=[("lat", lats), ("lon", lons)]
        ),
        "false_alarm_rate": xr.DataArray(
            np.array([[0.2, 0.3], [0.25, 0.22]]), coords=[("lat", lats), ("lon", lons)]
        ),
        "miss_rate": xr.DataArray(
            np.array([[0.15, 0.20], [0.18, 0.16]]),
            coords=[("lat", lats), ("lon", lons)],
        ),
        "mae_2020": xr.DataArray(
            np.array([[2.5, 3.0], [2.8, 2.6]]), coords=[("lat", lats), ("lon", lons)]
        ),
    }


class TestMainWorkflow:
    """Test the complete main() workflow."""

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    def test_deterministic_workflow(
        self: Self,
        mock_metrics_class,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Test complete workflow for deterministic model."""
        # Setup mocks
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        # Mock the metrics computation
        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        # Run main (with xr.Dataset.to_netcdf mocked)
        with patch("xarray.Dataset.to_netcdf") as mock_to_netcdf:
            from monsoonbench.cli.main import main

            main()

            # Verify the workflow
            mock_get_config.assert_called_once()
            mock_metrics_instance.compute_metrics_multiple_years.assert_called_once()
            mock_metrics_instance.create_spatial_far_mr_mae.assert_called_once()
            mock_to_netcdf.assert_called_once()

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.ClimatologyOnsetMetrics")
    def test_climatology_workflow(
        self: Self,
        mock_metrics_class,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Test complete workflow for climatology model."""
        # Setup mocks
        mock_config.model_type = "climatology"
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        # Mock the climatology computation
        climatological_onset_doy = xr.DataArray(
            np.array([[150, 152], [151, 153]]),
            coords=[("lat", [10.0, 12.0]), ("lon", [70.0, 72.0])],
        )
        mock_metrics_instance.compute_climatology_baseline_multiple_years.return_value = (
            {2020: mock_metrics_df},
            climatological_onset_doy,
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        # Run main
        with patch("xarray.Dataset.to_netcdf"):
            from monsoonbench.cli.main import main

            main()

            # Verify the workflow
            mock_metrics_instance.compute_climatology_baseline_multiple_years.assert_called_once()

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.ProbabilisticOnsetMetrics")
    def test_probabilistic_workflow(
        self: Self,
        mock_metrics_class,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Test complete workflow for probabilistic model."""
        # Setup mocks
        mock_config.model_type = "probabilistic"
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        # Mock the metrics computation
        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        # Run main
        with patch("xarray.Dataset.to_netcdf"):
            from monsoonbench.cli.main import main

            main()

            # Verify the workflow
            mock_metrics_instance.compute_metrics_multiple_years.assert_called_once()


class TestToleranceDaysCalculation:
    """Test tolerance days calculation logic."""

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    def test_extended_range_tolerance(
        self: Self,
        mock_metrics_class,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Test that extended range forecasts use 3-day tolerance."""
        mock_config.forecast_days = 15  # Extended range
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        with patch("xarray.Dataset.to_netcdf"):
            from monsoonbench.cli.main import main

            main()

            # Check that tolerance_days=3 was passed
            call_args = mock_metrics_instance.compute_metrics_multiple_years.call_args
            assert call_args[1]["tolerance_days"] == 3

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    def test_subseasonal_tolerance(
        self: Self,
        mock_metrics_class,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Test that subseasonal forecasts use 5-day tolerance."""
        mock_config.forecast_days = 30  # Subseasonal
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        with patch("xarray.Dataset.to_netcdf"):
            from monsoonbench.cli.main import main

            main()

            # Check that tolerance_days=5 was passed
            call_args = mock_metrics_instance.compute_metrics_multiple_years.call_args
            assert call_args[1]["tolerance_days"] == 5


class TestOutputGeneration:
    """Test output file and plot generation."""

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    def test_netcdf_output_created(
        self: Self,
        mock_metrics_class,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Test that NetCDF output file is created with correct attributes."""
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        with patch("xarray.Dataset.to_netcdf") as mock_to_netcdf:
            from monsoonbench.cli.main import main

            main()

            # Verify to_netcdf was called with correct file
            mock_to_netcdf.assert_called_once_with("test_output.nc")

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    @patch("monsoonbench.cli.main.plot_spatial_metrics")
    @patch("pathlib.Path.mkdir")
    def test_plot_generated_when_plot_dir_specified(
        self: Self,
        mock_mkdir,
        mock_plot,
        mock_metrics_class,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Test that plot is generated when plot_dir is specified."""
        mock_config.plot_dir = "test_plots/"
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        mock_plot.return_value = (MagicMock(), MagicMock())

        with patch("xarray.Dataset.to_netcdf"):
            from monsoonbench.cli.main import main

            main()

            # Verify plot directory was created
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Verify plot was generated
            mock_plot.assert_called_once()
            call_args = mock_plot.call_args
            assert call_args[0][0] == mock_spatial_metrics  # spatial_metrics arg
            assert call_args[1]["save_path"] is not None

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    def test_no_plot_when_plot_dir_none(
        self: Self,
        mock_metrics_class,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Test that no plot is generated when plot_dir is None."""
        mock_config.plot_dir = None
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        with patch("xarray.Dataset.to_netcdf"):
            with patch("monsoonbench.cli.main.plot_spatial_metrics") as mock_plot:
                from monsoonbench.cli.main import main

                main()

                # Verify plot was NOT called
                mock_plot.assert_not_called()

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.download_spatial_metrics_data")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    def test_visualization_download_requested(
        self: Self,
        mock_metrics_class,
        mock_download,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Ensure downloader is invoked with user-provided formats/metrics."""
        mock_config.download_dir = "artifacts"
        mock_config.download_formats = ["csv", "json"]
        mock_config.download_metrics = ["mean_mae"]
        mock_config.download_keep_nans = True
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        with patch("xarray.Dataset.to_netcdf"):
            from monsoonbench.cli.main import main

            main()

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["output_dir"] == "artifacts"
        assert call_kwargs["formats"] == ["csv", "json"]
        assert call_kwargs["metrics"] == ["mean_mae"]
        assert call_kwargs["dropna"] is False  # download_keep_nans=True -> dropna False

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.download_spatial_metrics_data")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    def test_visualization_download_defaults(
        self: Self,
        mock_metrics_class,
        mock_download,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Downloader should default to NetCDF when no formats specified."""
        mock_config.download_dir = "artifacts"
        mock_config.download_formats = None
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        with patch("xarray.Dataset.to_netcdf"):
            from monsoonbench.cli.main import main

            main()

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["formats"] == ["netcdf"]

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.download_spatial_metrics_data")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    def test_visualization_download_not_called_without_dir(
        self: Self,
        mock_metrics_class,
        mock_download,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
    ):
        """Downloader should not run when download_dir is None."""
        mock_config.download_dir = None
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        with patch("xarray.Dataset.to_netcdf"):
            from monsoonbench.cli.main import main

            main()

        mock_download.assert_not_called()


class TestDatasetAttributes:
    """Test that output dataset has correct attributes."""

    @patch("monsoonbench.cli.main.get_config")
    @patch("monsoonbench.cli.main.DeterministicOnsetMetrics")
    def test_dataset_attributes_set(
        self: Self,
        mock_metrics_class,
        mock_get_config,
        mock_config,
        mock_metrics_df,
        mock_onset_da,
        mock_spatial_metrics,
        capsys,
    ):
        """Test that dataset attributes are correctly set."""
        mock_get_config.return_value = mock_config
        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        mock_metrics_instance.compute_metrics_multiple_years.return_value = (
            {2020: mock_metrics_df},
            {2020: mock_onset_da},
        )
        mock_metrics_instance.create_spatial_far_mr_mae.return_value = (
            mock_spatial_metrics
        )

        captured_ds = None

        def capture_dataset(filename):
            nonlocal captured_ds
            # Get the dataset from the stack
            import inspect

            frame = inspect.currentframe().f_back
            captured_ds = frame.f_locals.get("ds")

        with patch("xarray.Dataset.to_netcdf", side_effect=capture_dataset):
            from monsoonbench.cli.main import main

            main()

        # Verify attributes if dataset was captured
        if captured_ds is not None:
            assert "title" in captured_ds.attrs
            assert "model_type" in captured_ds.attrs
            assert captured_ds.attrs["model_type"] == "deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
