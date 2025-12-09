"""Unit tests for monsoonbench CLI main function.

Tests the main CLI entry point including imports, class instantiation,
and method availability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Self


class TestCliImports:
    """Test that all required imports work."""

    def test_import_os(self: Self):
        """Test os module import."""
        import os

        assert os is not None

    def test_import_xarray(self: Self):
        """Test xarray import."""
        import xarray as xr

        assert xr is not None

    def test_import_config(self: Self):
        """Test config module import."""
        from monsoonbench.config import get_config

        assert callable(get_config)

    def test_import_metrics_classes(self: Self):
        """Test metrics classes import."""
        from monsoonbench.metrics import (
            ClimatologyOnsetMetrics,
            DeterministicOnsetMetrics,
            ProbabilisticOnsetMetrics,
        )

        assert ClimatologyOnsetMetrics is not None
        assert DeterministicOnsetMetrics is not None
        assert ProbabilisticOnsetMetrics is not None

    def test_import_visualization(self: Self):
        """Test visualization function import."""
        from monsoonbench.visualization.spatial import plot_spatial_metrics

        assert callable(plot_spatial_metrics)

    def test_import_main_function(self: Self):
        """Test main function import."""
        from monsoonbench.cli.main import main

        assert callable(main)


class TestMetricsClasses:
    """Test metrics classes have required methods."""

    def test_climatology_has_required_methods(self: Self):
        """Test ClimatologyOnsetMetrics has all required methods."""
        from monsoonbench.metrics import ClimatologyOnsetMetrics

        # Test class-specific methods (inherited methods tested separately)
        required_methods = [
            "compute_climatology_baseline_multiple_years",
            "compute_climatological_onset",
            "get_initialization_dates",
            "compute_climatology_as_forecast",
        ]

        for method in required_methods:
            assert hasattr(ClimatologyOnsetMetrics, method), (
                f"ClimatologyOnsetMetrics missing method: {method}"
            )

        # Verify it inherits from base class (which has create_spatial_far_mr_mae, etc.)
        from monsoonbench.metrics.base import OnsetMetricsBase

        assert issubclass(ClimatologyOnsetMetrics, OnsetMetricsBase), (
            "ClimatologyOnsetMetrics should inherit from OnsetMetricsBase"
        )

    def test_deterministic_has_required_methods(self: Self):
        """Test DeterministicOnsetMetrics has all required methods."""
        from monsoonbench.metrics import DeterministicOnsetMetrics

        required_methods = [
            "compute_metrics_multiple_years",
            "create_spatial_far_mr_mae",
            "get_forecast_deterministic_twice_weekly",
            "compute_onset_for_deterministic_model",
        ]

        for method in required_methods:
            assert hasattr(DeterministicOnsetMetrics, method), (
                f"DeterministicOnsetMetrics missing method: {method}"
            )

    def test_probabilistic_has_required_methods(self: Self):
        """Test ProbabilisticOnsetMetrics has all required methods."""
        from monsoonbench.metrics import ProbabilisticOnsetMetrics

        required_methods = [
            "compute_metrics_multiple_years",
            "create_spatial_far_mr_mae",
            "get_forecast_probabilistic_twice_weekly",
            "compute_mean_onset_for_all_members",
        ]

        for method in required_methods:
            assert hasattr(ProbabilisticOnsetMetrics, method), (
                f"ProbabilisticOnsetMetrics missing method: {method}"
            )

    def test_base_class_has_required_methods(self: Self):
        """Test OnsetMetricsBase has required methods."""
        from monsoonbench.metrics.base import OnsetMetricsBase

        required_methods = [
            "create_spatial_far_mr_mae",
            "compute_onset_metrics_with_windows",
            "load_imd_rainfall",
        ]

        for method in required_methods:
            assert hasattr(OnsetMetricsBase, method), (
                f"OnsetMetricsBase missing method: {method}"
            )


class TestVisualization:
    """Test visualization functions."""

    def test_plot_spatial_metrics_exists(self: Self):
        """Test plot_spatial_metrics function exists and is callable."""
        from monsoonbench.visualization.spatial import plot_spatial_metrics

        assert callable(plot_spatial_metrics)

    def test_plot_spatial_metrics_signature(self: Self):
        """Test plot_spatial_metrics has correct signature."""
        import inspect

        from monsoonbench.visualization.spatial import plot_spatial_metrics

        sig = inspect.signature(plot_spatial_metrics)
        params = list(sig.parameters.keys())

        # Check required parameters exist
        assert "spatial_metrics" in params
        assert "shpfile_path" in params

        # Check optional parameters exist
        assert "figsize" in params
        assert "save_path" in params


class TestMainFunction:
    """Test the main CLI function."""

    def test_main_function_exists(self: Self):
        """Test main function exists."""
        from monsoonbench.cli.main import main

        assert callable(main)

    def test_main_function_signature(self: Self):
        """Test main function has correct signature."""
        import inspect

        from monsoonbench.cli.main import main

        sig = inspect.signature(main)

        # Main should have no required parameters
        required_params = [
            p for p in sig.parameters.values() if p.default == inspect.Parameter.empty
        ]
        assert len(required_params) == 0, "main() should not have required parameters"

    def test_main_return_type(self: Self):
        """Test main function returns None."""
        import inspect

        from monsoonbench.cli.main import main

        sig = inspect.signature(main)
        # Should return None
        assert sig.return_annotation in [
            None,
            "None",
            type(None),
            inspect.Signature.empty,
        ]


class TestMetricsMethodSignatures:
    """Test method signatures of metrics classes."""

    def test_climatology_baseline_signature(self: Self):
        """Test compute_climatology_baseline_multiple_years signature."""
        import inspect

        from monsoonbench.metrics import ClimatologyOnsetMetrics

        sig = inspect.signature(
            ClimatologyOnsetMetrics.compute_climatology_baseline_multiple_years
        )
        params = list(sig.parameters.keys())

        # Check required parameters
        assert "years" in params
        assert "imd_folder" in params
        assert "thres_file" in params

    def test_deterministic_metrics_signature(self: Self):
        """Test compute_metrics_multiple_years signature for deterministic."""
        import inspect

        from monsoonbench.metrics import DeterministicOnsetMetrics

        sig = inspect.signature(
            DeterministicOnsetMetrics.compute_metrics_multiple_years
        )
        params = list(sig.parameters.keys())

        # Check required parameters
        assert "years" in params
        assert "model_forecast_dir" in params
        assert "imd_folder" in params
        assert "thres_file" in params

    def test_probabilistic_metrics_signature(self: Self):
        """Test compute_metrics_multiple_years signature for probabilistic."""
        import inspect

        from monsoonbench.metrics import ProbabilisticOnsetMetrics

        sig = inspect.signature(
            ProbabilisticOnsetMetrics.compute_metrics_multiple_years
        )
        params = list(sig.parameters.keys())

        # Check required parameters
        assert "years" in params
        assert "model_forecast_dir" in params
        assert "imd_folder" in params
        assert "thres_file" in params

    def test_create_spatial_far_mr_mae_signature(self: Self):
        """Test create_spatial_far_mr_mae signature."""
        import inspect

        from monsoonbench.metrics.base import OnsetMetricsBase

        sig = inspect.signature(OnsetMetricsBase.create_spatial_far_mr_mae)
        params = list(sig.parameters.keys())

        # Check required parameters
        assert "metrics_df_dict" in params
        assert "onset_da_dict" in params


@pytest.mark.parametrize(
    "model_type,metrics_class",
    [
        ("climatology", "ClimatologyOnsetMetrics"),
        ("deterministic", "DeterministicOnsetMetrics"),
        ("probabilistic", "ProbabilisticOnsetMetrics"),
    ],
)
class TestMetricsClassSelection:
    """Test that correct metrics class is used for each model type."""

    def test_class_exists_for_model_type(self: Self, model_type, metrics_class):
        """Test that metrics class exists for each model type."""
        from monsoonbench import metrics

        assert hasattr(metrics, metrics_class)

    def test_class_has_compute_method(self: Self, model_type, metrics_class):
        """Test that each class has appropriate compute method."""
        from monsoonbench import metrics

        cls = getattr(metrics, metrics_class)

        if model_type == "climatology":
            assert hasattr(cls, "compute_climatology_baseline_multiple_years")
        else:
            assert hasattr(cls, "compute_metrics_multiple_years")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
