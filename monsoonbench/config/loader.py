"""Configuration loading and management for MonsoonBench.

This module provides utilities for loading configuration from YAML files
and command-line arguments, with CLI arguments taking precedence.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config("examples/configs/deterministic.yaml")
        >>> print(config['model_type'])
        deterministic
    """
    config_path = Path(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)
    return config


def setup_parser_with_config() -> argparse.ArgumentParser:
    """Setup argument parser that accepts both config file and CLI arguments.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Compute monsoon onset metrics for deterministic or probabilistic models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  monsoonbench --config config.yaml

  # Run with config file and override specific parameters
  monsoonbench --config config.yaml --years 2022 2023

  # Run with all CLI arguments (old method still works)
  monsoonbench --model_type deterministic --years 2019 2020 2021 ...
        """,
    )

    # Config file argument
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )

    # Model type selection
    parser.add_argument(
        "--model_type",
        choices=["climatology", "deterministic", "probabilistic"],
        help="Type of model to process",
    )

    # Data paths
    parser.add_argument(
        "--years", nargs="+", type=int, help="Years to process (e.g., 2019 2020 2021)"
    )
    parser.add_argument(
        "--model_forecast_dir",
        type=str,
        help="Directory containing model forecast data",
    )
    parser.add_argument(
        "--imd_folder", type=str, help="Directory containing IMD rainfall data"
    )
    parser.add_argument("--thres_file", type=str, help="Path to threshold NetCDF file")
    parser.add_argument("--shpfile_path", type=str, help="Path to India shapefile")

    # Metric parameters
    parser.add_argument(
        "--verification_window",
        type=int,
        help="Days after init to start validation window (default: 1)",
    )
    parser.add_argument(
        "--forecast_days",
        type=int,
        help="Length of forecast window in days (default: 15)",
    )
    parser.add_argument(
        "--max_forecast_day",
        type=int,
        help="Maximum forecast day to consider for onset (default: 15)",
    )
    parser.add_argument(
        "--onset_window", type=int, help="Onset detection window in days (default: 5)"
    )
    parser.add_argument(
        "--mok",
        action="store_true",
        help="Use MOK date filter (June 2nd) for onset detection",
    )
    parser.add_argument(
        "--mok_month", type=int, help="Month for MOK date filter (default: 6)"
    )
    parser.add_argument(
        "--mok_day", type=int, help="Day for MOK date filter (default: 2)"
    )

    # Output parameters
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output NetCDF file name (default: spatial_metrics.nc)",
    )
    parser.add_argument(
        "--plot_dir", type=str, help="Directory to save plot PNG file (optional)"
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        help="Figure size in inches [width height] (default: 18 6)",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        help="Directory to export visualization-ready data (optional)",
    )
    parser.add_argument(
        "--download_formats",
        nargs="+",
        help="Formats for exported data (any of: netcdf, csv, parquet, json)",
    )
    parser.add_argument(
        "--download_metrics",
        nargs="+",
        help="Subset of spatial metrics to export (default: all)",
    )
    parser.add_argument(
        "--download_keep_nans",
        action="store_true",
        help="Keep rows with all-NaN metric values when exporting tabular data",
    )

    return parser


def get_config() -> argparse.Namespace:
    """Parse arguments and merge with config file if provided.

    Command-line arguments take precedence over config file.

    Returns:
        Configuration object as argparse.Namespace

    Raises:
        SystemExit: If required parameters are missing
    """
    parser = setup_parser_with_config()
    args = parser.parse_args()

    config = {
        "verification_window": 1,
        "forecast_days": 15,
        "max_forecast_day": 15,
        "onset_window": 5,
        "output_file": "spatial_metrics.nc",
        "figsize": [18, 6],
        "plot_dir": None,
        "mok": False,
        "mok_month": 6,
        "mok_day": 2,
        "download_dir": None,
        "download_formats": None,
        "download_metrics": None,
        "download_keep_nans": False,
    }

    # Load from YAML if provided
    if args.config:
        yaml_config = load_config(args.config)
        config.update(yaml_config)
        print(f"Loaded configuration from: {args.config}")

    # Override with command-line arguments (if explicitly provided)
    cli_args = vars(args)
    for key, value in cli_args.items():
        if key != "config" and value is not None:
            if key == "mok":
                if value is True:
                    config[key] = True
            else:
                config[key] = value

    # Validate required parameters
    required = ["model_type", "years", "imd_folder", "thres_file", "shpfile_path"]
    missing = [param for param in required if param not in config]

    if missing:
        parser.error(
            f"Missing required parameters: {', '.join(missing)}\n"
            f"Provide them via config file (--config) or command-line arguments."
        )
    if config.get("model_type") in ["deterministic", "probabilistic"]:
        if "model_forecast_dir" not in config:
            parser.error(
                "--model_forecast_dir is required for deterministic and probabilistic models"
            )

    return argparse.Namespace(**config)


if __name__ == "__main__":
    # Test the config loader
    config = get_config()
    print("\nFinal configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
