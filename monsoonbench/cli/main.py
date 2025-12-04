"""Command-line interface for MonsoonBench.

This module provides the main entry point for the monsoonbench CLI command.
"""

from pathlib import Path

import xarray as xr

from monsoonbench.config import get_config
from monsoonbench.data import load
from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    ProbabilisticOnsetMetrics,
)
from monsoonbench.visualization import (
    download_spatial_metrics_data,
    plot_spatial_metrics,
)


def main() -> None:
    """Main entry point for the MonsoonBench CLI.

    Parses configuration from CLI arguments or YAML file,
    computes onset metrics, and saves results.
    """
    args = get_config()

    print(f"Processing {args.model_type} model")
    print(f"Years: {args.years}")
    print(f"MOK filter: {args.mok}")

    # Create DataLoader instance
    data_loader = load(
        name="imd_rain",
        root=args.imd_folder,
        years=args.years,
        subset={"time": slice("2012-01-01", "2014-12-31")},
        chunks={"time": 64},
    )

    # Initialize appropriate metrics class with DataLoader
    if args.model_type == "deterministic":
        metrics = DeterministicOnsetMetrics(data_loader)
    elif args.model_type == "probabilistic":
        metrics = ProbabilisticOnsetMetrics(data_loader)
    else:
        metrics = ClimatologyOnsetMetrics(data_loader)

    # Determine tolerance days based on forecast window
    EXTENDED_RANGE_MAX_DAYS = 15
    if 1 <= args.forecast_days <= EXTENDED_RANGE_MAX_DAYS:
        tolerance_days = 3
    else:
        tolerance_days = 5

    # Process using common interface
    if args.model_type == "climatology":
        metrics_df_dict, climatological_onset_doy = (
            metrics.compute_climatology_baseline_multiple_years(
                years=args.years,
                tolerance_days=tolerance_days,
                verification_window=args.verification_window,
                forecast_days=args.forecast_days,
                max_forecast_day=args.max_forecast_day,
                mok=args.mok,
                onset_window=args.onset_window,
                mok_month=args.mok_month,
                mok_day=args.mok_day,
            )
        )
        onset_da_dict = dict.fromkeys(args.years, climatological_onset_doy)
    else:
        metrics_df_dict, onset_da_dict = metrics.compute_metrics_multiple_years(
            years=args.years,
            tolerance_days=tolerance_days,
            verification_window=args.verification_window,
            forecast_days=args.forecast_days,
            max_forecast_day=args.max_forecast_day,
            mok=args.mok,
            onset_window=args.onset_window,
            mok_month=args.mok_month,
            mok_day=args.mok_day,
        )

    # Create spatial metrics
    spatial_metrics = metrics.create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)

    years_str = f"{min(args.years)}-{max(args.years)}"
    window_str = f"{args.verification_window}-{args.forecast_days}day"
    mok_str = "MOK" if args.mok else "noMOK"
    artifact_basename = (
        f"spatial_metrics_{args.model_type}_{years_str}_{window_str}_{mok_str}"
    )

    # Save to NetCDF
    ds = xr.Dataset(spatial_metrics)
    ds.attrs["title"] = "Monsoon Onset MAE, FAR, MR Analysis"
    ds.attrs["model_type"] = args.model_type
    ds.attrs["years"] = str(args.years)
    ds.attrs["tolerance_days"] = tolerance_days
    ds.attrs["verification_window"] = args.verification_window
    ds.attrs["forecast_days"] = args.forecast_days
    ds.attrs["max_forecast_day"] = args.max_forecast_day
    ds.attrs["mok_filter"] = int(args.mok)

    ds.to_netcdf(args.output_file)
    print(f"\nSpatial metrics saved to: {args.output_file}")

    if args.download_dir:
        download_formats = args.download_formats or ["netcdf"]
        print(
            "\nExporting visualization data to "
            f"{args.download_dir} in formats {download_formats}..."
        )
        download_paths = download_spatial_metrics_data(
            spatial_metrics=ds,
            output_dir=args.download_dir,
            filename=artifact_basename,
            formats=download_formats,
            metadata=dict(ds.attrs),
            metrics=args.download_metrics,
            dropna=not args.download_keep_nans,
        )
        for path in download_paths:
            print(f" - {path}")

    # Generate and save plot if plot_dir is specified
    if args.plot_dir:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

        # Generate plot filename
        plot_filename = f"{artifact_basename}.png"
        plot_path = Path(args.plot_dir) / plot_filename

        print("\nGenerating spatial plot...")

        # Create the plot
        fig, axes = plot_spatial_metrics(
            spatial_metrics,
            args.shpfile_path,
            figsize=tuple(args.figsize),
            save_path=plot_path,
        )

        print(f"Plot saved to: {plot_path}")

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Mean MAE: {float(ds['mean_mae'].mean().values):.2f} days")
    print(f"Mean FAR: {float(ds['false_alarm_rate'].mean().values) * 100:.1f}%")
    print(f"Mean Miss Rate: {float(ds['miss_rate'].mean().values) * 100:.1f}%")


if __name__ == "__main__":
    main()
