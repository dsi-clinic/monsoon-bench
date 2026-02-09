from pathlib import Path
import os
import matplotlib.pyplot as plt

from monsoonbench.metrics import (
    ClimatologyOnsetMetrics,
    DeterministicOnsetMetrics,
    ProbabilisticOnsetMetrics,
)
from monsoonbench.visualization import (
    create_model_comparison_table,
    download_spatial_metrics_data,
    plot_model_comparison_dual_axis,
    create_probabilistic_comparison_table,
    plot_probabilistic_comparison_dual_axis,
)
from monsoonbench.visualization import plot_reliability_diagram, create_heatmap

from monsoonbench.visualization.spatial import plot_spatial_metrics
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr

# Initialize Metric Computation Classes
pr = ProbabilisticOnsetMetrics()
cl = ClimatologyOnsetMetrics()

def make_figure_2(args):
    if args['max_forecast_day'] == 15:
        day_bins = [(1, 5), (6, 10), (11, 15)]
    elif args['max_forecast_day'] == 30:
        day_bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
    else:
        raise ValueError(f"Unsupported max_forecast_day: {args['max_forecast_day']}")
    
    # Print args metadata
    print("="*60)
    print("S2S MONSOON ONSET SKILL SCORE ANALYSIS")
    print("="*60)
    print(f"Model: {args['model_name']}")
    print(f"Years: {args['years']}")
    print(f"Max forecast day: {args['max_forecast_day']}")
    print(f"Day bins: {day_bins}")
    print(f"MOK filter: {args['mok']} ({'June 2nd' if args['mok'] else 'May 1st'})")
    if args['save_dir']:
        print(f"Output directory: {args['save_dir']}")
    else:
        print("Output directory: current directory")
    print("="*60)

    # Create forecast observation dataframe
    print("\n1. Processing forecast model...")
    forecast_obs_df = pr.multi_year_forecast_obs_pairs(
            args['years'], args['model_forecast_dir'], args['imd_folder'], 
            args['thres_file'], args['mem_num'],
            args['max_forecast_day'], day_bins, 
            date_filter_year=args['date_filter_year'], 
            file_pattern=args['file_pattern'],
            mok=args['mok']
        )

    # Calculate brier and AUC forecast scores from observations
    print("\n2. Calculating brier and AUC forecast scores...")
    brier_forecast = pr.calculate_brier_score(forecast_obs_df)
    rps_forecast = pr.calculate_rps(forecast_obs_df)
    auc_forecast = pr.calculate_auc(forecast_obs_df)

    print("\n3. Computing climatological dataset...")
    thresh_ds = xr.open_dataset(args['thres_file'])
    thresh_slice = thresh_ds['MWmean']
    clim_onset = cl.compute_climatological_onset_dataset(
        args['imd_folder'], thresh_slice, years=None, mok=args['mok']
    )

    print("\n4. Processing climatological forecasts...")
    climatology_obs_df = cl.multi_year_climatological_forecast_obs_pairs(
        clim_onset, args['years'], day_bins, args['mem_num'], args['model_forecast_dir'], 
        date_filter_year=args['date_filter_year'], 
        file_pattern=args['file_pattern'], 
        max_forecast_day=args['max_forecast_day'], 
        mok=args['mok']
    )

    print("\n5. Calculating climatology scores...")
    brier_climatology = cl.calculate_brier_score_climatology(climatology_obs_df)
    rps_climatology = cl.calculate_rps_climatology(climatology_obs_df)
    auc_climatology = cl.calculate_auc_climatology(climatology_obs_df)

    # Calculate skill scores
    print("\n6. Calculating skill scores...")
    skill_results = pr.calculate_skill_scores(
        brier_forecast, rps_forecast,
        brier_climatology, rps_climatology
    )

    # Create heatmap
    print("\n8. Creating heatmap...")
    heatmap_file = create_heatmap(
        skill_results, auc_forecast, auc_climatology,
        brier_forecast, brier_climatology, args['model_name'], args['max_forecast_day'],
        save_dir=str(args['save_dir'])
    )

    ################# FIGURE C AND D
    # Loading models
    probabilistic_df, onset_da_dict = pr.compute_metrics_multiple_years(
                years=args["years"],
                imd_folder=args["imd_folder"],
                thres_file=args["thres_file"],
                model_forecast_dir=args["model_forecast_dir"],
                tolerance_days=3,
                verification_window=1,
                forecast_days=15,
                max_forecast_day=15,
                mok=True,
                onset_window=5,
                mok_month=6,
                mok_day=2,
            )

    # --- 5. Build Single-Model Comparison Table ---
    # Average the spatial maps to get scalar values for the table
    summary_row = {
        "model": args['model_name'],
        "fair_brier_skill_score": skill_results["fair_brier_skill_score"],
        "fair_rps_skill_score": skill_results["fair_rps_skill_score"],
    }

    single_model_df = pd.DataFrame([summary_row]).set_index("model")

    # --- 6. Plotting ---
    fig, (ax_left, ax_right) = plot_probabilistic_comparison_dual_axis(
        comparison_df=single_model_df,
        skill_cols=["fair_brier_skill_score", "fair_rps_skill_score"],
        title=f"Probabilistic Performance: {args['model_name'].upper()}",
        rotation=0
    )

    # Save output
    output_path = f"{args['save_dir']}/{args['model_name']}_probabilistic_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Evaluation complete for {args['model_name']}.")


    ################# FIGURE E
    # Plot reliability diagram and get results
    fig, ax, reliability_df = plot_reliability_diagram(
        forecast_obs_df, 
        args['years'], 
        args['max_forecast_day'],
        args['save_dir']
    )

    if args['mok']:
        mok_suffix = '_mok'
    else:
        mok_suffix = '_no_mok'
    # Save reliability DataFrame as CSV
    reliability_df.to_csv("reliability_results_30day", index=False)
    print(f"Reliability results saved to: path")
