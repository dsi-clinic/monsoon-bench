import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat
import xarray as xr

from utils import get_model_dfs, get_plot_metrics, get_climatological_dfs

# Loading models
config = {
    "years": [
        2019,
        2020,
        2021,
        2022,
        2023,
        2024
    ],  # The years over which the model forecast should be evaluated
    "alt_recent_years": np.arange(2013, 2019),
    "imd_folder": f"{parent_dir}/monsoon-benchmark/imd_rainfall_data/4p0",  # Ground truth rainfall data
    "thres_file": f"{parent_dir}/monsoon-benchmark/imd_onset_threshold/mwset4x4.nc4",  # Threshold for the onset of the monsoon
    "shpfile_path": f"{parent_dir}/monsoon-benchmark/ind_map_shpfile/india_shapefile.shp",  # Shapefile of India
    # "model_forecast_dir": "/data/model_forecast_data/aifs/daily_0z/tp_2p0_lsm",
}

model_paths = {
    "IFS": f"{parent_dir}/monsoon-benchmark/model_forecast_data/IFS-S2S/tp_4p0",
    "AIFS": f"{parent_dir}/monsoon-benchmark/model_forecast_data/aifs/tp_4p0_lsm",
    "FuXi": f"{parent_dir}/monsoon-benchmark/model_forecast_data/fuxi/output_daily_paper_0z_4p0/tp_lsm",
    "Graphcast": f"{parent_dir}/monsoon-benchmark/model_forecast_data/graphcast37/output_twice_weekly_paper_0z_4p0/tp_lsm",
    "GenCast": f"{parent_dir}/monsoon-benchmark/model_forecast_data/gencast52/tp_lsm_4p0",
    "FuXi-S2S": f"{parent_dir}/monsoon-benchmark/model_forecast_data/fuxi_s2s/tp_4p0",
    "NGCM": f"{parent_dir}/monsoon-benchmark/model_forecast_data/ngcm51/twice_weekly_0z/tp_4p0"
}

wyi_paths = {
    "IFS": f"{parent_dir}/monsoon-benchmark/model_forecast_data/IFS-S2S/wy",
    "AIFS": f"{parent_dir}/monsoon-benchmark/model_forecast_data/aifs/wy",
    "FuXi": f"{parent_dir}/monsoon-benchmark/model_forecast_data/fuxi/wy",
    "Graphcast": f"{parent_dir}/monsoon-benchmark/model_forecast_data/graphcast37/wy",
    "GenCast": f"{parent_dir}/monsoon-benchmark/model_forecast_data/gencast52/wy",
    "NGCM": f"{parent_dir}/monsoon-benchmark/model_forecast_data/ngcm51/twice_weekly_0z/wy"
}

year_ranges = {
    "AIFS": np.arange(2019, 2025),
    "IFS": np.arange(2019, 2024),
    "FuXi": np.arange(2019,2025), 
    "Graphcast": np.arange(2019,2025), 
    "GenCast": np.arange(2019, 2025),
    "FuXi-S2S": np.arange(2019, 2022),
    "NGCM": np.arange(2019, 2025)
}

extended_years = {
    "AIFS": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
    "IFS": np.arange(2013, 2024),
    "FuXi": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))), #Deterministic
    "Graphcast": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))), #deterministic
    "GenCast": np.arange(2019, 2025),
    "FuXi-S2S": np.arange(2019, 2022),
    "NGCM": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025)))
}

def save_data(mat_dict, output_dir, save_path):
    out_path = f"{output_dir}/{save_path}.mat"
    savemat(out_path, mat_dict)
    print("Saved to:", out_path)

if __name__ == "__main__":
    config = {
        "years": [
            2019,
            2020,
            2021,
            2022,
            2023,
            2024
        ],
        "imd_folder": f"{parent_dir}/monsoon-benchmark/imd_rainfall_data/4p0",
        "thres_file": f"{parent_dir}/monsoon-benchmark/imd_onset_threshold/mwset4x4.nc4",
        "shpfile_path": f"{parent_dir}/monsoon-benchmark/ind_map_shpfile/india_shapefile.shp"
    }

    print("Loading Model Data (2019-2024)...")
    model_dfs_15, model_onsets_15 = get_model_dfs(model_paths, year_ranges=year_ranges, config=config, days=15)
    model_dfs_30, model_onsets_30 = get_model_dfs(model_paths, year_ranges=year_ranges, config=config, days=30)

    print("Loading Model Data (extended period)...")
    model_dfs_15_ex, model_onsets_15_ex = get_model_dfs(model_paths, year_ranges=extended_years, config=config, days=15)
    model_dfs_30_ex, model_onsets_30_ex = get_model_dfs(model_paths, year_ranges=extended_years, config=config, days=30)

    print("Loading Climatoligcal Data")
    clim_data = get_climatological_dfs()
    clim_df_15, clim_onset_15 = clim_data["15_day"]
    clim_df_15_ex, clim_onset_15_ex = clim_data["15_day_ex"]
    clim_df_30, clim_onset_30 = clim_data["30_day"]
    clim_df_30_ex, clim_onset_30_ex = clim_data["30_day_ex"]

    print("Saving Data...")
    md_15 = get_plot_metrics(model_dfs_15, model_onsets_15, clim_df_15, clim_onset_15, config, 15)
    md_15_ex = get_plot_metrics(model_dfs_15_ex, model_onsets_15_ex, clim_df_15_ex, clim_onset_15_ex, config, 15)
    md_30  = get_plot_metrics(model_dfs_30, model_onsets_30, clim_df_30, clim_onset_30, config, 30)
    md_30_ex = get_plot_metrics(model_dfs_30_ex, model_onsets_30_ex, clim_df_30_ex, clim_onset_30_ex, config, 30)

    save_data(md_15, output_dir, "deterministic_scores_15_day_2019_2024")
    save_data(md_15_ex, output_dir, "deterministic_scores_15_day_1965_1978_2019_2024_with_gencast")
    save_data(md_30, output_dir, "deterministic_scores_30_day_2019_2024")
    save_data(md_30_ex, output_dir, "deterministic_scores_30_day_1965_1978_2019_2024_with_gencast")


