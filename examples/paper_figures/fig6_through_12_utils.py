"""Code for recreating figures 6-12 from paper"""

import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

from examples.paper_figures.plot_config import params
from monsoonbench.metrics.climatology import ClimatologyOnsetMetrics
from monsoonbench.metrics.deterministic import DeterministicOnsetMetrics
from monsoonbench.metrics.probabilistic import ProbabilisticOnsetMetrics
from monsoonbench.spatial.regions import get_india_outline

warnings.filterwarnings("ignore")


plt.rcParams.update(params)

climatology = ClimatologyOnsetMetrics()
p_metrics = ProbabilisticOnsetMetrics()
d_metrics = DeterministicOnsetMetrics()

model_years = {
    "FuXi S2S": [2019, 2020, 2021],
    "IFS": [2019, 2020, 2021, 2022, 2023],
    "Standard": [2019, 2020, 2021, 2022, 2023, 2024],
}

SMALL_SIZE = 6
MEDIUM_SIZE = 7
LARGE_SIZE = 8

brier_levels = np.arange(0, 0.2, 0.02)
rps_levels = np.arange(0, 0.5, 0.05)


def _get_config(parent_dir: str) -> dict:
    """Build file path configuration dictionary for IMD data and shapefiles.

    Parameters
    ----------
    parent_dir : str
        Root directory containing all data subdirectories.

    Returns:
    -------
    dict
        Dictionary with keys 'imd_folder', 'thresh_file', and 'india_shpfile',
        each mapping to the corresponding file path string.
    """
    config = {
        "imd_folder": f"{parent_dir}/imd_rainfall_data/4p0",
        "thresh_file": f"{parent_dir}/imd_onset_threshold/mwset4x4.nc4",
        "india_shpfile": f"{parent_dir}/ind_map_shpfile/india_shapefile.shp",
    }
    return config


def _get_prob_model_paths(parent_dir: str) -> dict:
    """Build file path dictionary for probabilistic forecast models.

    Parameters
    ----------
    parent_dir : str
        Root directory containing all model rainfall subdirectories.

    Returns:
    -------
    dict
        Dictionary mapping model name strings to their rainfall data directory paths.
        Models included: 'FuXi S2S', 'NGCM', 'IFS', 'GenCast'.
    """
    prob_model_paths = {
        "FuXi S2S": f"{parent_dir}/rainfall_4p0/FuXi_S2S",
        "NGCM": f"{parent_dir}/rainfall_4p0/NeuralGCM",
        "IFS": f"{parent_dir}/rainfall_4p0/IFS_S2S",
        "GenCast": f"{parent_dir}/rainfall_4p0/GenCast",
    }
    return prob_model_paths


def _get_det_model_paths(parent_dir: str) -> dict:
    """Build file path dictionary for deterministic forecast models.

    Parameters
    ----------
    parent_dir : str
        Root directory containing all model rainfall subdirectories.

    Returns:
    -------
    dict
        Dictionary mapping model name strings to their rainfall data directory paths.
        Models included: 'AIFS', 'FuXi', 'Graphcast'.
    """
    det_model_paths = {
        "AIFS": f"{parent_dir}/rainfall_4p0/AIFS",
        "FuXi": f"{parent_dir}/rainfall_4p0/FuXi",
        "Graphcast": f"{parent_dir}/rainfall_4p0/GraphCast",
    }
    return det_model_paths


def _get_brier_model_paths(parent_dir: str) -> dict:
    """Build file path dictionary for models used in Brier/RPS score calculations.

    Parameters
    ----------
    parent_dir : str
        Root directory containing all model rainfall subdirectories.

    Returns:
    -------
    dict
        Dictionary mapping model name strings to their rainfall data directory paths.
        Models included: 'FuXi S2S', 'NGCM', 'IFS'.
    """
    brier_model_paths = {
        "FuXi S2S": f"{parent_dir}/rainfall_4p0/FuXi_S2S",
        "NGCM": f"{parent_dir}/rainfall_4p0/NeuralGCM",
        "IFS": f"{parent_dir}/rainfall_4p0/IFS_S2S",
    }
    return brier_model_paths


def load_clim_data_fig_7_through_9(parent_dir: str) -> xr.DataArray:
    """Load and compute climatology baseline spatial metrics for Figures 7–9 (15-day forecast).

    Computes the climatology baseline using a 15-day forecast window with a 3-day
    tolerance and 1-day verification window, then returns a spatial DataArray of
    false alarm rate, miss rate, and MAE metrics.

    Parameters
    ----------
    parent_dir : str
        Root directory containing IMD rainfall data and threshold files.

    Returns:
    -------
    xr.DataArray
        Spatial dataset containing 'false_alarm_rate', 'miss_rate', and 'mean_mae'
        variables indexed by latitude and longitude.
    """
    config = _get_config(parent_dir)

    metrics_df_clim_15, onset_da_clim_15 = (
        climatology.compute_climatology_baseline_multiple_years(
            years=model_years["Standard"],
            imd_folder=config["imd_folder"],
            thres_file=config["thresh_file"],
            tolerance_days=3,
            verification_window=1,
            forecast_days=15,
            max_forecast_day=15,
            mok=True,
            onset_window=5,
            mok_month=6,
            mok_day=2,
        )
    )

    spatial_clim_15_day = climatology.create_spatial_far_mr_mae(
        metrics_df_clim_15, dict.fromkeys(model_years["Standard"], onset_da_clim_15)
    )

    return spatial_clim_15_day


def load_clim_data_fig_10_through_12(parent_dir: str) -> xr.DataArray:
    """Load and compute climatology baseline spatial metrics for Figures 10–12 (30-day forecast).

    Computes the climatology baseline using a 30-day forecast window with a 5-day
    tolerance and 16-day verification window, then returns a spatial DataArray of
    false alarm rate, miss rate, and MAE metrics.

    Parameters
    ----------
    parent_dir : str
        Root directory containing IMD rainfall data and threshold files.

    Returns:
    -------
    xr.DataArray
        Spatial dataset containing 'false_alarm_rate', 'miss_rate', and 'mean_mae'
        variables indexed by latitude and longitude.
    """
    config = _get_config(parent_dir)
    metrics_df_clim_30, onset_da_clim_30 = (
        climatology.compute_climatology_baseline_multiple_years(
            years=model_years["Standard"],
            imd_folder=config["imd_folder"],
            thres_file=config["thresh_file"],
            tolerance_days=5,
            verification_window=16,
            forecast_days=30,
            max_forecast_day=30,
            mok=True,
            onset_window=5,
            mok_month=6,
            mok_day=2,
        )
    )

    spatial_clim_30_day = climatology.create_spatial_far_mr_mae(
        metrics_df_clim_30, dict.fromkeys(model_years["Standard"], onset_da_clim_30)
    )

    return spatial_clim_30_day


def load_model_data_fig_7_through_9(parent_dir: str) -> tuple[dict, dict]:
    """Load probabilistic and deterministic model metrics for Figures 7–9 (15-day forecast).

    Iterates over all probabilistic and deterministic models, computing onset metrics
    with a 15-day forecast window, 3-day tolerance, and 1-day verification window.

    Parameters
    ----------
    parent_dir : str
        Root directory containing model forecast and IMD rainfall data.

    Returns:
    -------
    tuple[dict, dict]
        - model_dfs_15 : dict mapping model name to metrics DataFrame.
        - model_onsets_15 : dict mapping model name to onset DataArray dict.

    Notes:
    -----
    Due to the early ``return`` inside the deterministic loop, only the first
    deterministic model's results are included alongside all probabilistic models.
    """
    config = _get_config(parent_dir)
    prob_model_paths = _get_prob_model_paths(parent_dir)
    det_model_paths = _get_det_model_paths(parent_dir)

    model_dfs_15 = {}
    model_onsets_15 = {}

    for model_name, model_fp in prob_model_paths.items():
        print("=" * 80)
        print(f"Loading data from {model_name}")
        print("=" * 80)
        probabilistic_df_15, onset_da_dict_15 = (
            p_metrics.compute_metrics_multiple_years(
                years=(
                    model_years[model_name]
                    if model_name in model_years.keys()
                    else model_years["Standard"]
                ),
                imd_folder=config["imd_folder"],
                thres_file=config["thresh_file"],
                model_forecast_dir=model_fp,
                tolerance_days=3,
                verification_window=1,
                forecast_days=15,
                max_forecast_day=15,
                mok=True,
                onset_window=5,
                mok_month=6,
                mok_day=2,
            )
        )

        model_dfs_15[model_name] = probabilistic_df_15
        model_onsets_15[model_name] = onset_da_dict_15

    for model_name, model_fp in det_model_paths.items():
        print("=" * 80)
        print(f"Loading data from {model_name}")
        print("=" * 80)
        deterministic_df_15, onset_da_dict_15 = (
            d_metrics.compute_metrics_multiple_years(
                years=(
                    model_years[model_name]
                    if model_name in model_years.keys()
                    else model_years["Standard"]
                ),
                imd_folder=config["imd_folder"],
                thres_file=config["thresh_file"],
                model_forecast_dir=model_fp,
                tolerance_days=3,
                verification_window=1,
                forecast_days=15,
                max_forecast_day=15,
                mok=True,
                onset_window=5,
                mok_month=6,
                mok_day=2,
            )
        )

        model_dfs_15[model_name] = deterministic_df_15
        model_onsets_15[model_name] = onset_da_dict_15

    return model_dfs_15, model_onsets_15


def load_model_data_fig_10_through_12(parent_dir: str) -> tuple[dict, dict]:
    """Load probabilistic and deterministic model metrics for Figures 10–12 (30-day forecast).

    Iterates over all probabilistic and deterministic models, computing onset metrics
    with a 30-day forecast window, 5-day tolerance, and 16-day verification window.

    Parameters
    ----------
    parent_dir : str
        Root directory containing model forecast and IMD rainfall data.

    Returns:
    -------
    tuple[dict, dict]
        - model_dfs_30 : dict mapping model name to metrics DataFrame.
        - model_onsets_30 : dict mapping model name to onset DataArray dict.

    Notes:
    -----
    Due to the early ``return`` inside the deterministic loop, only the first
    deterministic model's results are included alongside all probabilistic models.
    """
    config = _get_config(parent_dir)
    prob_model_paths = _get_prob_model_paths(parent_dir)
    det_model_paths = _get_det_model_paths(parent_dir)

    model_dfs_30 = {}
    model_onsets_30 = {}

    for model_name, model_fp in prob_model_paths.items():
        print("=" * 80)
        print(f"Loading data from {model_name}")
        print("=" * 80)
        probabilistic_df_30, onset_da_dict_30 = (
            p_metrics.compute_metrics_multiple_years(
                years=(
                    model_years[model_name]
                    if model_name in model_years.keys()
                    else model_years["Standard"]
                ),
                imd_folder=config["imd_folder"],
                thres_file=config["thresh_file"],
                model_forecast_dir=model_fp,
                tolerance_days=5,
                verification_window=16,
                forecast_days=30,
                max_forecast_day=30,
                mok=True,
                onset_window=5,
                mok_month=6,
                mok_day=2,
            )
        )
        model_dfs_30[model_name] = probabilistic_df_30
        model_onsets_30[model_name] = onset_da_dict_30

    for model_name, model_fp in det_model_paths.items():
        print("=" * 80)
        print(f"Loading data from {model_name}")
        print("=" * 80)
        deterministic_df_30, onset_da_dict_30 = (
            d_metrics.compute_metrics_multiple_years(
                years=(
                    model_years[model_name]
                    if model_name in model_years.keys()
                    else model_years["Standard"]
                ),
                imd_folder=config["imd_folder"],
                thres_file=config["thresh_file"],
                model_forecast_dir=model_fp,
                tolerance_days=5,
                verification_window=16,
                forecast_days=30,
                max_forecast_day=30,
                mok=True,
                onset_window=5,
                mok_month=6,
                mok_day=2,
            )
        )

        model_dfs_30[model_name] = deterministic_df_30
        model_onsets_30[model_name] = onset_da_dict_30

    return model_dfs_30, model_onsets_30


def format_data_for_spatial_fig(
    parent_dir: str, tuple_of_model_data_dicts: tuple, clim_data: xr.DataArray
) -> dict:
    """Format and merge model and climatology metrics into a single ordered dict for plotting.

    Computes spatial FAR/MR/MAE metrics for each probabilistic and deterministic model,
    scales rates to percentages, appends the climatology baseline, and reorders all
    entries to match the paper's figure layout.

    Parameters
    ----------
    parent_dir : str
        Root directory used to resolve model path configurations.
    tuple_of_model_data_dicts : tuple
        Two-element tuple of (model_dfs, model_onsets) as returned by
        ``load_model_data_fig_7_through_9`` or ``load_model_data_fig_10_through_12``.
    clim_data : xr.DataArray
        Climatology spatial metrics DataArray as returned by the corresponding
        ``load_clim_data_*`` function.

    Returns:
    -------
    dict
        Ordered dictionary mapping model name to spatial metrics xr.Dataset,
        with keys in paper figure order:
        ['Climatology', 'IFS', 'AIFS', 'FuXi', 'Graphcast', 'GenCast', 'FuXi S2S', 'NGCM'].
    """

    def reorder_dict(dict) -> dict:
        """Reorders dict to match paper format."""
        order = [
            "Climatology",
            "IFS",
            "AIFS",
            "FuXi",
            "Graphcast",
            "GenCast",
            "FuXi S2S",
            "NGCM",
        ]
        reordered_dict = {key: dict[key] for key in order}
        return reordered_dict

    prob_model_paths = _get_prob_model_paths(parent_dir)
    det_model_paths = _get_det_model_paths(parent_dir)
    prob_plot_data = {}

    model_dfs = tuple_of_model_data_dicts[0]
    model_onsets = tuple_of_model_data_dicts[1]

    for model_name in prob_model_paths.keys():
        probabilistic_df = model_dfs[model_name]
        onset_da_dict = model_onsets[model_name]
        plot_probabilistic_metrics = p_metrics.create_spatial_far_mr_mae(
            probabilistic_df, onset_da_dict
        )
        plot_probabilistic_metrics["false_alarm_rate"] = (
            plot_probabilistic_metrics["false_alarm_rate"].round(3) * 100
        )
        plot_probabilistic_metrics["miss_rate"] = (
            plot_probabilistic_metrics["miss_rate"].round(3) * 100
        )
        prob_plot_data[model_name] = plot_probabilistic_metrics

    for model_name in det_model_paths.keys():
        deterministic_df = model_dfs[model_name]
        onset_da_dict = model_onsets[model_name]
        plot_deterministic_metrics = d_metrics.create_spatial_far_mr_mae(
            deterministic_df, onset_da_dict
        )
        plot_deterministic_metrics["false_alarm_rate"] = (
            plot_deterministic_metrics["false_alarm_rate"].round(3) * 100
        )
        plot_deterministic_metrics["miss_rate"] = (
            plot_deterministic_metrics["miss_rate"].round(3) * 100
        )

        prob_plot_data[model_name] = plot_deterministic_metrics

    clim_data["false_alarm_rate"] = clim_data["false_alarm_rate"].round(3) * 100
    clim_data["miss_rate"] = clim_data["miss_rate"].round(3) * 100

    prob_plot_data["Climatology"] = clim_data

    prob_plot_data = reorder_dict(prob_plot_data)

    return prob_plot_data


def calculate_gridwise_brier(forecast_obs_df: pd.DataFrame) -> xr.Dataset:
    """Calculate Brier Score and Fair Brier Score for each grid point (lat, lon).

    Iterates over all unique (lat, lon) pairs in the input DataFrame and computes
    the Brier Score and Fair Brier Score at each location using
    ``ProbabilisticOnsetMetrics.calculate_brier_score``.

    Parameters
    ----------
    forecast_obs_df : pd.DataFrame
        Output from ``create_forecast_observation_pairs_with_bins()``.
        Must contain columns: 'lat', 'lon', 'predicted_prob', 'observed_onset',
        'total_members'.

    Returns:
    -------
    xr.Dataset
        Dataset with dimensions (lat, lon) and data variables 'brier_score'
        and 'fair_brier_score'.
    """
    results_list = []
    unique_locs = forecast_obs_df[["lat", "lon"]].drop_duplicates()
    for _idx, row in unique_locs.iterrows():
        lat = row["lat"]
        lon = row["lon"]

        grid_data = forecast_obs_df[
            (forecast_obs_df["lat"] == lat) & (forecast_obs_df["lon"] == lon)
        ]

        loop_res = p_metrics.calculate_brier_score(grid_data)

        loop_dict = {
            "lat": lat,
            "lon": lon,
            "brier_score": loop_res["brier_score"],
            "fair_brier_score": loop_res["fair_brier_score"],
        }
        results_list.append(loop_dict)

    results_df = pd.DataFrame(results_list).set_index(["lat", "lon"])
    results_ds = results_df.to_xarray()

    return results_ds


def calculate_gridwise_rps(forecast_obs_df: pd.DataFrame) -> xr.Dataset:
    """Calculate RPS and Fair RPS for each grid point (lat, lon).

    Iterates over all unique (lat, lon) pairs in the input DataFrame and computes
    the Ranked Probability Score and Fair RPS at each location using
    ``ProbabilisticOnsetMetrics.calculate_rps``.

    Parameters
    ----------
    forecast_obs_df : pd.DataFrame
        Output from ``create_forecast_observation_pairs_with_bins()``.
        Must contain columns: 'lat', 'lon', 'predicted_prob', 'observed_onset',
        'total_members'.

    Returns:
    -------
    xr.Dataset
        Dataset with dimensions (lat, lon) and data variables 'rps' and 'fair_rps'.
    """
    results_list = []
    unique_locs = forecast_obs_df[["lat", "lon"]].drop_duplicates()
    for _idx, row in unique_locs.iterrows():
        lat = row["lat"]
        lon = row["lon"]

        grid_data = forecast_obs_df[
            (forecast_obs_df["lat"] == lat) & (forecast_obs_df["lon"] == lon)
        ]

        loop_res = p_metrics.calculate_rps(grid_data)

        loop_dict = {
            "lat": lat,
            "lon": lon,
            "rps": loop_res["rps"],
            "fair_rps": loop_res["fair_rps"],
        }
        results_list.append(loop_dict)

    results_df = pd.DataFrame(results_list).set_index(["lat", "lon"])
    results_ds = results_df.to_xarray()

    return results_ds


def load_data_for_fig_6(parent_dir: str) -> list:
    """Load and compute gridwise Brier and RPS scores for Figure 6.

    For each model in the Brier model paths, loads multi-year forecast–observation
    pairs for both 15-day and 30-day windows, then computes gridwise Brier scores
    and RPS scores for each.

    Parameters
    ----------
    parent_dir : str
        Root directory containing model forecast and IMD rainfall data.

    Returns:
    -------
    list
        Four-element list of dicts: [brier_15_ds, brier_30_ds, rps_15_ds, rps_30_ds].
        Each dict maps model name to an xr.Dataset with gridwise scores.
    """
    config = _get_config(parent_dir)
    brier_model_paths = _get_brier_model_paths(parent_dir)

    forecast_dfs_15 = {}
    forecast_dfs_30 = {}
    for model_name, model_fp in brier_model_paths.items():
        print("=" * 80)
        print(f"Loading data from {model_name}")
        print("=" * 80)
        multi_year_df = p_metrics.multi_year_forecast_obs_pairs(
            range(2004, 2022),
            model_forecast_dir=model_fp,
            imd_folder=config["imd_folder"],
            thres_file=config["thresh_file"],
            mem_num=51 if model_name != "IFS" else 11,
            max_forecast_day=15,
            day_bins=[(1, 5), (6, 10), (11, 15)],
            date_filter_year=2024 if model_name != "IFS" else 2022,
        )
        forecast_dfs_15[model_name] = multi_year_df

        multi_year_df = p_metrics.multi_year_forecast_obs_pairs(
            range(2004, 2022),
            model_forecast_dir=model_fp,
            imd_folder=config["imd_folder"],
            thres_file=config["thresh_file"],
            mem_num=51 if model_name != "IFS" else 11,
            max_forecast_day=30,
            day_bins=[(1, 10), (11, 20), (21, 30)],
            date_filter_year=2024 if model_name != "IFS" else 2022,
        )
        forecast_dfs_30[model_name] = multi_year_df

    brier_15_ds = {}
    brier_30_ds = {}
    rps_15_ds = {}
    rps_30_ds = {}

    for model, df in forecast_dfs_15.items():
        loop_grid_brier = calculate_gridwise_brier(df)
        brier_15_ds[model] = loop_grid_brier

    for model, df in forecast_dfs_30.items():
        loop_grid_brier = calculate_gridwise_brier(df)
        brier_30_ds[model] = loop_grid_brier

    for model, df in forecast_dfs_15.items():
        loop_grid_rps = calculate_gridwise_rps(df)
        rps_15_ds[model] = loop_grid_rps

    for model, df in forecast_dfs_30.items():
        loop_grid_rps = calculate_gridwise_rps(df)
        rps_30_ds[model] = loop_grid_rps

    brier_rps_graph_data_list = [brier_15_ds, brier_30_ds, rps_15_ds, rps_30_ds]

    return brier_rps_graph_data_list


def cmz_text_formatter(val: float, metric: str) -> str:
    """Format a Core Monsoon Zone average value as a labeled string for map annotation.

    Parameters
    ----------
    val : float
        The numeric value to display.
    metric : str
        The metric name. Supported values (matched by substring):
        'mae' -> 'MAE: X.X', 'false_alarm_rate' -> 'FAR: X.X',
        'miss_rate' -> 'MR: X.X', 'brier' -> 'Brier: X.XX',
        'rps' -> 'RPS: X.XX'.

    Returns:
    -------
    str
        Formatted annotation string with metric label and value.
    """
    if "mae" in metric:
        metric_text = f"MAE: {val:.1f}"
    elif metric == "false_alarm_rate":
        metric_text = f"FAR: {val:.1f}"
    elif metric == "miss_rate":
        metric_text = f"MR: {val:.1f}"
    elif "brier" in metric:
        metric_text = f"Brier: {val:.2f}"
    elif "rps" in metric:
        metric_text = f"RPS: {val:.2f}"
    return metric_text


def create_discrete_colormap(
    levels: np.ndarray, base_cmap: str = "RdBu"
) -> tuple[colors.Colormap, colors.BoundaryNorm]:
    """Create a discrete colormap and boundary normalization for a given set of levels.

    Parameters
    ----------
    levels : np.ndarray
        Array of boundary values defining the discrete color intervals.
    base_cmap : str, optional
        Name of the matplotlib colormap to discretize. Default is 'RdBu'.

    Returns:
    -------
    tuple[colors.Colormap, colors.BoundaryNorm]
        - cmap : Discretized colormap.
        - norm : BoundaryNorm instance mapping data values to colormap indices.
    """
    cmap = plt.cm.get_cmap(base_cmap)
    norm = colors.BoundaryNorm(levels, cmap.N, clip=True)
    return cmap, norm


def create_map_instance_for_model(
    ax,
    data_array: xr.DataArray,
    model: str,
    metric: str,
    vmin: float = -100,
    vmax: float = 100,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    title: str = None,
    levels: np.ndarray = None,
    cmap: str = "YlOrRd",
    parent_dir: str | None = None,
) -> list | None:
    """Render a single spatial skill map panel onto a Cartopy axes instance.

    Plots a pcolormesh of the provided DataArray over India, overlays India state
    boundaries and the Core Monsoon Zone (CMZ) polygon, and annotates the panel
    with the model name and the CMZ-averaged metric value.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axes on which to draw the map (must use PlateCarree projection).
    data_array : xr.DataArray
        2D DataArray with 'lat' and 'lon' coordinates containing the metric values.
    model : str
        Model label string to display in the top-right corner of the panel.
    metric : str
        Metric identifier used by ``cmz_text_formatter`` to format the CMZ annotation.
    vmin : float, optional
        Minimum color scale value (ignored when ``levels`` is provided). Default -100.
    vmax : float, optional
        Maximum color scale value (ignored when ``levels`` is provided). Default 100.
    show_ylabel : bool, optional
        Whether to display latitude tick labels. Default True.
    show_xlabel : bool, optional
        Whether to display longitude tick labels. Default True.
    title : str, optional
        Panel title text placed above the top-left corner. Default None.
    levels : np.ndarray, optional
        Discrete color boundary levels. If provided, a BoundaryNorm is used and
        vmin/vmax are ignored.
    cmap : str, optional
        Matplotlib colormap name. Default 'YlOrRd'.
    parent_dir : str or None, optional
        Root data directory; required to load India shapefile boundaries.
        If None, falls back to cartopy coastlines on failure.

    Returns:
    -------
    tuple or None
        (im, levels) where im is the QuadMesh object and levels is the boundary
        array used for colorbar creation, or None if data_array contains only NaN.
    """
    polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
    polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])

    panel_linewidth = 0.5
    map_lw = 0.5
    polygon_lw = 1.3
    tick_length = 2
    tick_width = 0.5

    if parent_dir:
        config = _get_config(parent_dir)

    if data_array.isnull().all():  # noqa: PD003
        ax.text(
            0.5,
            0.5,
            f"No data for {model}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return None

    lats = data_array.lat.values
    lons = data_array.lon.values

    lat_edges = np.concatenate([lats - 2, [lats[-1] + 2]])
    lon_edges = np.concatenate([lons - 2, [lons[-1] + 2]])

    if levels is not None:
        cmap, norm = create_discrete_colormap(levels, cmap)
        vmin, vmax = None, None
    else:
        cmap = cmap
        norm = None
        vmin, vmax = vmin, vmax

    im = ax.pcolormesh(
        lon_edges,
        lat_edges,
        data_array.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )

    try:
        india_boundaries = get_india_outline(shp_file_path=config["india_shpfile"])
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            ax.plot(
                india_lon,
                india_lat,
                color="black",
                linewidth=map_lw,
                transform=ccrs.PlateCarree(),
            )
    except Exception as e:
        print(f"Warning: Could not load India boundaries: {e}")
        ax.add_feature(cfeature.COASTLINE, linewidth=map_lw, color="black")

    polygon = Polygon(
        list(zip(polygon1_lon, polygon1_lat)),
        fill=False,
        edgecolor="black",
        linewidth=polygon_lw,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(polygon)

    from matplotlib.path import Path

    polygon_path = Path(list(zip(polygon1_lon, polygon1_lat)))

    values_in_polygon = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if polygon_path.contains_point((lon, lat)):
                value = data_array.values[i, j]
                if not np.isnan(value):
                    values_in_polygon.append(value)

    if values_in_polygon:
        avg_value = np.mean(values_in_polygon)
        cmz_text = cmz_text_formatter(val=avg_value, metric=metric)
        ax.text(
            0.95,
            0.05,
            cmz_text,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            color="black",
            fontsize=MEDIUM_SIZE,
            fontweight="normal",
        )

    ax.text(
        0.95,
        0.95,
        model,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        color="black",
        fontsize=MEDIUM_SIZE,
        fontweight="normal",
    )

    ax.set_xlim([lons[0] - 4, 100])
    ax.set_ylim([lats[0] - 4, lats[-1] + 4])

    yticks = np.arange(lats[0] - 2, lats[-1] + 3, 8)
    yticklabels = [f"{int(y)}°N" if i % 1 == 0 else "" for i, y in enumerate(yticks)]
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    if show_ylabel:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels([])

    xticks = np.arange(lons[0] - 2, lons[-1] + 3, 8)
    xticklabels = [f"{int(x)}°E" if i % 1 == 0 else "" for i, x in enumerate(xticks)]
    ax.set_xticks(xticks)
    if show_xlabel:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels([])

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=SMALL_SIZE,
        length=tick_length,
        width=tick_width,
    )
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(panel_linewidth)

    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    if title:
        ax.text(
            0.02,
            1.02,
            title,
            transform=ax.transAxes,
            verticalalignment="bottom",
            fontsize=LARGE_SIZE,
            fontweight="normal",
        )

    return im, levels


def create_spatial_metric_figure_xr(
    plot_data: dict[str, xr.Dataset],
    metric: str,
    vmin: float = None,
    vmax: float = None,
    figure_title: str = None,
    cmap: str = "YlOrRd",
    n_colors: int = 10,
    parent_dir: str | None = None,
) -> plt.Figure:
    """Create a multi-panel spatial metric figure from a dict of xr.Datasets.

    Arranges one map panel per model in a 2-column grid, with a shared vertical
    colorbar. Color scale bounds are inferred from data if not provided.

    Parameters
    ----------
    plot_data : dict[str, xr.Dataset]
        Ordered dict mapping model name to xr.Dataset containing the target metric
        as a variable (e.g., 'false_alarm_rate', 'miss_rate', 'mean_mae').
    metric : str
        Variable name to extract from each dataset and plot.
    vmin : float, optional
        Minimum colorbar value. Inferred from data if None.
    vmax : float, optional
        Maximum colorbar value. Inferred from data if None.
    figure_title : str, optional
        Overall figure suptitle. Default None.
    cmap : str, optional
        Matplotlib colormap name. Default 'YlOrRd'.
    n_colors : int, optional
        Number of discrete color levels between vmin and vmax. Default 10.
    parent_dir : str or None, optional
        Root data directory passed to ``create_map_instance_for_model`` for
        loading India shapefile boundaries.

    Returns:
    -------
    plt.Figure
        Completed matplotlib Figure with all panels and colorbar.
    """
    color_bar_map = {
        "mean_mae": "MAE (days)",
        "false_alarm_rate": "False alarm rate (%)",
        "miss_rate": "Miss rate (%)",
    }
    plot_data_arrays = {model: data[metric] for model, data in plot_data.items()}

    all_values = np.concatenate(
        [da.values.flatten() for da in plot_data_arrays.values()]
    )

    if vmin is None:
        vmin = np.nanmin(all_values)

    if vmax is None:
        vmax = np.nanmax(all_values)

    levels = np.linspace(vmin, vmax, n_colors + 1)

    n_models = len(plot_data_arrays)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))

    fig = plt.figure(figsize=(6, 9), dpi=300)

    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        hspace=0.05,
        wspace=-0.2,
        left=0.05,
        right=0.85,
        top=0.95,
        bottom=0.05,
        width_ratios=[1, 1, 0.08],
    )

    axes = np.empty((n_rows, n_cols), dtype=object)

    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())

    im_for_colorbar = None
    levels_for_colorbar = None

    for i, (model, model_data) in enumerate(plot_data_arrays.items()):
        r, c = divmod(i, n_cols)

        show_ylabel = c == 0
        show_xlabel = r == n_rows - 1

        result = create_map_instance_for_model(
            axes[r, c],
            model_data,
            model,
            metric=metric,
            vmin=vmin,
            vmax=vmax,
            show_ylabel=show_ylabel,
            show_xlabel=show_xlabel,
            cmap=cmap,
            levels=levels,
            parent_dir=parent_dir,
        )

        axes[r, c].set_aspect("equal", adjustable="box")

        if result is not None and im_for_colorbar is None:
            im_for_colorbar, levels_for_colorbar = result

    for idx in range(n_models, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    if im_for_colorbar is not None:
        active_axes = [
            axes[r, c]
            for r in range(n_rows)
            for c in range(n_cols)
            if axes[r, c].axison
        ]

        positions = [ax.get_position() for ax in active_axes]

        top = max(p.y1 for p in positions)
        bottom = min(p.y0 for p in positions)

        single_panel_height = positions[0].y1 - positions[0].y0
        two_panel_height = single_panel_height * 2

        grid_center = (top + bottom) / 2
        cbar_y = grid_center - two_panel_height / 2

        cax = fig.add_axes([0.87, cbar_y, 0.025, two_panel_height])

        cbar = fig.colorbar(
            im_for_colorbar,
            cax=cax,
            orientation="vertical",
            extend=None if metric == "miss_rate" else "max",
            spacing="uniform",
            boundaries=levels_for_colorbar,
            ticks=levels_for_colorbar,
        )

        cbar.set_label(color_bar_map[metric], usetex=False)
        cbar.set_ticks(np.arange(vmin, vmax + 1, (vmin + vmax) / 5))
        cbar.ax.minorticks_off()
        cbar.ax.tick_params(length=2, width=1)

    if figure_title:
        fig.suptitle(figure_title)

    return fig


def create_brier_rps_maps_figure_from_datasets(
    datasets_list: list[dict], parent_dir: str = None
) -> plt.Figure:
    """Create a 4-row multi-panel figure of Brier and RPS skill score maps (Figure 6).

    Renders spatial maps for Brier Score (15-day, 30-day) and Ranked Probability
    Score (15-day, 30-day), one model per column, with shared colorbars for each
    pair of rows. Row titles follow the paper's labeling convention.

    Parameters
    ----------
    datasets_list : list[dict]
        List of four dicts, one per row, each mapping model name to an xr.Dataset
        containing the relevant score variable ('brier_score' or 'rps').
        Expected order:
        [brier_15_dict, brier_30_dict, rps_15_dict, rps_30_dict].

    Returns:
    -------
    plt.Figure
        Completed matplotlib Figure with all map panels and colorbars.
    """
    titles = {
        0: "(a) Brier Skill Score: 15-day forecast",
        1: "(b) Brier Skill Score: 30-day forecast",
        2: "(c) Ranked Probability Skill Score: 15-day forecast",
        3: "(d) Ranked Probability Skill Score: 30-day forecast",
    }

    row_configs = [("brier_score", 15), ("brier_score", 30), ("rps", 15), ("rps", 30)]

    n_rows = len(datasets_list)
    n_models = len(list(datasets_list[0].keys()))
    n_cols = n_models + 1

    fig = plt.figure(figsize=(6, 4 * n_rows / 2), dpi=300)

    width_ratios = [1] * n_models + [0.02]

    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        hspace=0.15,
        wspace=0.3,
        left=0.05,
        right=0.85,
        top=0.95,
        bottom=0.05,
        width_ratios=width_ratios,
    )

    axes = []
    data_arrays = []
    colorbars_data = []

    for row_idx, tuple_ in enumerate(row_configs):
        metric = tuple_[0]

        row_axes = []
        row_data = []

        if "brier" in metric:
            levels = brier_levels
        else:
            levels = rps_levels

        plot_data = datasets_list[row_idx]

        im = None
        for model_idx, (model, data_array) in enumerate(plot_data.items()):
            ax = fig.add_subplot(gs[row_idx, model_idx], projection=ccrs.PlateCarree())
            row_axes.append(ax)
            row_data.append(data_array)

            show_ylabel = model_idx == 0
            show_xlabel = row_idx == n_rows - 1

            title = titles.get(row_idx) if (titles and model_idx == 0) else None

            result = create_map_instance_for_model(
                ax,
                data_array[metric],
                model,
                metric,
                vmin=None,
                vmax=None,
                show_ylabel=show_ylabel,
                show_xlabel=show_xlabel,
                cmap="RdBu",
                levels=levels,
                title=title,
                parent_dir=parent_dir,
            )

            if model_idx == 0 and result is not None:
                if isinstance(result, tuple):
                    im, used_levels = result
                else:
                    im = result
                    used_levels = levels
                colorbars_data.append((row_idx, im, metric, used_levels))

        axes.append(row_axes)
        data_arrays.append(row_data)

    if len(colorbars_data) >= 2:
        brier_data = [item for item in colorbars_data if "brier" in item[2]]
        if brier_data:
            _, im, _, levels = brier_data[0]

            pos_row0 = axes[0][0].get_position()
            pos_row1 = axes[1][0].get_position()

            colorbar_height = (pos_row0.y1 - pos_row1.y0) * 0.8
            center_y = (pos_row0.y1 + pos_row1.y0) / 2
            colorbar_bottom = center_y - colorbar_height / 2

            cax_brier = fig.add_axes([0.93, colorbar_bottom, 0.025, colorbar_height])
            cbar_brier = fig.colorbar(
                im, cax=cax_brier, orientation="vertical", extend="both"
            )
            cbar_brier.set_ticks(levels[::2])
            cbar_brier.set_label(
                "Brier Score", fontsize=MEDIUM_SIZE, rotation=270, labelpad=15
            )
            cbar_brier.ax.tick_params(labelsize=SMALL_SIZE, length=2, width=1)

    if len(colorbars_data) >= 4:
        rps_data = [item for item in colorbars_data if "rps" in item[2]]
        if rps_data:
            _, im, _, levels = rps_data[0]

            pos_row2 = axes[2][0].get_position()
            pos_row3 = axes[3][0].get_position()

            colorbar_height = (pos_row2.y1 - pos_row3.y0) * 0.8
            center_y = (pos_row2.y1 + pos_row3.y0) / 2
            colorbar_bottom = center_y - colorbar_height / 2

            cax_rps = fig.add_axes([0.93, colorbar_bottom, 0.025, colorbar_height])
            cbar_rps = fig.colorbar(
                im, cax=cax_rps, orientation="vertical", extend="both"
            )
            cbar_rps.set_ticks(levels[::2])
            cbar_rps.set_label("RPS", fontsize=MEDIUM_SIZE, rotation=270, labelpad=15)
            cbar_rps.ax.tick_params(labelsize=SMALL_SIZE, length=2, width=1)

    return fig


def create_fig_6(parent_dir: str) -> plt.Figure:
    """Generate Figure 6: gridwise Brier and RPS skill score spatial maps.

    Loads forecast–observation pairs for all Brier model paths, computes gridwise
    Brier and RPS scores for 15-day and 30-day windows, and renders the 4-row
    multi-panel figure.

    Parameters
    ----------
    parent_dir : str
        Root directory containing model forecast and IMD rainfall data.

    Returns:
    -------
    plt.Figure
        Completed Figure 6 with Brier and RPS spatial maps.
    """
    fig_data = load_data_for_fig_6(parent_dir)
    fig = create_brier_rps_maps_figure_from_datasets(fig_data, parent_dir=parent_dir)
    return fig


def create_figs_7_8_9(parent_dir: str) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Generate Figures 7, 8, and 9: spatial FAR, MAE, and MR maps for 15-day forecasts.

    Loads climatology and model data for the 15-day forecast window and produces
    three spatial metric figures: False Alarm Rate (Blues colormap), Mean Absolute
    Error (YlOrRd colormap), and Miss Rate (Blues colormap).

    Parameters
    ----------
    parent_dir : str
        Root directory containing model forecast and IMD rainfall data.

    Returns:
    -------
    tuple[plt.Figure, plt.Figure, plt.Figure]
        - fig1 : False Alarm Rate figure (Figure 7).
        - fig2 : MAE figure (Figure 8).
        - fig3 : Miss Rate figure (Figure 9).
    """
    title1 = "False Alarm Rate for Models - 15 Day"
    title2 = "MAE for Ensemble Models - 15 Day"
    title3 = "MR for Ensemble Models - 15 Day"

    clim_data = load_clim_data_fig_7_through_9(parent_dir)
    model_data = load_model_data_fig_7_through_9(parent_dir)

    plot_data = format_data_for_spatial_fig(parent_dir, model_data, clim_data)

    fig1 = create_spatial_metric_figure_xr(
        plot_data=plot_data,
        metric="false_alarm_rate",
        figure_title=title1,
        cmap="Blues",
        vmin=0,
        vmax=60,
        n_colors=10,
        parent_dir=parent_dir,
    )
    fig2 = create_spatial_metric_figure_xr(
        plot_data=plot_data,
        metric="mean_mae",
        figure_title=title2,
        cmap="YlOrRd",
        vmin=0,
        vmax=15,
        n_colors=10,
        parent_dir=parent_dir,
    )

    fig3 = create_spatial_metric_figure_xr(
        plot_data=plot_data,
        metric="miss_rate",
        figure_title=title3,
        cmap="Blues",
        vmin=0,
        vmax=100,
        n_colors=10,
        parent_dir=parent_dir,
    )

    return fig1, fig2, fig3


def create_figs_10_11_12(parent_dir: str) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Generate Figures 10, 11, and 12: spatial FAR, MAE, and MR maps for 30-day forecasts.

    Loads climatology and model data for the 30-day forecast window and produces
    three spatial metric figures: False Alarm Rate (Blues colormap), Mean Absolute
    Error (YlOrRd colormap), and Miss Rate (Blues colormap).

    Parameters
    ----------
    parent_dir : str
        Root directory containing model forecast and IMD rainfall data.

    Returns:
    -------
    tuple[plt.Figure, plt.Figure, plt.Figure]
        - fig4 : False Alarm Rate figure (Figure 10).
        - fig5 : MAE figure (Figure 11).
        - fig6 : Miss Rate figure (Figure 12).
    """
    title4 = "False Alarm Rate for Models - 30 Day"
    title5 = "MAE for Ensemble Models - 30 Day"
    title6 = "MR for Ensemble Models - 30 Day"

    clim_data = load_clim_data_fig_10_through_12(parent_dir)
    model_data = load_model_data_fig_10_through_12(parent_dir)

    plot_data = format_data_for_spatial_fig(parent_dir, model_data, clim_data)

    fig4 = create_spatial_metric_figure_xr(
        plot_data=plot_data,
        metric="false_alarm_rate",
        figure_title=title4,
        cmap="Blues",
        vmin=0,
        vmax=60,
        n_colors=10,
        parent_dir=parent_dir,
    )
    fig5 = create_spatial_metric_figure_xr(
        plot_data=plot_data,
        metric="mean_mae",
        figure_title=title5,
        cmap="YlOrRd",
        vmin=0,
        vmax=15,
        n_colors=10,
        parent_dir=parent_dir,
    )

    fig6 = create_spatial_metric_figure_xr(
        plot_data=plot_data,
        metric="miss_rate",
        figure_title=title6,
        cmap="Blues",
        vmin=0,
        vmax=100,
        n_colors=10,
        parent_dir=parent_dir,
    )

    return fig4, fig5, fig6
