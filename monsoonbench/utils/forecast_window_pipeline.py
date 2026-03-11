"""Forecast-window pipeline utilities for CMZ model-vs-climatology analysis.

This module contains reusable data-processing logic extracted from notebook
workflows. It supports:

1. scanning available model-year coverage
2. computing per-model delta tables (model - climatology)
3. running batch evaluation across models
4. exporting CSV artifacts for downstream plotting and review
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from monsoonbench.metrics.base import OnsetMetricsBase
from monsoonbench.metrics.climatology import ClimatologyOnsetMetrics
from monsoonbench.metrics.cmz_window_metrics import (
    WindowScoringConfig,
    compute_cmz_window_metrics,
)
from monsoonbench.metrics.deterministic import DeterministicOnsetMetrics
from monsoonbench.metrics.probabilistic import ProbabilisticOnsetMetrics
from monsoonbench.spatial.regions import get_cmz_polygon_coords, points_inside_polygon

__all__ = [
    "DEFAULT_WINDOW_BINS",
    "DEFAULT_WINDOW_TOLERANCE",
    "MAT_MODEL_ORDER",
    "MODEL_TO_MAT_NAME",
    "PAPER_YEARS_BY_MODEL_4DEG",
    "PLOT_MODEL_LABELS",
    "PLOT_MODEL_ORDER",
    "WindowDeltaRunConfig",
    "build_default_base_config",
    "build_default_scoring_config",
    "build_model_run_config",
    "coverage_preview_columns",
    "choose_recommended_model",
    "compute_window_delta_table",
    "get_default_four_degree_model_specs",
    "run_and_save_multi_model_analysis",
    "run_multi_model_window_analysis",
    "run_single_model_window_analysis",
    "save_window_analysis_outputs",
    "scan_forecast_data_coverage",
]

DEFAULT_WINDOW_BINS: list[tuple[int, int]] = [
    (1, 5),
    (6, 10),
    (11, 15),
    (16, 20),
    (21, 25),
    (26, 30),
]
DEFAULT_WINDOW_TOLERANCE: list[int] = [2, 2, 3, 3, 5, 5]

PAPER_YEARS_BY_MODEL_4DEG: dict[str, list[int]] = {
    "ifs": [2019, 2020, 2021, 2022, 2023],
    "fuxi-s2s": [2019, 2020, 2021],
    "gencast": [2019, 2020, 2021, 2022, 2023, 2024],
    "ngcm": [2019, 2020, 2021, 2022, 2023, 2024],
    "aifs": [2019, 2020, 2021, 2022, 2023, 2024],
    "fuxi": [2019, 2020, 2021, 2022, 2023, 2024],
    "graphcast": [2019, 2020, 2021, 2022, 2023, 2024],
}

MAT_MODEL_ORDER = [
    "clim",
    "ifs",
    "aifs",
    "fuxi",
    "graphcast",
    "gencast",
    "fuxis2s",
    "ngcm51",
]
MODEL_TO_MAT_NAME = {
    "ifs": "ifs",
    "aifs": "aifs",
    "fuxi": "fuxi",
    "graphcast": "graphcast",
    "gencast": "gencast",
    "fuxi-s2s": "fuxis2s",
    "ngcm": "ngcm51",
}
PLOT_MODEL_ORDER = ["ifs", "aifs", "fuxi", "graphcast", "gencast", "fuxi-s2s", "ngcm"]
PLOT_MODEL_LABELS = {
    "ifs": "IFS*",
    "aifs": "AIFS",
    "fuxi": "FuXi",
    "graphcast": "GraphCast",
    "gencast": "GenCast",
    "fuxi-s2s": "FuXi-S2S*",
    "ngcm": "NGCM",
}


@dataclass
class WindowDeltaRunConfig:
    """Configuration for one model run in the forecast-window pipeline."""

    model_name: str
    model_type: str
    model_forecast_dir: Path
    imd_folder: Path
    thres_file: Path
    years: list[int]
    mem_num: int | None = None
    mok: bool = True
    max_forecast_day: int = 30
    date_filter_year: int = 2024
    file_pattern: str = "{}.nc"
    temporal_agg_mode: str = "year_then_average"
    use_legacy_cmz_polygon: bool = True
    strict_year_failures: bool = False
    scoring: WindowScoringConfig = field(default_factory=WindowScoringConfig)


def build_default_base_config(data_dir: Path) -> dict[str, Any]:
    """Build standard 4-degree base configuration from repository data path."""
    return {
        "imd_folder": data_dir / "imd_rainfall_data" / "4p0",
        "thres_file": data_dir / "imd_onset_threshold" / "mwset4x4.nc4",
        "mok": True,
        "max_forecast_day": 30,
        "file_pattern": "{}.nc",
        "temporal_agg_mode": "year_then_average",
        "use_legacy_cmz_polygon": True,
        "strict_year_failures": False,
    }


def build_default_scoring_config() -> WindowScoringConfig:
    """Return the standard strict-window scoring setup used for Figure 3."""
    return WindowScoringConfig(
        obs_window_mode="bin",
        classification_mode="strict_window",
        fn_tn_obs_mode="bin",
        mr_denom_mode="obs_in_bin",
        aggregate_mode="gridmean",
    )


def coverage_preview_columns() -> list[str]:
    """Return a compact column list for displaying model coverage summary."""
    return [
        "model",
        "model_type",
        "supports_paper_years",
        "ensemble_check_ok",
        "n_years",
        "min_year",
        "max_year",
        "missing_paper_years",
    ]


def get_default_four_degree_model_specs(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Return default 4-degree model catalog used in paper-style analysis."""
    forecast_dir = data_dir / "model_forecast_data"
    return {
        "ifs": {
            "model_type": "probabilistic",
            "expected_ens": 11,
            "mem_num": 11,
            "date_filter_year": 2024,
            "path": forecast_dir / "IFS-S2S" / "tp_4p0",
        },
        "aifs": {
            "model_type": "deterministic",
            "expected_ens": None,
            "mem_num": None,
            "date_filter_year": 2024,
            "path": forecast_dir / "aifs" / "daily_0z" / "tp_4p0",
        },
        "fuxi": {
            "model_type": "deterministic",
            "expected_ens": None,
            "mem_num": None,
            "date_filter_year": 2024,
            "path": forecast_dir / "fuxi" / "output_daily_paper_0z_4p0" / "tp_lsm",
        },
        "graphcast": {
            "model_type": "deterministic",
            "expected_ens": None,
            "mem_num": None,
            "date_filter_year": 2024,
            "path": forecast_dir
            / "graphcast37"
            / "output_twice_weekly_paper_0z_4p0"
            / "tp_lsm",
        },
        "gencast": {
            "model_type": "probabilistic",
            "expected_ens": 51,
            "mem_num": 51,
            "date_filter_year": 2024,
            "path": forecast_dir / "gencast52" / "tp_lsm_4p0",
        },
        "fuxi-s2s": {
            "model_type": "probabilistic",
            "expected_ens": 51,
            "mem_num": 51,
            "date_filter_year": 2022,
            "path": forecast_dir / "fuxi_s2s" / "tp_4p0",
        },
        "ngcm": {
            "model_type": "probabilistic",
            "expected_ens": 51,
            "mem_num": 51,
            "date_filter_year": 2024,
            "path": forecast_dir / "ngcm51" / "climatology" / "tp_4p0",
        },
    }


def _legacy_cmz_polygon(resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Return legacy CMZ polygon coordinates used by prior notebook workflows."""
    if abs(resolution - 4.0) < 0.1:
        poly_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
        poly_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
        return poly_lon, poly_lat
    if abs(resolution - 2.0) < 0.1:
        poly_lon = np.array(
            [83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83]
        )
        poly_lat = np.array(
            [17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17]
        )
        return poly_lon, poly_lat
    if abs(resolution - 1.0) < 0.1:
        poly_lon = np.array(
            [
                74,
                85,
                85,
                86,
                86,
                87,
                87,
                88,
                88,
                88,
                85,
                85,
                82,
                82,
                79,
                79,
                78,
                78,
                69,
                69,
                74,
                74,
            ]
        )
        poly_lat = np.array(
            [
                18,
                18,
                19,
                19,
                20,
                20,
                21,
                21,
                21,
                24,
                24,
                25,
                25,
                26,
                26,
                27,
                27,
                28,
                28,
                21,
                21,
                18,
            ]
        )
        return poly_lon, poly_lat
    raise ValueError(f"Unsupported legacy CMZ resolution: {resolution}")


def _resolve_cmz_polygon(
    resolution: float, use_legacy_cmz_polygon: bool
) -> tuple[np.ndarray, np.ndarray]:
    if use_legacy_cmz_polygon:
        return _legacy_cmz_polygon(resolution)

    cmz_coords = get_cmz_polygon_coords(resolution)
    if cmz_coords is None:
        raise ValueError(f"Unsupported CMZ resolution: {resolution}")
    return cmz_coords


def _extract_cmz_pairs(
    thres_da: xr.DataArray,
    use_legacy_cmz_polygon: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, xr.DataArray]:
    """Build CMZ point pairs and aligned threshold slice from threshold grid."""
    if "lat" not in thres_da.coords or "lon" not in thres_da.coords:
        raise ValueError("Threshold data must include 'lat' and 'lon' coordinates.")

    lat_vals = thres_da.lat.values
    lon_vals = thres_da.lon.values
    resolution = float(abs(lat_vals[1] - lat_vals[0]))

    poly_lon, poly_lat = _resolve_cmz_polygon(
        resolution, use_legacy_cmz_polygon=use_legacy_cmz_polygon
    )
    _, inside_lons_raw, inside_lats_raw = points_inside_polygon(
        poly_lon, poly_lat, lon_vals, lat_vals
    )

    cmz_pairs = pd.DataFrame({"lat": inside_lats_raw, "lon": inside_lons_raw})
    cmz_pairs = cmz_pairs.drop_duplicates()
    inside_lons = np.unique(inside_lons_raw)
    inside_lats = np.unique(inside_lats_raw)
    thres_slice = thres_da.sel(lat=inside_lats, lon=inside_lons)
    return cmz_pairs, inside_lats, inside_lons, thres_slice


def _nanmean_dict(items: list[dict[str, float]], key: str) -> float:
    vals = [row[key] for row in items if pd.notna(row[key])]
    return float(np.mean(vals)) if vals else np.nan


def _parse_ensemble_size(nc_file: Path) -> tuple[str | None, int | None]:
    with xr.open_dataset(nc_file) as ds:
        for dim_name in ["member", "number", "sample"]:
            if dim_name in ds.dims:
                return dim_name, int(ds.sizes[dim_name])
    return None, None


def scan_forecast_data_coverage(
    model_specs: dict[str, dict[str, Any]],
    paper_years_by_model: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Scan model folders and summarize year coverage plus ensemble metadata."""
    rows: list[dict[str, Any]] = []
    paper_years_map = paper_years_by_model or PAPER_YEARS_BY_MODEL_4DEG

    for model_name, spec in model_specs.items():
        model_path = Path(spec["path"])
        years: list[int] = []
        ensemble_dim = None
        actual_ens = None

        if model_path.exists():
            files = sorted(model_path.glob("*.nc"))
            years = sorted(int(f.stem) for f in files if f.stem.isdigit())
            if files:
                ensemble_dim, actual_ens = _parse_ensemble_size(files[0])

        expected_ens = spec.get("expected_ens")
        if expected_ens is None:
            ensemble_ok = actual_ens is None
        else:
            ensemble_ok = (actual_ens is not None) and (actual_ens >= expected_ens)

        paper_years = paper_years_map.get(
            model_name, [2019, 2020, 2021, 2022, 2023, 2024]
        )
        missing_paper_years = [y for y in paper_years if y not in years]

        rows.append(
            {
                "model": model_name,
                "model_type": spec.get("model_type"),
                "expected_ens": expected_ens,
                "actual_ens": actual_ens,
                "ensemble_dim": ensemble_dim,
                "ensemble_check_ok": ensemble_ok,
                "dir_exists": model_path.exists(),
                "paper_years": paper_years,
                "supports_paper_years": len(missing_paper_years) == 0,
                "missing_paper_years": missing_paper_years,
                "min_year": min(years) if years else None,
                "max_year": max(years) if years else None,
                "n_years": len(years),
                "years": years,
                "path": str(model_path),
            }
        )

    coverage_df = pd.DataFrame(rows).sort_values(
        by=["supports_paper_years", "ensemble_check_ok", "n_years", "model"],
        ascending=[False, False, False, True],
    )
    return coverage_df


def choose_recommended_model(coverage_df: pd.DataFrame) -> pd.Series:
    """Choose one recommended model row from a coverage summary table."""
    if coverage_df.empty:
        raise ValueError("coverage_df is empty. Cannot select a recommended model.")
    candidates = coverage_df[
        coverage_df["supports_paper_years"] & coverage_df["ensemble_check_ok"]
    ]
    return candidates.iloc[0] if len(candidates) > 0 else coverage_df.iloc[0]


def compute_window_delta_table(
    config: WindowDeltaRunConfig,
    window_bins: list[tuple[int, int]] | None = None,
    window_tolerance: list[int] | None = None,
) -> pd.DataFrame:
    """Compute one model's forecast-window delta table against climatology."""
    bins = window_bins or DEFAULT_WINDOW_BINS
    tolerance = window_tolerance or DEFAULT_WINDOW_TOLERANCE
    if len(bins) != len(tolerance):
        raise ValueError("`window_bins` and `window_tolerance` must have equal length.")

    probabilistic = ProbabilisticOnsetMetrics()
    deterministic = DeterministicOnsetMetrics()
    climatology = ClimatologyOnsetMetrics()

    with xr.open_dataset(config.thres_file) as thres_ds:
        thres_da = thres_ds["MWmean"].load()

    cmz_pairs, inside_lats, inside_lons, thres_slice = _extract_cmz_pairs(
        thres_da, use_legacy_cmz_polygon=config.use_legacy_cmz_polygon
    )

    climatology_onset = climatology.compute_climatological_onset(
        str(config.imd_folder), str(config.thres_file), mok=config.mok
    ).sel(lat=inside_lats, lon=inside_lons)

    model_by_year: dict[int, pd.DataFrame] = {}
    climatology_by_year: dict[int, pd.DataFrame] = {}
    processed_years: list[int] = []
    skipped_years: list[int] = []

    for year in config.years:
        forecast_file = config.model_forecast_dir / config.file_pattern.format(year)
        if not forecast_file.exists():
            print(f"Skipping year {year}: missing forecast file {forecast_file.name}")
            skipped_years.append(year)
            continue

        print(f"Processing year {year} for model '{config.model_name}' ...")
        try:
            if config.model_type == "probabilistic":
                if config.mem_num is None:
                    raise ValueError("`mem_num` is required for probabilistic models.")
                p_model, init_times = (
                    probabilistic.get_forecast_probabilistic_twice_weekly_2(
                        year,
                        str(config.model_forecast_dir),
                        config.mem_num,
                        date_filter_year=config.date_filter_year,
                        file_pattern=config.file_pattern,
                    )
                )
                p_model = p_model.sel(lat=inside_lats, lon=inside_lons)
            elif config.model_type == "deterministic":
                p_model = deterministic.get_forecast_deterministic_twice_weekly(
                    year, str(config.model_forecast_dir)
                ).sel(lat=inside_lats, lon=inside_lons)
                init_times = p_model.init_time.values
            else:
                raise ValueError(f"Unsupported model_type: {config.model_type}")

            rainfall = OnsetMetricsBase.load_imd_rainfall(
                year, str(config.imd_folder)
            ).sel(lat=inside_lats, lon=inside_lons)
            observed_onset = OnsetMetricsBase.detect_observed_onset(
                rainfall, thres_slice, year, mok=config.mok
            )

            if config.model_type == "probabilistic":
                model_df = probabilistic.compute_mean_onset_for_all_members(
                    p_model,
                    thres_slice,
                    observed_onset,
                    max_forecast_day=config.max_forecast_day,
                    mok=config.mok,
                )
            else:
                model_df = deterministic.compute_onset_for_deterministic_model(
                    p_model,
                    thres_slice,
                    observed_onset,
                    max_forecast_day=config.max_forecast_day,
                    mok=config.mok,
                )
            model_df = model_df.merge(cmz_pairs, on=["lat", "lon"], how="inner")

            clim_df = climatology.compute_climatology_as_forecast(
                climatology_onset,
                year,
                pd.to_datetime(init_times),
                observed_onset,
                max_forecast_day=config.max_forecast_day,
                mok=config.mok,
            )
            clim_df = clim_df.merge(cmz_pairs, on=["lat", "lon"], how="inner")

            model_by_year[year] = model_df
            climatology_by_year[year] = clim_df
            processed_years.append(year)
        except Exception as exc:
            if config.strict_year_failures:
                raise RuntimeError(
                    f"Year {year} failed for model '{config.model_name}'."
                ) from exc
            print(f"Skipping year {year}: {exc}")
            skipped_years.append(year)

    if not model_by_year or not climatology_by_year:
        raise ValueError(
            "No years were successfully processed. Check paths, year coverage, and model settings."
        )

    print(f"Processed years for '{config.model_name}': {processed_years}")
    if skipped_years:
        print(f"Skipped years for '{config.model_name}': {skipped_years}")

    rows: list[dict[str, Any]] = []
    for (start_day, end_day), tolerance_days in zip(bins, tolerance):
        if config.temporal_agg_mode == "year_then_average":
            model_vals = []
            clim_vals = []
            model_counts = []
            clim_counts = []

            for year in processed_years:
                m = compute_cmz_window_metrics(
                    model_by_year[year],
                    start_day,
                    end_day,
                    tolerance_days,
                    scoring=config.scoring,
                )
                c = compute_cmz_window_metrics(
                    climatology_by_year[year],
                    start_day,
                    end_day,
                    tolerance_days,
                    scoring=config.scoring,
                )
                model_vals.append(m)
                clim_vals.append(c)
                model_counts.append(
                    (
                        m["tp"],
                        m["fp"],
                        m["fn"],
                        m["tn"],
                        m.get("n_rows", 0),
                        m.get("n_ignored", 0),
                    )
                )
                clim_counts.append(
                    (
                        c["tp"],
                        c["fp"],
                        c["fn"],
                        c["tn"],
                        c.get("n_rows", 0),
                        c.get("n_ignored", 0),
                    )
                )

            model_stats = {
                "mae": _nanmean_dict(model_vals, "mae"),
                "far": _nanmean_dict(model_vals, "far"),
                "mr": _nanmean_dict(model_vals, "mr"),
                "tp": int(sum(x[0] for x in model_counts)),
                "fp": int(sum(x[1] for x in model_counts)),
                "fn": int(sum(x[2] for x in model_counts)),
                "tn": int(sum(x[3] for x in model_counts)),
                "n_rows": int(sum(x[4] for x in model_counts)),
                "n_ignored": int(sum(x[5] for x in model_counts)),
            }
            clim_stats = {
                "mae": _nanmean_dict(clim_vals, "mae"),
                "far": _nanmean_dict(clim_vals, "far"),
                "mr": _nanmean_dict(clim_vals, "mr"),
                "tp": int(sum(x[0] for x in clim_counts)),
                "fp": int(sum(x[1] for x in clim_counts)),
                "fn": int(sum(x[2] for x in clim_counts)),
                "tn": int(sum(x[3] for x in clim_counts)),
                "n_rows": int(sum(x[4] for x in clim_counts)),
                "n_ignored": int(sum(x[5] for x in clim_counts)),
            }
        elif config.temporal_agg_mode == "pool_all_years":
            all_model = pd.concat(
                [model_by_year[year] for year in processed_years], ignore_index=True
            )
            all_clim = pd.concat(
                [climatology_by_year[year] for year in processed_years],
                ignore_index=True,
            )
            model_stats = compute_cmz_window_metrics(
                all_model, start_day, end_day, tolerance_days, scoring=config.scoring
            )
            clim_stats = compute_cmz_window_metrics(
                all_clim, start_day, end_day, tolerance_days, scoring=config.scoring
            )
        else:
            raise ValueError(
                "Unsupported temporal_agg_mode. Use 'year_then_average' or 'pool_all_years'."
            )

        for metric_name, metric_label in [
            ("mae", "MAE (days)"),
            ("far", "FAR (%)"),
            ("mr", "MR (%)"),
        ]:
            model_value = model_stats[metric_name]
            climatology_value = clim_stats[metric_name]
            delta = (
                model_value - climatology_value
                if pd.notna(model_value) and pd.notna(climatology_value)
                else np.nan
            )
            rows.append(
                {
                    "metric": metric_label,
                    "window": f"{start_day}-{end_day}",
                    "tolerance_days": tolerance_days,
                    "model": model_value,
                    "climatology": climatology_value,
                    "delta": delta,
                    "model_tp": model_stats["tp"],
                    "model_fp": model_stats["fp"],
                    "model_fn": model_stats["fn"],
                    "model_tn": model_stats["tn"],
                    "clim_tp": clim_stats["tp"],
                    "clim_fp": clim_stats["fp"],
                    "clim_fn": clim_stats["fn"],
                    "clim_tn": clim_stats["tn"],
                    "n_rows_model": model_stats["n_rows"],
                    "n_rows_clim": clim_stats["n_rows"],
                    "model_n_ignored": int(model_stats.get("n_ignored", 0)),
                    "clim_n_ignored": int(clim_stats.get("n_ignored", 0)),
                    "processed_years": ",".join(str(y) for y in processed_years),
                    "skipped_years": ",".join(str(y) for y in skipped_years),
                    "date_filter_year": config.date_filter_year,
                    "obs_window_mode": config.scoring.obs_window_mode,
                    "classification_mode": config.scoring.classification_mode,
                    "fn_tn_obs_mode": config.scoring.fn_tn_obs_mode,
                    "mr_denom_mode": config.scoring.mr_denom_mode,
                    "aggregate_mode": config.scoring.aggregate_mode,
                    "temporal_agg_mode": config.temporal_agg_mode,
                    "model_name": config.model_name,
                }
            )

    return pd.DataFrame(rows)


def build_model_run_config(
    *,
    model_name: str,
    base_config: dict[str, Any],
    model_specs: dict[str, dict[str, Any]],
    paper_years_by_model: dict[str, list[int]] | None = None,
    scoring: WindowScoringConfig | None = None,
) -> WindowDeltaRunConfig:
    """Build :class:`WindowDeltaRunConfig` for one model from shared settings."""
    if model_name not in model_specs:
        raise ValueError(f"Unknown model '{model_name}'.")
    years_map = paper_years_by_model or PAPER_YEARS_BY_MODEL_4DEG
    if model_name not in years_map:
        raise ValueError(f"Missing paper years for model '{model_name}'.")

    spec = model_specs[model_name]
    scoring_cfg = scoring or WindowScoringConfig()
    return WindowDeltaRunConfig(
        model_name=model_name,
        model_type=str(spec["model_type"]),
        model_forecast_dir=Path(spec["path"]),
        imd_folder=Path(base_config["imd_folder"]),
        thres_file=Path(base_config["thres_file"]),
        years=list(years_map[model_name]),
        mem_num=spec.get("mem_num"),
        mok=bool(base_config.get("mok", True)),
        max_forecast_day=int(base_config.get("max_forecast_day", 30)),
        date_filter_year=int(
            spec.get("date_filter_year", base_config.get("date_filter_year", 2024))
        ),
        file_pattern=str(base_config.get("file_pattern", "{}.nc")),
        temporal_agg_mode=str(
            base_config.get("temporal_agg_mode", "year_then_average")
        ),
        use_legacy_cmz_polygon=bool(base_config.get("use_legacy_cmz_polygon", True)),
        strict_year_failures=bool(base_config.get("strict_year_failures", False)),
        scoring=scoring_cfg,
    )


def run_single_model_window_analysis(
    model_name: str,
    base_config: dict[str, Any],
    model_specs: dict[str, dict[str, Any]],
    paper_years_by_model: dict[str, list[int]] | None = None,
    scoring: WindowScoringConfig | None = None,
    window_bins: list[tuple[int, int]] | None = None,
    window_tolerance: list[int] | None = None,
) -> pd.DataFrame:
    """Run one model and return its model-vs-climatology delta table."""
    run_cfg = build_model_run_config(
        model_name=model_name,
        base_config=base_config,
        model_specs=model_specs,
        paper_years_by_model=paper_years_by_model,
        scoring=scoring,
    )
    return compute_window_delta_table(
        run_cfg,
        window_bins=window_bins,
        window_tolerance=window_tolerance,
    )


def run_multi_model_window_analysis(
    *,
    base_config: dict[str, Any],
    model_specs: dict[str, dict[str, Any]],
    paper_years_by_model: dict[str, list[int]] | None = None,
    scoring: WindowScoringConfig | None = None,
    window_bins: list[tuple[int, int]] | None = None,
    window_tolerance: list[int] | None = None,
) -> dict[str, Any]:
    """Run a paper-style multi-model window analysis and collect tabular outputs."""
    bins = window_bins or DEFAULT_WINDOW_BINS
    tolerance = window_tolerance or DEFAULT_WINDOW_TOLERANCE
    years_map = paper_years_by_model or PAPER_YEARS_BY_MODEL_4DEG
    scoring_cfg = scoring or WindowScoringConfig()
    window_labels = [f"{start}-{end}" for start, end in bins]

    metric_key_map = {
        "MAE (days)": "mae_cmz",
        "FAR (%)": "far_cmz",
        "MR (%)": "mr_cmz",
    }

    matrices = {
        "mae_cmz": pd.DataFrame(np.nan, index=window_labels, columns=MAT_MODEL_ORDER),
        "far_cmz": pd.DataFrame(np.nan, index=window_labels, columns=MAT_MODEL_ORDER),
        "mr_cmz": pd.DataFrame(np.nan, index=window_labels, columns=MAT_MODEL_ORDER),
        "std_er": pd.DataFrame(np.nan, index=window_labels, columns=MAT_MODEL_ORDER),
    }
    delta_for_plot = {
        "MAE (days)": pd.DataFrame(
            np.nan, index=PLOT_MODEL_ORDER, columns=window_labels
        ),
        "FAR (%)": pd.DataFrame(np.nan, index=PLOT_MODEL_ORDER, columns=window_labels),
        "MR (%)": pd.DataFrame(np.nan, index=PLOT_MODEL_ORDER, columns=window_labels),
    }

    all_delta_rows: list[dict[str, Any]] = []
    per_model_delta_tables: dict[str, pd.DataFrame] = {}
    baseline_conflicts: list[tuple[Any, ...]] = []

    for model_name in model_specs:
        if model_name not in years_map:
            raise ValueError(f"Missing paper years for model '{model_name}'.")
        if model_name not in MODEL_TO_MAT_NAME:
            raise ValueError(f"Missing MAT mapping for model '{model_name}'.")

        run_config = build_model_run_config(
            model_name=model_name,
            base_config=base_config,
            model_specs=model_specs,
            paper_years_by_model=years_map,
            scoring=scoring_cfg,
        )

        print(f"\nRunning model={model_name}, years={run_config.years}")
        delta_table = compute_window_delta_table(
            run_config,
            window_bins=bins,
            window_tolerance=tolerance,
        )
        per_model_delta_tables[model_name] = delta_table

        mat_name = MODEL_TO_MAT_NAME[model_name]
        for _, row in delta_table.iterrows():
            metric = str(row["metric"])
            window = str(row["window"])
            key = metric_key_map[metric]

            model_val = float(row["model"]) if pd.notna(row["model"]) else np.nan
            clim_val = (
                float(row["climatology"]) if pd.notna(row["climatology"]) else np.nan
            )
            delta_val = float(row["delta"]) if pd.notna(row["delta"]) else np.nan

            matrices[key].loc[window, mat_name] = model_val
            previous = matrices[key].loc[window, "clim"]
            if pd.isna(previous):
                matrices[key].loc[window, "clim"] = clim_val
            elif pd.notna(clim_val) and abs(previous - clim_val) > 1e-8:
                baseline_conflicts.append(
                    (metric, window, model_name, previous, clim_val)
                )

            if model_name in PLOT_MODEL_ORDER:
                delta_for_plot[metric].loc[model_name, window] = delta_val

            all_delta_rows.append(
                {
                    "model": model_name,
                    "window": window,
                    "metric": metric,
                    "tolerance_days": int(row["tolerance_days"])
                    if pd.notna(row["tolerance_days"])
                    else np.nan,
                    "model_value": model_val,
                    "climatology_value": clim_val,
                    "delta": delta_val,
                    "years_used": ",".join(str(y) for y in run_config.years),
                }
            )

    return {
        "per_model_delta_tables": per_model_delta_tables,
        "matrices": matrices,
        "delta_for_plot": delta_for_plot,
        "all_delta_long": pd.DataFrame(all_delta_rows),
        "baseline_conflicts": baseline_conflicts,
        "window_labels": window_labels,
        "plot_model_order": PLOT_MODEL_ORDER,
        "plot_model_labels": PLOT_MODEL_LABELS,
    }


def save_window_analysis_outputs(
    batch_result: dict[str, Any],
    output_dir: Path,
    prefix: str = "forecast_window_skill",
) -> dict[str, Path]:
    """Save batch analysis outputs to CSV files and return file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for model_name, table in batch_result["per_model_delta_tables"].items():
        path = output_dir / f"{prefix}_delta_table_{model_name}.csv"
        table.to_csv(path, index=False)
        outputs[f"delta_table_{model_name}"] = path

    for key, matrix in batch_result["matrices"].items():
        path = output_dir / f"{prefix}_{key}.csv"
        matrix.to_csv(path, index=True)
        outputs[key] = path

    model_order_path = output_dir / f"{prefix}_model_order.csv"
    pd.DataFrame({"model": MAT_MODEL_ORDER}).to_csv(model_order_path, index=False)
    outputs["model_order"] = model_order_path

    long_path = output_dir / f"{prefix}_all_models_delta_long.csv"
    batch_result["all_delta_long"].to_csv(long_path, index=False)
    outputs["all_delta_long"] = long_path

    return outputs


def run_and_save_multi_model_analysis(
    output_dir: Path,
    base_config: dict[str, Any],
    model_specs: dict[str, dict[str, Any]],
    paper_years_by_model: dict[str, list[int]] | None = None,
    scoring: WindowScoringConfig | None = None,
    window_bins: list[tuple[int, int]] | None = None,
    window_tolerance: list[int] | None = None,
    prefix: str = "forecast_window_skill",
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Run multi-model analysis and save all CSV outputs in one call."""
    batch_result = run_multi_model_window_analysis(
        base_config=base_config,
        model_specs=model_specs,
        paper_years_by_model=paper_years_by_model,
        scoring=scoring,
        window_bins=window_bins,
        window_tolerance=window_tolerance,
    )
    saved = save_window_analysis_outputs(
        batch_result=batch_result,
        output_dir=output_dir,
        prefix=prefix,
    )
    return batch_result, saved
