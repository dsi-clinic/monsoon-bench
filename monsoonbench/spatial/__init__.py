"""Spatial utilities for grids, regions, and subgrid variability."""

from monsoonbench.spatial.regions import (
    detect_resolution,
    get_cmz_polygon_coords,
    get_india_outline,
    points_inside_polygon,
)
from monsoonbench.spatial.subgrid_variability import (
    compute_mean_subgrid_variability_maps,
    resolve_threshold_file,
    summarize_sample_rainfall_grid,
)

__all__ = [
    "compute_mean_subgrid_variability_maps",
    "resolve_threshold_file",
    "summarize_sample_rainfall_grid",
    "detect_resolution",
    "get_cmz_polygon_coords",
    "get_india_outline",
    "points_inside_polygon",
]
