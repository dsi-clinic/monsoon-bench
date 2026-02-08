"""Spatial regions and boundaries for India.

This module provides utilities for working with Indian geographic boundaries,
including the national outline and the Core Monsoon Zone (CMZ).
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
from matplotlib.path import Path as MplPath

# Constants for resolution matching
RESOLUTION_TOLERANCE = 0.1  # Tolerance for resolution matching in degrees
MIN_LAT_POINTS = 2  # Minimum number of latitude points needed for resolution detection


def get_india_outline(
    shp_file_path: str | Path,
) -> list[tuple[list[float], list[float]]]:
    """Extract India outline coordinates from shapefile.

    Args:
        shp_file_path: Path to India shapefile

    Returns:
        List of (lon_coords, lat_coords) tuples for each boundary segment

    Example:
        >>> boundaries = get_india_outline("data/india.shp")
        >>> for lon_coords, lat_coords in boundaries:
        ...     plt.plot(lon_coords, lat_coords, 'k-')
    """
    india_gdf = gpd.read_file(shp_file_path)

    boundaries = []
    for geom in india_gdf.geometry:
        if hasattr(geom, "exterior"):
            coords = list(geom.exterior.coords)
            lon_coords = [coord[0] for coord in coords]
            lat_coords = [coord[1] for coord in coords]
            boundaries.append((lon_coords, lat_coords))
        elif hasattr(geom, "geoms"):
            for sub_geom in geom.geoms:
                if hasattr(sub_geom, "exterior"):
                    coords = list(sub_geom.exterior.coords)
                    lon_coords = [coord[0] for coord in coords]
                    lat_coords = [coord[1] for coord in coords]
                    boundaries.append((lon_coords, lat_coords))
    return boundaries


def get_cmz_polygon_coords(resolution: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Get Core Monsoon Zone (CMZ) polygon coordinates for a given resolution.

    The CMZ is defined based on the grid resolution to ensure proper alignment
    with the data grid.

    Args:
        resolution: Grid resolution in degrees (e.g., 1.0, 2.0, 4.0)

    Returns:
        Tuple of (lon_coords, lat_coords) arrays, or None if resolution not supported

    Example:
        >>> lon, lat = get_cmz_polygon_coords(2.0)
        >>> plt.plot(lon, lat, 'r-', linewidth=2)
    """
    # 2-degree resolution CMZ
    if abs(resolution - 2.0) < RESOLUTION_TOLERANCE:
        polygon_lon = np.array(
            [83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83]
        )
        polygon_lat = np.array(
            [17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17]
        )
        return polygon_lon, polygon_lat

    # 1-degree resolution CMZ
    if abs(resolution - 1.0) < RESOLUTION_TOLERANCE:
        polygon_lon = np.array(
            [82, 74, 74, 70, 70, 76, 76, 78, 78, 82, 82, 88, 88, 84, 84, 82, 82]
        )
        polygon_lat = np.array(
            [16, 16, 20, 20, 28, 28, 26, 26, 24, 24, 22, 22, 20, 20, 18, 18, 16]
        )
        return polygon_lon, polygon_lat

    # 4-degree resolution CMZ
    if abs(resolution - 4.0) < RESOLUTION_TOLERANCE:
        polygon_lon = np.array([84, 76, 76, 72, 72, 80, 80, 84, 84, 88, 88, 84, 84])
        polygon_lat = np.array([16, 16, 20, 20, 28, 28, 24, 24, 20, 20, 16, 16, 16])
        return polygon_lon, polygon_lat

    # Resolution not supported
    return None


def detect_resolution(lats: np.ndarray) -> float:
    """Detect grid resolution from latitude array.

    Args:
        lats: Array of latitude values

    Returns:
        Detected resolution in degrees

    Example:
        >>> lats = np.array([10.0, 12.0, 14.0, 16.0])
        >>> resolution = detect_resolution(lats)
        >>> print(f"Resolution: {resolution}°")
        Resolution: 2.0°
    """
    if len(lats) < MIN_LAT_POINTS:
        msg = f"Need at least {MIN_LAT_POINTS} latitude values to detect resolution"
        raise ValueError(msg)

    lat_diff = abs(lats[1] - lats[0])
    return float(lat_diff)


# Function to find grid points inside a polygon (For core-monsoon zone analysis)
def points_inside_polygon(polygon_lon, polygon_lat, grid_lons, grid_lats):
    """Find grid points that are inside a polygon.

    Parameters:
    polygon_lon: array of polygon longitude vertices
    polygon_lat: array of polygon latitude vertices
    grid_lons: array of grid longitude points
    grid_lats: array of grid latitude points

    Returns:
    inside_mask: boolean array indicating which points are inside
    inside_lons: longitude coordinates of points inside polygon
    inside_lats: latitude coordinates of points inside polygon
    """
    # Create polygon path
    polygon_vertices = np.column_stack((polygon_lon, polygon_lat))
    polygon_path = MplPath(polygon_vertices)

    # Create meshgrid if needed
    if grid_lons.ndim == 1 and grid_lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    else:
        lon_grid, lat_grid = grid_lons, grid_lats

    # Flatten the grids to test each point
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    # Test which points are inside the polygon
    inside_mask = polygon_path.contains_points(points)
    inside_mask = inside_mask.reshape(lon_grid.shape)

    # Get coordinates of points inside polygon
    inside_lons = lon_grid[inside_mask]
    inside_lats = lat_grid[inside_mask]

    return inside_mask, inside_lons, inside_lats
