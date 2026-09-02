import os
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("satquery.geo_utils")

def extract_bounds_and_crs(file_path: str) -> Dict[str, Any]:
    """
    Extracts CRS and bounding box from satellite imagery using rasterio.
    Reprojects bounds to EPSG:4326 (WGS84) for Leaflet rendering.
    Returns metadata dict containing CRS info and GeoJSON bounds polygon.
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError:
        logger.warning("rasterio not installed. Falling back to default WGS84 bounding box.")
        return get_default_bounds()

    try:
        with rasterio.open(file_path) as dataset:
            crs_str = dataset.crs.to_string() if dataset.crs else "EPSG:4326"
            bounds = dataset.bounds  # (left, bottom, right, top)
            
            # Reproject to EPSG:4326 if needed
            if dataset.crs and dataset.crs.to_string() != "EPSG:4326":
                min_lon, min_lat, max_lon, max_lat = transform_bounds(
                    dataset.crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top
                )
            else:
                min_lon, min_lat, max_lon, max_lat = bounds.left, bounds.bottom, bounds.right, bounds.top

            center_lat = (min_lat + max_lat) / 2.0
            center_lon = (min_lon + max_lon) / 2.0

            return {
                "crs": crs_str,
                "reprojected_crs": "EPSG:4326",
                "width": dataset.width,
                "height": dataset.height,
                "bounds": [min_lon, min_lat, max_lon, max_lat],
                "center": [center_lon, center_lat],
                "geojson_bounds": {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat]
                        ]]
                    },
                    "properties": {
                        "label": "Image Footprint",
                        "crs": crs_str
                    }
                }
            }
    except Exception as e:
        logger.error(f"Error reading geospatial metadata from {file_path}: {e}")
        return get_default_bounds()

def get_default_bounds() -> Dict[str, Any]:
    """Default fallback bounds (ISRO SAC area, Ahmedabad) in EPSG:4326"""
    min_lon, min_lat, max_lon, max_lat = 72.50, 23.01, 72.55, 23.05
    return {
        "crs": "EPSG:4326",
        "reprojected_crs": "EPSG:4326",
        "width": 1024,
        "height": 1024,
        "bounds": [min_lon, min_lat, max_lon, max_lat],
        "center": [(min_lon + max_lon)/2.0, (min_lat + max_lat)/2.0],
        "geojson_bounds": {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            },
            "properties": {
                "label": "Fallback Satellite Image Footprint",
                "crs": "EPSG:4326"
            }
        }
    }
