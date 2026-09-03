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
            
            # If dataset has no valid spatial bounds or default unit bounds, transform or set fallback
            if dataset.crs and crs_str != "EPSG:4326":
                try:
                    min_lon, min_lat, max_lon, max_lat = transform_bounds(
                        dataset.crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top
                    )
                except Exception as transform_err:
                    logger.warning(f"Transform bounds failed for CRS {crs_str}: {transform_err}. Falling back to default WGS84 coordinates.")
                    min_lon, min_lat, max_lon, max_lat = 72.50, 23.01, 72.55, 23.05
            else:
                # If bounds are standard unit box [0,0,1,1] or missing
                if bounds.left == 0.0 and bounds.right == 1.0:
                    min_lon, min_lat, max_lon, max_lat = 72.50, 23.01, 72.55, 23.05
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
        logger.info(f"Non-geospatial image or rasterio open notice for {file_path}: {e}. Extracting image metadata via PIL...")
        return get_pil_metadata(file_path)

def get_pil_metadata(file_path: str) -> Dict[str, Any]:
    """Fallback for non-geospatial images (PNG/JPG) using PIL to extract dimensions."""
    width, height = 1024, 1024
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            width, height = img.size
    except Exception as img_err:
        logger.warning(f"Could not read image dimensions using PIL: {img_err}")

    default = get_default_bounds()
    default["width"] = width
    default["height"] = height
    return default

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