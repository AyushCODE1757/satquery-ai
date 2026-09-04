# """
# run_change_tool.py — ML-2's real change_tool, wrapped to match FS-2's
# TOOL_REGISTRY contract (same pattern as ML-3's run_fusion_tool.py).
# """

# import cv2
# import numpy as np
# import rasterio
# from rasterio.warp import transform_bounds


# def change_tool(image1, image2, threshold_value=30, min_area=5):
#     """ML-2's original function, unchanged — pixel-space contours."""
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#     diff = cv2.absdiff(gray1, gray2)
#     _, change_mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     results = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area < min_area:
#             continue
#         points = contour.reshape(-1, 2).tolist()  # pixel [x, y] pairs
#         results.append({"pixel_points": points, "area": float(area)})
#     return results


# def get_wgs84_bounds(tif_path: str) -> list:
#     """Same approach as ML-3's fusion tool — real CRS-based bounds."""
#     with rasterio.open(tif_path) as src:
#         if src.crs is None:
#             raise ValueError(f"{tif_path} has no CRS — cannot compute bounds.")
#         bounds = src.bounds
#         return list(transform_bounds(src.crs, "EPSG:4326", *bounds))


# def pixel_to_lonlat(px, py, img_width, img_height, wgs84_bounds):
#     """Linear-interpolate a pixel coordinate into the image's real-world bounds.
#     Good enough for a false-color/contour overlay; not a substitute for a
#     full affine geotransform if precision becomes critical later."""
#     min_lon, min_lat, max_lon, max_lat = wgs84_bounds
#     lon = min_lon + (px / img_width) * (max_lon - min_lon)
#     lat = max_lat - (py / img_height) * (max_lat - min_lat)  # image Y grows downward
#     return [lon, lat]


# def run_change_tool(query: str, image1_path: str, image2_path: str,
#                      threshold_value: int = 30, min_area: int = 5) -> dict:
#     """
#     Full change-detection tool matching FS-2's TOOL_REGISTRY schema.
#     Reads two co-registered .tif images, runs ML-2's real cv2 diff logic,
#     and converts pixel contours into real WGS84 GeoJSON polygons.
#     """
#     with rasterio.open(image1_path) as src1:
#         img1 = src1.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8) if src1.count >= 3 \
#             else cv2.cvtColor(src1.read(1).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#         height, width = src1.height, src1.width

#     with rasterio.open(image2_path) as src2:
#         img2 = src2.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8) if src2.count >= 3 \
#             else cv2.cvtColor(src2.read(1).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#     wgs84_bounds = get_wgs84_bounds(image1_path)  # assumes co-registered pair, per PS spec

#     raw_regions = change_tool(img1, img2, threshold_value, min_area)

#     features = []
#     for region in raw_regions:
#         lonlat_points = [
#             pixel_to_lonlat(px, py, width, height, wgs84_bounds)
#             for px, py in region["pixel_points"]
#         ]
#         if lonlat_points[0] != lonlat_points[-1]:
#             lonlat_points.append(lonlat_points[0])
#         features.append({
#             "type": "Feature",
#             "geometry": {"type": "Polygon", "coordinates": [lonlat_points]},
#             "properties": {
#                 "label": "Detected change region",
#                 "area_px": region["area"],
#                 "confidence": 0.0,  # honest placeholder — see note below
#             },
#         })

#     num_changes = len(features)
#     if num_changes == 0:
#         text = "No significant structural change detected between the two dates above the current sensitivity threshold."
#     else:
#         text = (
#             f"Detected {num_changes} region(s) of structural change between the two dates. "
#             f"[PENDING] Semantic description of WHAT changed requires ML-2's CDVQA-based "
#             f"upgrade (V2 plan) — this MVP version identifies WHERE change occurred via "
#             f"pixel-difference only."
#         )

#     return {
#         "type": "final",
#         "text": text,
#         "geojson": {"type": "FeatureCollection", "features": features},
#         "confidence": 0.0 if num_changes else 0.5,  # don't fabricate a number for un-validated detections
#         "execution_summary": {
#             "task": "change_vqa",
#             "models_used": ["change_tool_cv2_mvp"],
#             "params": {"query": query, "threshold_value": threshold_value, "min_area": min_area},
#         },
#     }

"""
run_change_tool.py — wraps ML-2's real change_tool() to match FS-2's
TOOL_REGISTRY contract. Falls back to a default bounding box if the
input images have no CRS (e.g. CDVQA benchmark PNGs).
"""

import cv2
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from change_tool import change_tool

DEFAULT_BOUNDS = [72.50, 23.01, 72.55, 23.05]  # fallback for non-georeferenced test images


def get_wgs84_bounds_or_default(tif_path: str) -> list:
    try:
        with rasterio.open(tif_path) as src:
            if src.crs is None:
                return DEFAULT_BOUNDS
            return list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
    except Exception:
        return DEFAULT_BOUNDS


def load_image_as_bgr(path: str) -> np.ndarray:
    """Reads either a georeferenced .tif or a plain .png/.jpg into a cv2-compatible BGR array."""
    try:
        with rasterio.open(path) as src:
            if src.count >= 3:
                arr = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                gray = src.read(1).astype(np.uint8)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return img


def pixel_to_lonlat(px, py, img_width, img_height, wgs84_bounds):
    min_lon, min_lat, max_lon, max_lat = wgs84_bounds
    lon = min_lon + (px / img_width) * (max_lon - min_lon)
    lat = max_lat - (py / img_height) * (max_lat - min_lat)
    return [lon, lat]


def run_change_tool(query: str, image1_path: str, image2_path: str,
                     threshold_value: int = 30, min_area: int = 5) -> dict:
    img1 = load_image_as_bgr(image1_path)
    img2 = load_image_as_bgr(image2_path)
    height, width = img1.shape[:2]

    wgs84_bounds = get_wgs84_bounds_or_default(image1_path)
    pixel_geojson = change_tool(img1, img2, threshold_value, min_area)

    features = []
    for feat in pixel_geojson["features"]:
        pixel_points = feat["geometry"]["coordinates"][0]
        lonlat_points = [pixel_to_lonlat(px, py, width, height, wgs84_bounds) for px, py in pixel_points]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [lonlat_points]},
            "properties": {
                "label": "Detected change region",
                "area_px": feat["properties"]["area"],
                "confidence": 0.0,
            },
        })

    num_changes = len(features)
    text = (
        "No significant structural change detected between the two dates above the current sensitivity threshold."
        if num_changes == 0 else
        f"Detected {num_changes} region(s) of structural change between the two dates. "
        f"[PENDING] Semantic 'what changed' description requires ML-2's CDVQA-based upgrade — "
        f"this version identifies WHERE via pixel-difference only."
    )

    return {
        "type": "final",
        "text": text,
        "geojson": {"type": "FeatureCollection", "features": features},
        "confidence": 0.0 if num_changes else 0.5,
        "execution_summary": {
            "task": "change_vqa",
            "models_used": ["change_tool_cv2_mvp"],
            "params": {"query": query, "threshold_value": threshold_value, "min_area": min_area},
        },
    }