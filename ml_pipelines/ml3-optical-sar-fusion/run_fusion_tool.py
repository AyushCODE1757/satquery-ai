"""
run_fusion_tool.py — ML-3 final wrapper matching FS-2's TOOL_REGISTRY schema.
"""

import rasterio
from rasterio.warp import transform_bounds

from fusion_tool import (
    fusion_tool,
    normalize_ben_sar_to_uint8,
    normalize_raw_sar_to_uint8,
)


def get_wgs84_bounds(tif_path: str) -> list:
    """
    Extract real WGS84 (EPSG:4326) bounding box from a GeoTIFF's CRS.
    Use the optical file's CRS — SAR files here may have CRS: None.
    """
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"{tif_path} has no CRS — cannot compute bounds.")
        bounds = src.bounds
        wgs84_bounds = transform_bounds(src.crs, "EPSG:4326", *bounds)
    return list(wgs84_bounds)  # [min_lon, min_lat, max_lon, max_lat]


def bounds_to_geojson(bounds: list, label: str, confidence: float) -> dict:
    min_lon, min_lat, max_lon, max_lat = bounds
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]],
                },
                "properties": {
                    "label": label,
                    "confidence": confidence,
                },
            }
        ],
    }


def run_fusion_tool(
    query: str,
    optical_path: str,
    sar_path: str,
    sar_vv_band: int = 2,
    optical_green_band: int = 3,
    optical_blue_band: int = 2,
    normalize_fn=normalize_ben_sar_to_uint8,
    vqa_fn=None,
    static_confidence: float = 0.80,
) -> dict:
    """
    Full optical-SAR fusion tool matching FS-2's TOOL_REGISTRY schema.

    Args:
        query: the user's natural-language query.
        optical_path, sar_path: paths to the input GeoTIFFs.
        sar_vv_band, optical_green_band, optical_blue_band, normalize_fn:
            passed straight through to fusion_tool() — see that file's
            docstring for BEN vs Bhoonidhi values.
        vqa_fn: optional callable(image_path: str, query: str) -> str.
            This should be ML-1's PaliGemma inference function once ready.
            If None, returns a clearly-labeled placeholder instead of
            fabricating a specific-sounding description.
        static_confidence: documented fallback confidence (e.g. model's
            validation-set accuracy) used when no real per-query
            confidence signal (like generation sequence scores) is wired
            in yet. Replace with a real signal when available — don't
            invent a different specific number per query.

    Returns:
        dict matching TOOL_REGISTRY schema: text, geojson, confidence,
        execution_summary.
    """
    # --- Real fusion composite ---
    fusion_result = fusion_tool(
        optical_path,
        sar_path,
        sar_vv_band=sar_vv_band,
        optical_green_band=optical_green_band,
        optical_blue_band=optical_blue_band,
        normalize_fn=normalize_fn,
    )
    composite_path = fusion_result["image_path"]

    # --- Real bounds from optical file's CRS ---
    bounds = get_wgs84_bounds(optical_path)

    # --- Text: real VQA output if wired in, else honest placeholder ---
    if vqa_fn is not None:
        text = vqa_fn(composite_path, query)
        confidence = static_confidence  # swap for real generation score once available
    else:
        text = (
            "[PENDING] Optical-SAR fusion composite generated successfully, "
            "but VQA text description is not yet wired in — awaiting "
            "ML-1's inference function."
        )
        confidence = 0.0  # explicitly zero — do not fabricate a number for un-run inference

    geojson = bounds_to_geojson(
        bounds,
        label=f"Optical-SAR Fusion Composite: {query}",
        confidence=confidence,
    )

    return {
        "text": text,
        "geojson": geojson,
        "confidence": confidence,
        "execution_summary": {
            "task": "optical_sar_fusion",
            "models_used": ["fusion_tool"] + (["vqa_tool"] if vqa_fn else []),
            "params": {
                "query": query,
                "num_images": 2,
                "composite_image_path": composite_path,
            },
        },
    }


if __name__ == "__main__":
    # Bhoonidhi real demo test
    result = run_fusion_tool(
        query="Use the optical and SAR images together to identify built-up and water-covered regions.",
        optical_path="optical_crop.tif",
        sar_path="sar_vv_crop.tif",
        sar_vv_band=1,
        optical_green_band=1,
        optical_blue_band=2,
        normalize_fn=normalize_raw_sar_to_uint8,
    )
    print(result)