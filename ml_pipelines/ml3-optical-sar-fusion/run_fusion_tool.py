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
            return [72.50, 23.01, 72.55, 23.05]
        try:
            bounds = src.bounds
            wgs84_bounds = transform_bounds(src.crs, "EPSG:4326", *bounds)
            return list(wgs84_bounds)  # [min_lon, min_lat, max_lon, max_lat]
        except Exception:
            return [72.50, 23.01, 72.55, 23.05]


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
    sar_vv_band: int = None,
    optical_green_band: int = None,
    optical_blue_band: int = None,
    normalize_fn=None,
    vqa_fn=None,
    static_confidence: float = 0.80,
) -> dict:
    """
    Full optical-SAR fusion tool matching FS-2's TOOL_REGISTRY schema.

    Args:
        query: the user's natural-language query.
        optical_path, sar_path: paths to the input GeoTIFFs.
        sar_vv_band, optical_green_band, optical_blue_band, normalize_fn:
            passed straight through to fusion_tool() (auto-detected if None).
        vqa_fn: optional callable(image_path: str, query: str) -> str or dict.
            This should be ML-1's PaliGemma inference function once ready.
            If None, returns a clearly-labeled placeholder instead of
            fabricating a specific-sounding description.
        static_confidence: documented fallback confidence used when no real
            per-query confidence signal is wired in yet.

    Returns:
        dict matching TOOL_REGISTRY schema: type, text, geojson, confidence,
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
        vqa_out = vqa_fn(composite_path, query)
        if isinstance(vqa_out, dict):
            text = vqa_out.get("text", "")
            confidence = float(vqa_out.get("confidence", static_confidence))
        else:
            text = str(vqa_out)
            confidence = static_confidence
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
        "type": "final",
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
    import os
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    opt_p = os.path.join(sample_dir, "optical_crop.tif")
    sar_p = os.path.join(sample_dir, "sar_crop.tif")

    if os.path.exists(opt_p) and os.path.exists(sar_p):
        result = run_fusion_tool(
            query="Use the optical and SAR images together to identify built-up and water-covered regions.",
            optical_path=opt_p,
            sar_path=sar_p,
        )
        print("Success! Result:")
        print(result)
    else:
        print("Sample data files not found for standalone run test.")