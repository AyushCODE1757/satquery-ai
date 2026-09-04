"""
run_fusion_tool.py — ML-3 final wrapper.

Matches the expected FS-2 TOOL_REGISTRY schema.

Pipeline:

    optical TIFF
          +
    SAR TIFF
          ↓
    fusion_tool()
          ↓
    composite PNG
          ↓
    optional VQA/PaliGemma
          ↓
    text + confidence
          ↓
    GeoJSON + execution_summary
"""

import rasterio
from rasterio.warp import transform_bounds

from fusion_tool import (
    fusion_tool,
    normalize_ben_sar_to_uint8,
)


def get_wgs84_bounds(
    tif_path: str,
) -> list:
    """
    Extract the real WGS84 bounding box from a GeoTIFF.

    The optical raster is expected to provide the CRS.
    Native Sentinel-1 demo products may have CRS=None.

    Returns:
        [min_lon, min_lat, max_lon, max_lat]
    """

    with rasterio.open(tif_path) as src:

        if src.crs is None:
            raise ValueError(
                f"{tif_path} has no CRS — "
                "cannot compute WGS84 bounds."
            )

        bounds = src.bounds

        wgs84_bounds = transform_bounds(
            src.crs,
            "EPSG:4326",
            *bounds,
        )

    return list(wgs84_bounds)


def bounds_to_geojson(
    bounds: list,
    label: str,
    confidence: float,
) -> dict:
    """
    Convert WGS84 bounding box into a GeoJSON FeatureCollection.
    """

    min_lon, min_lat, max_lon, max_lat = bounds

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [
                                min_lon,
                                min_lat,
                            ],
                            [
                                max_lon,
                                min_lat,
                            ],
                            [
                                max_lon,
                                max_lat,
                            ],
                            [
                                min_lon,
                                max_lat,
                            ],
                            [
                                min_lon,
                                min_lat,
                            ],
                        ]
                    ],
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
) -> dict:
    """
    Run the complete optical-SAR fusion pipeline.

    Args:
        query:
            User's natural-language query.

        optical_path:
            Optical TIFF path.

        sar_path:
            SAR TIFF path.

        sar_vv_band:
            SAR VV band index.

            BigEarthNet:
                2

            Native Bhoonidhi:
                1

        optical_green_band:
            Green band index.

        optical_blue_band:
            Blue band index.

        normalize_fn:
            SAR normalization function.

        vqa_fn:
            Optional VQA inference callable:

                vqa_fn(image_path, query)

            Expected preferred return format:

                {
                    "text": "...",
                    "confidence": 0.87
                }

            Legacy string return values are also accepted.

    Returns:
        Dictionary matching the TOOL_REGISTRY schema.
    """

    # ---------------------------------------------------------
    # 1. Generate optical-SAR composite
    # ---------------------------------------------------------

    fusion_result = fusion_tool(
        optical_path,
        sar_path,
        sar_vv_band=sar_vv_band,
        optical_green_band=optical_green_band,
        optical_blue_band=optical_blue_band,
        normalize_fn=normalize_fn,
    )

    composite_path = fusion_result[
        "image_path"
    ]

    # ---------------------------------------------------------
    # 2. Extract geographic bounds
    # ---------------------------------------------------------

    bounds = get_wgs84_bounds(
        optical_path
    )

    # ---------------------------------------------------------
    # 3. Optional VQA inference
    # ---------------------------------------------------------

    if vqa_fn is not None:

        inference_result = vqa_fn(
            composite_path,
            query,
        )

        if isinstance(
            inference_result,
            dict,
        ):

            text = str(
                inference_result.get(
                    "text",
                    "",
                )
            )

            confidence = float(
                inference_result.get(
                    "confidence",
                    0.0,
                )
            )

        else:

            # Backward-compatible support
            # for string-returning VQA functions.
            text = str(
                inference_result
            )

            confidence = 0.0

    else:

        text = (
            "[PENDING] Optical-SAR fusion "
            "composite generated successfully, "
            "but VQA inference is not wired in."
        )

        confidence = 0.0

    # ---------------------------------------------------------
    # 4. GeoJSON
    # ---------------------------------------------------------

    geojson = bounds_to_geojson(
        bounds,
        label=(
            "Optical-SAR Fusion Composite: "
            f"{query}"
        ),
        confidence=confidence,
    )

    # ---------------------------------------------------------
    # 5. Final TOOL_REGISTRY schema
    # ---------------------------------------------------------

    return {
        "text": text,

        "geojson": geojson,

        "confidence": confidence,

        "execution_summary": {
            "task": "optical_sar_fusion",

            "models_used": (
                ["fusion_tool"]
                + (
                    ["vqa_tool"]
                    if vqa_fn is not None
                    else []
                )
            ),

            "params": {
                "query": query,
                "num_images": 2,
                "composite_image_path": composite_path,
            },
        },
    }