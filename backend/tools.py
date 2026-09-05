# """
# Specialist ML Tool wrappers for SatQuery AI.
# Wraps model inference for VQA, Grounding, Change Analysis, and SAR Fusion.
# Linked to ml_pipelines/ modules.
# """

# import os
# import tempfile
# import logging
# import numpy as np
# from typing import Dict, Any, List, Optional

# logger = logging.getLogger("satquery.tools")

# def get_image_bounds_or_default(image_ids: List[str], image_store: Optional[Dict[str, Any]] = None) -> List[float]:
#     """Helper to extract WGS84 bounding box from image store if available"""
#     if image_ids and image_store and image_ids[0] in image_store:
#         geo_meta = image_store[image_ids[0]].get("geo_meta", {})
#         bounds = geo_meta.get("bounds")
#         if bounds and len(bounds) == 4:
#             return bounds
#     return [72.50, 23.01, 72.55, 23.05]

# def run_vqa_tool(query: str, image_ids: List[str], image_store: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """Single Image VQA Model Execution Wrapper"""
#     logger.info(f"Executing vqa_tool for query: '{query}' with image_ids: {image_ids}")
#     bounds = get_image_bounds_or_default(image_ids, image_store)
#     center_lon = (bounds[0] + bounds[2]) / 2.0
#     center_lat = (bounds[1] + bounds[3]) / 2.0

#     return {
#         "type": "final",
#         "text": f"VQA Analysis: Inspected target optical imagery for query '{query}'. Identified 4 solar farm arrays and 2 power substations within the active satellite footprint.",
#         "geojson": {
#             "type": "FeatureCollection",
#             "features": [
#                 {
#                     "type": "Feature",
#                     "geometry": {"type": "Point", "coordinates": [center_lon, center_lat]},
#                     "properties": {
#                         "label": f"VQA Detected Feature for '{query}'",
#                         "confidence": 0.89
#                     }
#                 }
#             ]
#         },
#         "confidence": 0.89,
#         "execution_summary": {
#             "task": "single_image_vqa",
#             "models_used": ["vqa_tool"],
#             "params": {"query": query, "num_images": len(image_ids)}
#         }
#     }

# def run_grounding_tool(query: str, image_ids: List[str], image_store: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """Visual Grounding Model Execution Wrapper"""
#     logger.info(f"Executing grounding_tool for query: '{query}' with image_ids: {image_ids}")
#     bounds = get_image_bounds_or_default(image_ids, image_store)
#     min_lon, min_lat, max_lon, max_lat = bounds

#     b_min_lon = min_lon + (max_lon - min_lon) * 0.2
#     b_max_lon = min_lon + (max_lon - min_lon) * 0.6
#     b_min_lat = min_lat + (max_lat - min_lat) * 0.5
#     b_max_lat = min_lat + (max_lat - min_lat) * 0.9

#     return {
#         "type": "final",
#         "text": f"Visual Grounding: Located object/facility corresponding to '{query}' within bounding coordinates [{b_min_lon:.3f}, {b_min_lat:.3f}, {b_max_lon:.3f}, {b_max_lat:.3f}].",
#         "geojson": {
#             "type": "FeatureCollection",
#             "features": [
#                 {
#                     "type": "Feature",
#                     "geometry": {
#                         "type": "Polygon",
#                         "coordinates": [[
#                             [b_min_lon, b_min_lat],
#                             [b_max_lon, b_min_lat],
#                             [b_max_lon, b_max_lat],
#                             [b_min_lon, b_max_lat],
#                             [b_min_lon, b_min_lat]
#                         ]]
#                     },
#                     "properties": {
#                         "label": f"Grounded Bounding Box: {query}",
#                         "confidence": 0.92
#                     }
#                 }
#             ]
#         },
#         "confidence": 0.92,
#         "execution_summary": {
#             "task": "visual_grounding",
#             "models_used": ["grounding_tool"],
#             "params": {"query": query, "num_images": len(image_ids)}
#         }
#     }

# def run_change_tool(query: str, image_ids: List[str], image_store: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """Bi-temporal Change Detection Model Execution Wrapper"""
#     logger.info(f"Executing change_tool for query: '{query}' with image_ids: {image_ids}")
#     bounds = get_image_bounds_or_default(image_ids, image_store)
#     min_lon, min_lat, max_lon, max_lat = bounds

#     c_min_lon = min_lon + (max_lon - min_lon) * 0.1
#     c_max_lon = min_lon + (max_lon - min_lon) * 0.4
#     c_min_lat = min_lat + (max_lat - min_lat) * 0.1
#     c_max_lat = min_lat + (max_lat - min_lat) * 0.4

#     return {
#         "type": "final",
#         "text": "Bi-Temporal Change Analysis: Detected 14.2% structural expansion between T1 and T2 imagery. Significant land clearing and construction identified.",
#         "geojson": {
#             "type": "FeatureCollection",
#             "features": [
#                 {
#                     "type": "Feature",
#                     "geometry": {
#                         "type": "Polygon",
#                         "coordinates": [[
#                             [c_min_lon, c_min_lat],
#                             [c_max_lon, c_min_lat],
#                             [c_max_lon, c_max_lat],
#                             [c_min_lon, c_max_lat],
#                             [c_min_lon, c_min_lat]
#                         ]]
#                     },
#                     "properties": {
#                         "label": "Bi-temporal Construction Expansion Zone (T1 -> T2)",
#                         "confidence": 0.86
#                     }
#                 }
#             ]
#         },
#         "confidence": 0.86,
#         "execution_summary": {
#             "task": "change_vqa",
#             "models_used": ["change_tool"],
#             "params": {"query": query, "num_images": len(image_ids)}
#         }
#     }

# # ---------------------------------------------------------------------------
# # run_fusion_tool — the composite-generation math below is adapted directly
# # from ML-3's tested fusion_tool.py (SAR VV -> Red, Optical B3 -> Green,
# # Optical B2 -> Blue). I ran ML-3's original version against synthetic
# # .tif files before porting it here, so this math is verified, not guessed.
# # What's still a placeholder: the descriptive `text` field. Once ML-1/ML-3
# # hand off a trained model call, replace the marked block below with a real
# # inference call over `composite_path`.
# # ---------------------------------------------------------------------------

# def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
#     arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
#     if arr_max - arr_min == 0:
#         return np.zeros_like(arr, dtype=np.uint8)
#     scaled = (arr - arr_min) / (arr_max - arr_min) * 255.0
#     return np.clip(scaled, 0, 255).astype(np.uint8)

# def _identify_optical_and_sar(image_ids: List[str], image_store: Dict[str, Any]):
#     """Uses geo_utils' band_count/modality (added in this V2 pass) to tell
#     the optical image apart from the SAR image, regardless of upload order."""
#     optical_id, sar_id = None, None
#     for img_id in image_ids:
#         modality = image_store.get(img_id, {}).get("geo_meta", {}).get("modality")
#         if modality == "sar" and sar_id is None:
#             sar_id = img_id
#         elif modality == "optical" and optical_id is None:
#             optical_id = img_id
#     # Fallback if modality detection is ambiguous: assume upload order (optical first)
#     if optical_id is None and image_ids:
#         optical_id = image_ids[0]
#     if sar_id is None and len(image_ids) > 1:
#         sar_id = image_ids[1]
#     return optical_id, sar_id

# def run_fusion_tool(query: str, image_ids: List[str], image_store: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """Optical-SAR Fusion — real composite generation, text still pending ML model integration."""
#     logger.info(f"Executing fusion_tool for query: '{query}' with image_ids: {image_ids}")
#     bounds = get_image_bounds_or_default(image_ids, image_store)
#     min_lon, min_lat, max_lon, max_lat = bounds

#     composite_path = None
#     if image_store and len(image_ids) >= 2:
#         optical_id, sar_id = _identify_optical_and_sar(image_ids, image_store)
#         try:
#             if optical_id and sar_id:
#                 import rasterio
#                 from PIL import Image

#                 optical_path = image_store[optical_id]["file_path"]
#                 sar_path = image_store[sar_id]["file_path"]

#                 with rasterio.open(sar_path) as sar_src:
#                     sar_vv = sar_src.read(1).astype(np.float32)
#                 with rasterio.open(optical_path) as opt_src:
#                     band3_green = opt_src.read(3).astype(np.float32)
#                     band2_blue = opt_src.read(2).astype(np.float32)

#                 if sar_vv.shape == band3_green.shape == band2_blue.shape:
#                     red = _normalize_to_uint8(sar_vv)
#                     green = _normalize_to_uint8(band3_green)
#                     blue = _normalize_to_uint8(band2_blue)
#                     composite = np.dstack([red, green, blue])

#                     tmp_fd, composite_path = tempfile.mkstemp(suffix=".png", prefix="fusion_")
#                     os.close(tmp_fd)
#                     Image.fromarray(composite, mode="RGB").save(composite_path)
#                     logger.info(f"Fusion composite generated at {composite_path}")
#                 else:
#                     logger.warning(
#                         f"Shape mismatch, skipping real composite — SAR {sar_vv.shape} vs "
#                         f"Optical {band3_green.shape}/{band2_blue.shape}. Resample needed before fusing."
#                     )
#         except Exception as e:
#             logger.warning(f"Real fusion composite generation failed, falling back to placeholder: {e}")

#     # TODO(ML-1 / ML-3): once the fine-tuned fusion/VLM model is ready, replace this
#     # block with: model_output = run_inference(composite_path, query) and use its
#     # real text + confidence instead of the fixed strings below.
#     text = (
#         "Optical-SAR Fusion Analysis: Sentinel-2 optical imagery indicates cloud cover "
#         "obscuring 35% of the coastal zone. Sentinel-1 SAR backscatter pierces cloud cover, "
#         "confirming active maritime vessel presence."
#     )
#     confidence = 0.91

#     return {
#         "type": "final",
#         "text": text,
#         "composite_generated": composite_path is not None,  # honest flag for the frontend/judges
#         "geojson": {
#             "type": "FeatureCollection",
#             "features": [
#                 {
#                     "type": "Feature",
#                     "geometry": {
#                         "type": "Polygon",
#                         "coordinates": [[
#                             [min_lon, min_lat],
#                             [max_lon, min_lat],
#                             [max_lon, max_lat],
#                             [min_lon, max_lat],
#                             [min_lon, min_lat]
#                         ]]
#                     },
#                     "properties": {
#                         "source": "optical",
#                         "label": "Optical (Sentinel-2): Cloud Obscured Sector",
#                         "confidence": 0.75
#                     }
#                 },
#                 {
#                     "type": "Feature",
#                     "geometry": {
#                         "type": "Polygon",
#                         "coordinates": [[
#                             [min_lon + (max_lon - min_lon)*0.1, min_lat + (max_lat - min_lat)*0.1],
#                             [max_lon - (max_lon - min_lon)*0.1, min_lat + (max_lat - min_lat)*0.1],
#                             [max_lon - (max_lon - min_lon)*0.1, max_lat - (max_lat - min_lat)*0.1],
#                             [min_lon + (max_lon - min_lon)*0.1, max_lat - (max_lat - min_lat)*0.1],
#                             [min_lon + (max_lon - min_lon)*0.1, min_lat + (max_lat - min_lat)*0.1]
#                         ]]
#                     },
#                     "properties": {
#                         "source": "sar",
#                         "label": "SAR (Sentinel-1): Cloud-Penetrating Vessel Detection",
#                         "confidence": 0.94
#                     }
#                 }
#             ]
#         },
#         "confidence": confidence,
#         "execution_summary": {
#             "task": "optical_sar_fusion",
#             "models_used": ["fusion_tool"],
#             "params": {"query": query, "num_images": len(image_ids), "composite_path": composite_path}
#         }
#     }

# TOOL_REGISTRY = {
#     "vqa_tool": run_vqa_tool,
#     "grounding_tool": run_grounding_tool,
#     "change_tool": run_change_tool,
#     "fusion_tool": run_fusion_tool
# }

# ── ADD near the top of tools.py, before any function defs ──
import os
import sys

_ML_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_pipelines")
for _sub in ["ml1-vqa-grounding", "ml2-change-detection", "ml3-optical-sar-fusion"]:
    _p = os.path.join(_ML_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_vqa_tool import run_vqa_tool as _ml1_vqa, run_grounding_tool as _ml1_grounding, generate_caption
from run_change_tool import run_change_tool as _ml2_change
from run_fusion_tool import run_fusion_tool as _ml3_fusion
from fusion_tool import normalize_raw_sar_to_uint8


def _identify_optical_and_sar(image_ids, image_store):
    """Uses geo_utils' band_count/modality field (added earlier in geo_utils.py)
    to tell optical apart from SAR regardless of upload order."""
    optical_id, sar_id = None, None
    for img_id in image_ids:
        modality = image_store.get(img_id, {}).get("geo_meta", {}).get("modality")
        if modality == "sar" and sar_id is None:
            sar_id = img_id
        elif modality == "optical" and optical_id is None:
            optical_id = img_id
    if optical_id is None and image_ids:
        optical_id = image_ids[0]
    if sar_id is None and len(image_ids) > 1:
        sar_id = image_ids[1]
    return optical_id, sar_id


# ── REPLACE the four run_*_tool functions with these ──

def run_vqa_tool(query, image_ids, image_store=None):
    return _ml1_vqa(query, image_ids, image_store or {})


def run_grounding_tool(query, image_ids, image_store=None):
    return _ml1_grounding(query, image_ids, image_store or {})


def run_change_tool(query, image_ids, image_store=None):
    if not image_store or len(image_ids) < 2:
        return {
            "type": "final",
            "text": "Change detection requires two images (before and after).",
            "geojson": {"type": "FeatureCollection", "features": []},
            "confidence": 0.0,
            "execution_summary": {"task": "change_vqa", "models_used": [], "params": {"query": query}},
        }
    path1 = image_store[image_ids[0]]["file_path"]
    path2 = image_store[image_ids[1]]["file_path"]
    return _ml2_change(query, path1, path2)


def run_fusion_tool(query, image_ids, image_store=None):
    if not image_store or len(image_ids) < 2:
        return {
            "type": "final",
            "text": "Optical-SAR fusion requires two images (optical and SAR pair).",
            "geojson": {"type": "FeatureCollection", "features": []},
            "confidence": 0.0,
            "execution_summary": {"task": "optical_sar_fusion", "models_used": [], "params": {"query": query}},
        }
    optical_id, sar_id = _identify_optical_and_sar(image_ids, image_store)
    optical_path = image_store[optical_id]["file_path"]
    sar_path = image_store[sar_id]["file_path"]

    result = _ml3_fusion(
        query, optical_path, sar_path,
        sar_vv_band=None, optical_green_band=None, optical_blue_band=None,
        normalize_fn=None,
        vqa_fn=generate_caption,
        static_confidence=0.80,
    )
    return result


TOOL_REGISTRY = {
    "vqa_tool": run_vqa_tool,
    "grounding_tool": run_grounding_tool,
    "change_tool": run_change_tool,
    "fusion_tool": run_fusion_tool
}