"""
ML-3 Action 1 (MVP Logic): fusion_tool

Reads an Optical .tif and a SAR .tif, maps:
    SAR VV band     -> Red
    Optical Band 3  -> Green
    Optical Band 2  -> Blue
and writes a false-color composite .png to a temp file for the VLM to read.

Drop this function into backend/tools.py so FS-2's router.py can call it
as one of the 4 forced tool choices (vqa_tool, grounding_tool, change_tool, fusion_tool).
"""

import os
import tempfile
import numpy as np
import rasterio
from PIL import Image


def fusion_tool(optical_path: str, sar_path: str) -> dict:
    """
    Fuse an optical .tif and a SAR .tif into a false-color composite PNG.

    Args:
        optical_path: path to optical .tif (must contain at least Bands 2 and 3)
        sar_path: path to SAR .tif (must contain a VV band)

    Returns:
        dict with:
            "image_path": path to the saved .png composite
            "shape": (H, W) of the output
    """
    # --- Read SAR (VV band -> Red) ---
    with rasterio.open(sar_path) as sar_src:
        # Assumes band 1 is VV. If the file has VV/VH stacked, adjust index here.
        sar_vv = sar_src.read(1).astype(np.float32)

    # --- Read Optical (Band 3 -> Green, Band 2 -> Blue) ---
    with rasterio.open(optical_path) as opt_src:
        band3_green = opt_src.read(3).astype(np.float32)
        band2_blue = opt_src.read(2).astype(np.float32)

    # --- Sanity check: shapes must match before stacking ---
    if not (sar_vv.shape == band3_green.shape == band2_blue.shape):
        raise ValueError(
            f"Shape mismatch — SAR {sar_vv.shape}, "
            f"Optical B3 {band3_green.shape}, Optical B2 {band2_blue.shape}. "
            f"Resample to a common grid before fusing."
        )

    # --- Normalize each band independently to 0-255 uint8 ---
    def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
        if arr_max - arr_min == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr - arr_min) / (arr_max - arr_min) * 255.0
        return np.clip(scaled, 0, 255).astype(np.uint8)

    red = normalize_to_uint8(sar_vv)
    green = normalize_to_uint8(band3_green)
    blue = normalize_to_uint8(band2_blue)

    # --- Stack into HxWx3 RGB composite ---
    composite = np.dstack([red, green, blue])

    # --- Save to a temp .png for the VLM to read ---
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="fusion_")
    os.close(tmp_fd)
    Image.fromarray(composite, mode="RGB").save(tmp_path)

    return {
        "image_path": tmp_path,
        "shape": composite.shape[:2],
    }


if __name__ == "__main__":
    # Quick local smoke test — replace with real file paths
    result = fusion_tool("sample_optical.tif", "sample_sar.tif")
    print(result)