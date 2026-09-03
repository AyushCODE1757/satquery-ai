# BEN-14K Narendra Aironi repackaging:
# empirical/statistical inspection indicates Band 1 = VH, Band 2 = VV.
# Therefore Band 2 is used as VV.

import os
import tempfile
import numpy as np
import rasterio
from PIL import Image
from resampler import read_and_resample_band


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
    # BEN-14K: Band 1 = VH, Band 2 = VV
    with rasterio.open(sar_path) as sar_src:
        sar_vv = sar_src.read(2).astype(np.float32)

    # --- Read Optical (Band 3 -> Green, Band 2 -> Blue) ---
    with rasterio.open(optical_path) as opt_src:
        band3_green = opt_src.read(3).astype(np.float32)
        band2_blue = opt_src.read(2).astype(np.float32)

    # --- Use optical Band 3 as the target shape ---
    target_shape = band3_green.shape

    # --- Resample SAR if its shape doesn't match ---
    if sar_vv.shape != target_shape:
        sar_vv = read_and_resample_band(
            sar_path,
            band_index=2,
            target_shape=target_shape
        )

    # --- Resample optical Band 2 if its shape doesn't match ---
    if band2_blue.shape != target_shape:
        band2_blue = read_and_resample_band(
            optical_path,
            band_index=2,
            target_shape=target_shape
    )

    # --- Final sanity check ---
    if not (sar_vv.shape == band3_green.shape == band2_blue.shape):
        raise ValueError(
            f"Shape mismatch after resampling — "
        f"SAR {sar_vv.shape}, "
        f"Optical B3 {band3_green.shape}, "
        f"Optical B2 {band2_blue.shape}."
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