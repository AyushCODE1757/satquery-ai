# BEN-14K Narendra Aironi repackaging:
# empirical/statistical inspection indicates Band 1 = VH, Band 2 = VV.
# Therefore Band 2 is used as VV for BigEarthNet.
#
# Native Bhoonidhi Sentinel-1 VV TIFF: single-band file, use Band 1.
#
# optical_crop.tif (Bhoonidhi demo crop): 2-band file,
# Band 1 = Green (B03), Band 2 = Blue (B02) — NOT native S2 numbering.

import os
import tempfile

import numpy as np
import rasterio
from PIL import Image

from resampler import read_and_resample_band
from rasterio.warp import transform_bounds


def normalize_raw_sar_to_uint8(
    arr: np.ndarray,
    low_pct: float = 2,
    high_pct: float = 98
) -> np.ndarray:
    """
    Normalize raw Sentinel-1 SAR DN values to uint8
    using percentile stretching. Use for native/demo SAR products.
    """
    lo, hi = np.percentile(arr, [low_pct, high_pct])

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    scaled = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def normalize_ben_sar_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalize BigEarthNet SAR VV dB values (-25 dB to 0 dB) into uint8.
    """
    clipped = np.clip(arr, -25.0, 0.0)
    return ((clipped + 25.0) / 25.0 * 255.0).astype(np.uint8)


def normalize_optical_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an optical band independently to uint8 (min-max stretch).
    """
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - arr_min) / (arr_max - arr_min) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

def get_wgs84_bounds(tif_path: str) -> list:
    """
    Get raster bounds in WGS84 (EPSG:4326).

    Returns:
        [min_lon, min_lat, max_lon, max_lat]
    """
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            return [72.50, 23.01, 72.55, 23.05]

        try:
            bounds = src.bounds
            wgs84_bounds = transform_bounds(
                src.crs,
                "EPSG:4326",
                *bounds
            )
            return list(wgs84_bounds)
        except Exception:
            return [72.50, 23.01, 72.55, 23.05]


def fusion_tool(
    optical_path: str,
    sar_path: str,
    sar_vv_band: int = None,
    optical_green_band: int = None,
    optical_blue_band: int = None,
    normalize_fn=None,
) -> dict:
    """
    Fuse an optical TIFF and a SAR TIFF into a false-color composite PNG.

    Args:
        optical_path: Path to optical TIFF.
        sar_path: Path to SAR TIFF containing the VV band.
        sar_vv_band: Band index containing SAR VV (auto-detected if None).
        optical_green_band: Band index for Green (auto-detected if None).
        optical_blue_band: Band index for Blue (auto-detected if None).
        normalize_fn: Function used to normalize the SAR VV array (auto-detected if None).

    Returns:
        dict with "image_path" (saved PNG) and "shape" (H, W).
    """

    # --- Read optical bands (Green, Blue) ---
    with rasterio.open(optical_path) as opt_src:
        if optical_green_band is None or optical_green_band > opt_src.count:
            optical_green_band = 3 if opt_src.count >= 3 else 1
        if optical_blue_band is None or optical_blue_band > opt_src.count:
            optical_blue_band = 2 if opt_src.count >= 2 else 1

        band3_green = opt_src.read(optical_green_band).astype(np.float32)
        band2_blue = opt_src.read(optical_blue_band).astype(np.float32)

    # --- Read SAR VV ---
    with rasterio.open(sar_path) as sar_src:
        if sar_vv_band is None or sar_vv_band > sar_src.count or sar_vv_band < 1:
            sar_vv_band = 2 if sar_src.count >= 2 else 1
        sar_vv = sar_src.read(sar_vv_band).astype(np.float32)

    # --- Use optical Green band as target shape ---
    target_shape = band3_green.shape

    # --- Resample SAR if its shape doesn't match ---
    if sar_vv.shape != target_shape:
        sar_vv = read_and_resample_band(
            sar_path,
            band_index=sar_vv_band,
            target_shape=target_shape,
        )

    # --- Resample optical Blue band if its shape doesn't match ---
    if band2_blue.shape != target_shape:
        band2_blue = read_and_resample_band(
            optical_path,
            band_index=optical_blue_band,
            target_shape=target_shape,
        )

    # --- Final shape sanity check ---
    if not (sar_vv.shape == band3_green.shape == band2_blue.shape):
        raise ValueError(
            "Shape mismatch after resampling — "
            f"SAR {sar_vv.shape}, "
            f"Optical Green {band3_green.shape}, "
            f"Optical Blue {band2_blue.shape}."
        )

    # --- Auto-detect SAR normalization fn if not provided ---
    if normalize_fn is None:
        if np.any(sar_vv < 0):
            normalize_fn = normalize_ben_sar_to_uint8
        else:
            normalize_fn = normalize_raw_sar_to_uint8

    # --- Normalize (SAR uses caller/auto-detected fn; optical always min-max) ---
    red = normalize_fn(sar_vv)
    green = normalize_optical_to_uint8(band3_green)
    blue = normalize_optical_to_uint8(band2_blue)

    # --- Stack into H x W x 3 RGB composite ---
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

    bounds = get_wgs84_bounds("optical_crop.tif")

    print("WGS84 bounds:", bounds)