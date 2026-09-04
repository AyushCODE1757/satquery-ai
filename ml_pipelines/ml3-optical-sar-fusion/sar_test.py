import os
import rasterio
import numpy as np

measurement_dir = r"C:\Users\aayus\OneDrive\Desktop\New folder\ml_pipelines\demo_data\bhoomi\sar\S1D_IW_GRDH_1SDV_20260901T005430_20260901T005455_004377_008165_5094.SAFE\measurement"

vv_path = [
    os.path.join(measurement_dir, f)
    for f in os.listdir(measurement_dir)
    if "-vv-" in f.lower()
][0]

vh_path = [
    os.path.join(measurement_dir, f)
    for f in os.listdir(measurement_dir)
    if "-vh-" in f.lower()
][0]


# --- sanity check: VV ---
with rasterio.open(vv_path) as src:
    print("VV shape:", src.shape)
    print("VV dtype:", src.dtypes)
    print("VV CRS  :", src.crs)
    print("VV bounds:", src.bounds)

    vv = src.read(1)

    print("VV min :", np.nanmin(vv))
    print("VV max :", np.nanmax(vv))
    print("VV mean:", np.nanmean(vv))


# --- sanity check: VH ---
with rasterio.open(vh_path) as src:
    print("\nVH shape:", src.shape)
    print("VH dtype:", src.dtypes)
    print("VH CRS  :", src.crs)
    print("VH bounds:", src.bounds)

    vh = src.read(1)

    print("VH min :", np.nanmin(vh))
    print("VH max :", np.nanmax(vh))
    print("VH mean:", np.nanmean(vh))


# --- crop a small window from the center ---
crop_size = 512

with rasterio.open(vv_path) as src:
    full_h, full_w = src.shape

    row_off = (full_h - crop_size) // 2
    col_off = (full_w - crop_size) // 2

    window = rasterio.windows.Window(
        col_off,
        row_off,
        crop_size,
        crop_size
    )

    vv_crop = src.read(1, window=window)

with rasterio.open(vh_path) as src:
    vh_crop = src.read(1, window=window)


print("\nCropped VV shape:", vv_crop.shape)
print("Cropped VH shape:", vh_crop.shape)


# --- save VV + VH crop as a small GeoTIFF ---
with rasterio.open(vv_path) as src:
    profile = src.profile.copy()
    profile.update({
        "height": crop_size,
        "width": crop_size,
        "transform": src.window_transform(window),
        "count": 2
    })

out_path = "sar_crop.tif"

with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(vv_crop, 1)   # Band 1 = VV
    dst.write(vh_crop, 2)   # Band 2 = VH

print("Saved SAR crop to:", out_path)