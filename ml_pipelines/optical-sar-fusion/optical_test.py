import os
import rasterio
from rasterio.windows import Window
img_data = r"C:\Users\aayus\OneDrive\Desktop\New folder\ml_pipelines\demo_data\bhoomi\S2C_MSIL2A_20260831T052641_N0512_R105_T43QCA_20260831T102215.SAFE\GRANULE\L2A_T43QCA_A010371_20260831T053602\IMG_DATA\R10m"
b02_path = [f for f in os.listdir(img_data) if "B02" in f][0]
b03_path = [f for f in os.listdir(img_data) if "B03" in f][0]

b02_path = os.path.join(img_data, b02_path)
b03_path = os.path.join(img_data, b03_path)


# --- sanity check: shape, dtype, CRS ---
with rasterio.open(b03_path) as src:
    print("B03 shape:", src.shape)
    print("B03 dtype:", src.dtypes)
    print("B03 CRS  :", src.crs)
    full_h, full_w = src.shape

    # --- crop a small window from roughly the center ---
    crop_size = 512  # pixels, adjust as needed
    row_off = (full_h - crop_size) // 2
    col_off = (full_w - crop_size) // 2
    window = Window(col_off, row_off, crop_size, crop_size)

    b3_crop = src.read(1, window=window)

with rasterio.open(b02_path) as src:
    b2_crop = src.read(1, window=window)

print("Cropped B03 shape:", b3_crop.shape)
print("Cropped B02 shape:", b2_crop.shape)

# --- optional: save the crop as its own small GeoTIFF for reuse ---
with rasterio.open(b03_path) as src:
    profile = src.profile.copy()
    profile.update({
        "height": crop_size,
        "width": crop_size,
        "transform": src.window_transform(window)
    })

out_path = "optical_crop.tif"
with rasterio.open(out_path, "w", **{**profile, "count": 2}) as dst:
    dst.write(b3_crop, 1)  # band 1 = Green (B03) in this new file
    dst.write(b2_crop, 2)  # band 2 = Blue (B02)
print("Saved crop to:", out_path)