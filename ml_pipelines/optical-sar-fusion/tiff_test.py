import rasterio
import numpy as np

tif_path = r"C:/Users/aayus/OneDrive/Desktop/New folder/ml_pipelines/demo_data/bhoomi/sar/S1D_IW_GRDH_1SDV_20260901T005430_20260901T005455_004377_008165_5094.SAFE/measurement/s1d-iw-grd-vv-20260901t005430-20260901t005455-004377-008165-001.tiff"

with rasterio.open(tif_path) as src:
    data = src.read(1)

print("Shape:", data.shape)
print("Min:", data.min())
print("Max:", data.max())
print("Mean:", data.mean())
print("Std:", data.std())

print("\nPercentiles:")
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f"{p:>3}%:", np.percentile(data, p))

print("\nUnique sample values:", np.unique(data[:1000, :1000])[:20])