import rasterio
from rasterio.enums import Resampling
import numpy as np

def read_and_resample_band(tif_path: str, band_index: int = 1, target_shape: tuple = (224, 224)) -> np.ndarray:
    """
    Action 3: Reads any Sentinel-2 band (10m, 20m, 60m) and resamples 
    it directly to target_shape (224, 224) using bilinear interpolation.
    """
    with rasterio.open(tif_path) as src:
        # out_shape resamples on the fly during read, saving memory
        resampled_band = src.read(
            band_index,
            out_shape=target_shape,
            resampling=Resampling.bilinear
        ).astype(np.float32)
        
    return resampled_band