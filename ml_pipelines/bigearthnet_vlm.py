import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import rasterio
from rasterio.enums import Resampling

class BigEarthNetVLMDataset(Dataset):
    """
    Unified Pipeline combining:
    - Action 1: SAR VV (Red) + Opt B3 (Green) + Opt B2 (Blue) composite
    - Action 2: SAR -25dB to 0dB normalization -> uint8
    - Action 3: Resampling all optical/SAR bands to uniform 224x224
    """
    def __init__(self, samples: list, dataset_dir: str, transform=None):
        """
        samples: list of dicts with keys ['patch_id', 'input', 'output']
        """
        self.samples = samples
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patch_id = sample['patch_id']
        
        opt_path = os.path.join(self.dataset_dir, patch_id, f"{patch_id}_optical.tif")
        sar_path = os.path.join(self.dataset_dir, patch_id, f"{patch_id}_sar.tif")

        # Action 3 & 1: Resample SAR VV band to 224x224
        with rasterio.open(sar_path) as src_sar:
            sar_vv = src_sar.read(1, out_shape=(224, 224), resampling=Resampling.bilinear).astype(np.float32)

        # Action 3 & 1: Resample Optical B3 (Green) & B2 (Blue) to 224x224
        with rasterio.open(opt_path) as src_opt:
            opt_b3 = src_opt.read(3, out_shape=(224, 224), resampling=Resampling.bilinear).astype(np.float32)
            opt_b2 = src_opt.read(2, out_shape=(224, 224), resampling=Resampling.bilinear).astype(np.float32)

        # Action 2: SAR Clip (-25dB to 0dB) & uint8 Scaling
        sar_clipped = np.clip(sar_vv, -25.0, 0.0)
        sar_8bit = ((sar_clipped - (-25.0)) / 25.0 * 255.0).astype(np.uint8)

        # Optical reflectance scaling to uint8 (0-10000 -> 0-255)
        opt_b3_8bit = np.clip((opt_b3 / 10000.0) * 255.0, 0, 255).astype(np.uint8)
        opt_b2_8bit = np.clip((opt_b2 / 10000.0) * 255.0, 0, 255).astype(np.uint8)

        # Stack Red=SAR, Green=B3, Blue=B2
        rgb_composite = np.dstack((sar_8bit, opt_b3_8bit, opt_b2_8bit))
        pil_image = Image.fromarray(rgb_composite, mode="RGB")

        if self.transform:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = pil_image

        return {
            "pixel_values": image_tensor,
            "prompt": sample.get("input", ""),
            "target": sample.get("output", "")
        }