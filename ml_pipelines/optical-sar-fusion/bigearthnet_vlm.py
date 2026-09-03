import torch
from torch.utils.data import Dataset

from PIL import Image

import numpy as np
import rasterio
from rasterio.enums import Resampling
import os
import pandas as pd

def build_samples(
    dataset_root: str,
    split: str,
    metadata_path: str = None
) -> list:
    """
    Build verified S1 + S2 samples for a BigEarthNet split.

    Only samples with both physical S1 and S2 files are included.
    """

    if metadata_path is None:
        metadata_path = os.path.join(
            dataset_root,
            "metadata.parquet"
        )

    s1_dir = os.path.join(
        dataset_root,
        "BigEarthNet-S1",
        split
    )

    s2_dir = os.path.join(
        dataset_root,
        "BigEarthNet-S2",
        split
    )

    # ---------------------------------------------------------
    # Load metadata
    # ---------------------------------------------------------

    df = pd.read_parquet(metadata_path)

    df = df[df["split"] == split].copy()

    # ---------------------------------------------------------
    # Build S1 lookup
    # ---------------------------------------------------------

    s1_files = {}

    for filename in os.listdir(s1_dir):

        if filename.lower().endswith(".tif"):

            basename = os.path.splitext(filename)[0]

            s1_files[basename] = os.path.join(
                s1_dir,
                filename
            )

    # ---------------------------------------------------------
    # Build S2 lookup
    #
    # Metadata:
    # S2B_MSIL2A_20170825T093029_26_57
    #
    # Physical:
    # S2B_MSIL2A_20170825T093029_N9999_R136_T34TEQ_26_57.tif
    #
    # Match using acquisition + final row/column.
    # ---------------------------------------------------------

    s2_files = {}

    for filename in os.listdir(s2_dir):

        if filename.lower().endswith(".tif"):

            basename = os.path.splitext(filename)[0]
            parts = basename.split("_")

            if len(parts) >= 5:

                key = (
                    "_".join(parts[:3])
                    + "_"
                    + "_".join(parts[-2:])
                )

                s2_files[key] = os.path.join(
                    s2_dir,
                    filename
                )

    # ---------------------------------------------------------
    # Resolve verified pairs
    # ---------------------------------------------------------

    samples = []

    for _, row in df.iterrows():

        s1_name = row["s1_name"]
        s2_name = row["s2v1_name"]

        # S1 exact match
        s1_path = s1_files.get(s1_name)

        if s1_path is None:
            continue

        # S2 acquisition + row/column key
        s2_parts = s2_name.split("_")

        if len(s2_parts) < 5:
            continue

        s2_key = (
            "_".join(s2_parts[:3])
            + "_"
            + "_".join(s2_parts[-2:])
        )

        s2_path = s2_files.get(s2_key)

        if s2_path is None:
            continue

        samples.append({
            "patch_id": row["patch_id"],
            "s1_path": s1_path,
            "s2_path": s2_path,
            "input": "",
            "output": row["labels"]
        })

    print(f"{split}: {len(samples)} verified pairs")

    return samples


class BigEarthNetVLMDataset(Dataset):
    """
    Unified BigEarthNet VLM Pipeline

    Action 1:
        SAR VV (Red) + Optical B03 (Green) + Optical B02 (Blue)

    Action 2:
        SAR -25 dB to 0 dB normalization -> uint8

    Action 3:
        Resample SAR and optical bands to uniform 224x224

    Expected sample format:
        {
            "patch_id": ...,
            "s1_path": ...,
            "s2_path": ...,
            "input": ...,
            "output": ...
        }
    """

    def __init__(self, samples: list, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        s1_path = sample["s1_path"]
        s2_path = sample["s2_path"]

        # =====================================================
        # ACTION 1 + ACTION 3
        # SAR VV
        #
        # BEN-14K Kaggle repackaging:
        # Band 1 = VH
        # Band 2 = VV
        #
        # Resample directly to 224x224
        # =====================================================

        with rasterio.open(s1_path) as src_sar:
            sar_vv = src_sar.read(
                2,
                out_shape=(224, 224),
                resampling=Resampling.bilinear
            ).astype(np.float32)

        # =====================================================
        # ACTION 1 + ACTION 3
        # Sentinel-2 optical bands
        #
        # Band 3 = B03 (Green)
        # Band 2 = B02 (Blue)
        #
        # Resample directly to 224x224
        # =====================================================

        with rasterio.open(s2_path) as src_opt:
            opt_b3 = src_opt.read(
                3,
                out_shape=(224, 224),
                resampling=Resampling.bilinear
            ).astype(np.float32)

            opt_b2 = src_opt.read(
                2,
                out_shape=(224, 224),
                resampling=Resampling.bilinear
            ).astype(np.float32)

        # =====================================================
        # ACTION 2
        # SAR clipping and normalization
        #
        # Input range:
        #     -25 dB to 0 dB
        #
        # Output:
        #     uint8 [0, 255]
        # =====================================================

        sar_clipped = np.clip(
            sar_vv,
            -25.0,
            0.0
        )

        sar_8bit = (
            (sar_clipped + 25.0)
            / 25.0
            * 255.0
        ).astype(np.uint8)

        # =====================================================
        # Optical reflectance scaling
        #
        # Sentinel-2 reflectance:
        #     0 -> 10000
        #
        # Convert to uint8 [0, 255]
        # =====================================================

        opt_b3_8bit = np.clip(
            (opt_b3 / 10000.0) * 255.0,
            0,
            255
        ).astype(np.uint8)

        opt_b2_8bit = np.clip(
            (opt_b2 / 10000.0) * 255.0,
            0,
            255
        ).astype(np.uint8)

        # =====================================================
        # ACTION 1
        #
        # Red   = SAR VV
        # Green = Sentinel-2 B03
        # Blue  = Sentinel-2 B02
        # =====================================================

        rgb_composite = np.dstack(
            (
                sar_8bit,
                opt_b3_8bit,
                opt_b2_8bit
            )
        )

        # =====================================================
        # Convert fused image to PIL
        # =====================================================

        pil_image = Image.fromarray(rgb_composite)

        # =====================================================
        # Apply optional VLM transform
        # =====================================================

        if self.transform:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = pil_image

        # =====================================================
        # Return VLM sample
        # =====================================================

        return {
            "pixel_values": image_tensor,
            "prompt": sample.get("input", ""),
            "target": sample.get("output", "")
        }