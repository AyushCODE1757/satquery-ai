import os
import numpy as np
import rasterio

DB_MIN = -25.0
DB_MAX = 0.0

def clip_and_normalize_sar(sar_array: np.ndarray, db_min: float = DB_MIN, db_max: float = DB_MAX) -> np.ndarray:
    denom = db_max - db_min
    if denom <= 0:
        return np.zeros_like(sar_array, dtype=np.uint8)

    # Clean NaNs prior to clipping
    sar_clean = np.nan_to_num(sar_array, nan=db_min)
    clipped = np.clip(sar_clean, db_min, db_max)
    normalized = (clipped - db_min) / denom * 255.0
    return normalized.astype(np.uint8)

def process_sar_file(input_tif_path: str, output_tif_path: str) -> None:
    with rasterio.open(input_tif_path) as src:
        profile = src.profile.copy()
        band_count = src.count

        processed_bands = []
        for band_idx in range(1, band_count + 1):
            raw_band = src.read(band_idx).astype(np.float32)
            processed_bands.append(clip_and_normalize_sar(raw_band))

    profile.update(dtype=rasterio.uint8, count=band_count, nodata=None)

    os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)
    with rasterio.open(output_tif_path, "w", **profile) as dst:
        for i, band in enumerate(processed_bands, start=1):
            dst.write(band, i)

def find_sar_files(input_dir: str, extensions: tuple = (".tif", ".tiff")) -> list:
    matches = []
    for root, _dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(extensions):
                matches.append(os.path.join(root, f))
    return sorted(matches)

def batch_process_sar_folder(
    input_dir: str,
    output_dir: str,
    limit: int = None,
    flatten_output: bool = False,  # Default to False to prevent overwrites
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    input_paths = find_sar_files(input_dir)

    if not input_paths:
        print(f"No .tif/.tiff files found under {input_dir}")
        return

    if limit is not None:
        input_paths = input_paths[:limit]

    print(f"Found {len(input_paths)} SAR files. Normalizing...")

    for i, input_path in enumerate(input_paths, start=1):
        if flatten_output:
            # Append parent folder to ensure unique filenames when flattened
            parent_dir = os.path.basename(os.path.dirname(input_path))
            out_name = f"{parent_dir}_{os.path.basename(input_path)}"
            output_path = os.path.join(output_dir, out_name)
        else:
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)

        try:
            process_sar_file(input_path, output_path)
        except Exception as e:
            print(f"[{i}/{len(input_paths)}] FAILED on {inputA_path}: {e}")