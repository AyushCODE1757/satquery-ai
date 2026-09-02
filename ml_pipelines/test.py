import unittest
import tempfile
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from resampler import read_and_resample_band
from bigearthnet_vlm import BigEarthNetVLMDataset

class TestAction3AndDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_band(self, path, width, height, num_bands=1):
        transform = from_origin(10.0, 50.0, 0.0001, 0.0001)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with rasterio.open(
            path, "w", driver="GTiff", width=width, height=height, count=num_bands,
            dtype="float32", crs="EPSG:4326", transform=transform
        ) as dst:
            for b in range(1, num_bands + 1):
                dst.write(np.full((height, width), 1000.0, dtype=np.float32), b)

    def test_action_3_resampling_60m_to_224(self):
        """Tests that a low-res 20x20 band (60m) resamples cleanly to 224x224."""
        tif_path = os.path.join(self.dir_path, "low_res_60m.tif")
        self._create_mock_band(tif_path, width=20, height=20)

        resampled = read_and_resample_band(tif_path, target_shape=(224, 224))
        self.assertEqual(resampled.shape, (224, 224))

    def test_dataset_item_shape(self):
        """Tests end-to-end dataset fetching returns correct shapes."""
        patch_id = "test_patch"
        patch_folder = os.path.join(self.dir_path, patch_id)
        
        opt_path = os.path.join(patch_folder, f"{patch_id}_optical.tif")
        sar_path = os.path.join(patch_folder, f"{patch_id}_sar.tif")

        self._create_mock_band(opt_path, width=120, height=120, num_bands=4)
        self._create_mock_band(sar_path, width=60, height=60, num_bands=1)

        samples = [{"patch_id": patch_id, "input": "Describe terrain", "output": "Forest"}]
        ds = BigEarthNetVLMDataset(samples, dataset_dir=self.dir_path)

        data = ds[0]
        self.assertEqual(data["pixel_values"].size, (224, 224))
        self.assertEqual(data["prompt"], "Describe terrain")

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)