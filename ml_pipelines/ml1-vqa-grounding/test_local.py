"""Run with SATQUERY_MOCK_ML1=1 set, right now, before training finishes.
Re-run without that env var once lora_adapter/ is populated."""

import os
from run_vqa_tool import run_vqa_tool, run_grounding_tool

TEST_IMAGE_STORE = {
    "img_test1": {
        "file_path": "sample_data/test1.jpg",  # put any test image here
        "geo_meta": {"bounds": [72.50, 23.01, 72.55, 23.05], "width": 1024, "height": 1024},
    }
}

if __name__ == "__main__":
    print("MOCK_MODE:", os.environ.get("SATQUERY_MOCK_ML1"))

    vqa_result = run_vqa_tool("What is visible in this image?", ["img_test1"], TEST_IMAGE_STORE)
    print("\n--- VQA RESULT ---")
    print(vqa_result)
    assert vqa_result["type"] == "final"
    assert "text" in vqa_result

    grounding_result = run_grounding_tool("railway track", ["img_test1"], TEST_IMAGE_STORE)
    print("\n--- GROUNDING RESULT ---")
    print(grounding_result)
    assert grounding_result["type"] == "final"

    print("\nAll contract checks passed.")