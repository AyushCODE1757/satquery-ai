import sys
import os

# Add backend directory to module search path
sys.path.append(os.path.dirname(__file__))

from router import classify_task, VALID_TASKS
from fallback_payloads import FALLBACK_PAYLOADS
from geo_utils import get_default_bounds

def run_tests():
    print("--- Running SatQuery Backend Verification ---")

    # 1. Test Router Classification
    test_cases = [
        ("What solar panels are present?", [], "single_image_vqa"),
        ("Where is the industrial storage building?", [], "visual_grounding"),
        ("Detect new urban construction development between T1 and T2", ["img_1", "img_2"], "change_vqa"),
        ("Analyze cloud obscured region using Sentinel-1 SAR radar", ["img_1"], "optical_sar_fusion"),
    ]

    for query, images, expected in test_cases:
        task = classify_task(query, images)
        print(f"Query: '{query}' -> Task: '{task}' (Expected: '{expected}')")
        assert task == expected, f"Expected {expected}, got {task}"

    print("\n[OK] Task Router verification passed!")

    # 2. Test Fallback Payloads
    print("\nVerifying Fallback Payloads...")
    for task_name in VALID_TASKS:
        assert task_name in FALLBACK_PAYLOADS, f"Missing payload for {task_name}"
        payload = FALLBACK_PAYLOADS[task_name]
        assert payload["type"] == "final", "Payload type must be 'final'"
        assert "text" in payload and "geojson" in payload and "confidence" in payload
        assert payload["execution_summary"]["task"] == task_name
        print(f"[OK] Validated fallback payload for task: {task_name}")

    # 3. Test Geo Utils Fallback Bounds
    bounds = get_default_bounds()
    assert bounds["crs"] == "EPSG:4326"
    assert len(bounds["bounds"]) == 4
    print("\n[OK] GeoUtils bounds extraction passed!")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()