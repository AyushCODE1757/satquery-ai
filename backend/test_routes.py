"""
Route testing script for SatQuery AI FastAPI backend.
Tests /api/upload and /api/query (SSE streaming) with demo_mode = True and False.
"""

import sys
import os
import json

# Add backend directory to module search path
sys.path.append(os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_routes():
    print("--- Testing SatQuery AI API Routes ---")

    # 1. Test Root and Health Endpoints
    res_root = client.get("/")
    assert res_root.status_code == 200, f"Root endpoint failed: {res_root.status_code}"
    print("[OK] GET / ->", res_root.json())

    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health endpoint failed: {res_health.status_code}"
    print("[OK] GET /api/health ->", res_health.json())

    # 2. Test Image Upload Endpoint (/api/upload)
    dummy_file_content = b"DUMMY_TIFF_IMAGE_CONTENT_FOR_TESTING"
    files = [
        ("images", ("test_sat1.tif", dummy_file_content, "image/tiff")),
        ("images", ("test_sat2.tif", dummy_file_content, "image/tiff"))
    ]
    res_upload = client.post("/api/upload", files=files)
    assert res_upload.status_code == 200, f"Upload endpoint failed: {res_upload.status_code}"
    upload_data = res_upload.json()
    assert "image_ids" in upload_data, "Response missing 'image_ids'"
    image_ids = upload_data["image_ids"]
    assert len(image_ids) == 2, f"Expected 2 image_ids, got {len(image_ids)}"
    print(f"[OK] POST /api/upload -> Returned image_ids: {image_ids}")

    # 3. Test GET /api/images/{image_id}
    res_info = client.get(f"/api/images/{image_ids[0]}")
    assert res_info.status_code == 200
    print(f"[OK] GET /api/images/{image_ids[0]} -> Metadata loaded")

    # 4. Test Query Endpoint (/api/query) - SSE Stream (demo_mode = True)
    query_payload_demo = {
        "query": "Detect bi-temporal change between T1 and T2",
        "image_ids": image_ids,
        "demo_mode": True
    }
    res_query_demo = client.post("/api/query", json=query_payload_demo)
    assert res_query_demo.status_code == 200, f"Query endpoint failed: {res_query_demo.status_code}"
    
    body_text = res_query_demo.text
    assert "data: " in body_text, "SSE stream response missing 'data: ' prefix"
    assert '"type": "trace"' in body_text, "SSE stream missing trace events"
    assert '"type": "final"' in body_text, "SSE stream missing final event"
    print("[OK] POST /api/query (demo_mode=True) -> Streamed trace & final fallback payload successfully")

    # 5. Test Query Endpoint (/api/query) - SSE Stream (demo_mode = False)
    query_payload_real = {
        "query": "Analyze cloud obscured region using Sentinel-1 SAR radar",
        "image_ids": [image_ids[0]],
        "demo_mode": False
    }
    res_query_real = client.post("/api/query", json=query_payload_real)
    assert res_query_real.status_code == 200
    body_real = res_query_real.text
    assert '"task": "optical_sar_fusion"' in body_real
    print("[OK] POST /api/query (demo_mode=False) -> Streamed trace & model inference payload successfully")

    print("\nALL ROUTE VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_routes()
