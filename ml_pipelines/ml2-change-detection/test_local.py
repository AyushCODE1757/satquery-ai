"""Tests run_change_tool.py against two real local images. Needs two
co-registered .tif files — even small synthetic ones work for a shape check."""

from run_change_tool import run_change_tool
from shared.contract_validator import validate_contract

if __name__ == "__main__":
    result = run_change_tool(
        query="What changed between these two dates?",
        image1_path="sample_data/before.tif",   # replace with a real local test pair
        image2_path="sample_data/after.tif",
    )
    print(result)
    assert validate_contract(result) == []
    assert result["type"] == "final"
    assert result["geojson"]["type"] == "FeatureCollection"
    assert "execution_summary" in result
    print("\nContract check passed.")