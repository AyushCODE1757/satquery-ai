"""One shared check all three ML people can import into their test_local.py,
so nobody accidentally ships a payload FS-2's main.py will silently drop."""

REQUIRED_KEYS = {"type", "text", "geojson", "confidence", "execution_summary"}
REQUIRED_SUMMARY_KEYS = {"task", "models_used", "params"}

def validate_contract(payload: dict) -> list:
    errors = []
    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        errors.append(f"Missing top-level keys: {missing}")
    if payload.get("type") != "final":
        errors.append(f"'type' must be 'final', got: {payload.get('type')}")
    summary = payload.get("execution_summary", {})
    missing_summary = REQUIRED_SUMMARY_KEYS - summary.keys()
    if missing_summary:
        errors.append(f"Missing execution_summary keys: {missing_summary}")
    if payload.get("geojson", {}).get("type") != "FeatureCollection":
        errors.append("geojson must be a FeatureCollection")
    return errors