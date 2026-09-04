"""
bbox_utils.py — corrected PaliGemma bounding-box conversion.
Fixes 3 verified bugs from the original VRSBench preprocessing notebook:
  1. Scale: 1024 bins (0-1023), not 1000
  2. Order: y_min, x_min, y_max, x_max (PaliGemma's real token order)
  3. Format: four separate <locNNNN> tokens, not one {<..>} group
"""

def extract_bbox(obj_corner):
    """Unchanged from ML-1's original — converts 4 corner points to axis-aligned bbox."""
    xs = obj_corner[0::2]
    ys = obj_corner[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_bbox_paligemma(bbox, scale=1024):
    """FIXED: scale=1024 (was 1000). Values must land in [0, 1023]."""
    x_min, y_min, x_max, y_max = bbox
    normalized = [round(v * scale) for v in (x_min, y_min, x_max, y_max)]
    return [max(0, min(scale - 1, v)) for v in normalized]


def bbox_to_location_tokens(bbox_1024):
    """
    FIXED: real PaliGemma format — four separate <locNNNN> tokens,
    4-digit zero-padded, in y_min, x_min, y_max, x_max order.
    """
    x_min, y_min, x_max, y_max = bbox_1024
    return f"<loc{y_min:04d}><loc{x_min:04d}><loc{y_max:04d}><loc{x_max:04d}>"


def parse_location_tokens(token_string):
    """
    Inverse of the above — parses PaliGemma's real output string back into
    a pixel/normalized bbox. Use this on the MODEL'S OUTPUT during inference
    (not just for preprocessing training data).
    Expected input like: "<loc0390><loc0023><loc0678><loc0327> railway track"
    """
    import re
    matches = re.findall(r"<loc(\d{4})>", token_string)
    if len(matches) < 4:
        return None
    y_min, x_min, y_max, x_max = [int(m) for m in matches[:4]]
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "scale": 1024}