"""
inference_cache.py — pre-computed inference cache for demo reliability.

Replaces the old demo_mode.py concept. When demo_mode is on, we first check
whether the uploaded image(s) + task combination matches a pre-cached,
verified real result. If yes: instant, reliable, and honestly real. If no:
falls through to live tool execution in main.py exactly like the non-demo
path — so a judge uploading their own image still gets a genuine (if slower)
answer, never a hardcoded lie.
"""

import hashlib
import json
import logging
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger("satquery.inference_cache")

CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache_store.json")

def _load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read cache_store.json: {e}")
    return {}

def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write cache_store.json: {e}")

def _file_hash(file_path: str) -> str:
    """Hash actual file bytes, not the opaque upload ID, so the same demo
    image always matches regardless of when/how many times it's uploaded."""
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except Exception as e:
        logger.warning(f"Could not hash file {file_path}: {e}")
        return "unhashable"
    return h.hexdigest()

def compute_cache_key(image_ids: List[str], task: str, image_store: Dict[str, Any]) -> str:
    file_hashes = []
    for img_id in image_ids:
        file_path = image_store.get(img_id, {}).get("file_path")
        file_hashes.append(_file_hash(file_path) if file_path else "missing")
    combined = task + "|" + "|".join(sorted(file_hashes))
    return hashlib.sha256(combined.encode()).hexdigest()

def get_cached_or_none(image_ids: List[str], task: str, image_store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not image_ids:
        return None
    cache = _load_cache()
    key = compute_cache_key(image_ids, task, image_store)
    hit = cache.get(key)
    if hit:
        logger.info(f"Inference cache HIT for task={task}, key={key[:12]}...")
    return hit

def store_result(image_ids: List[str], task: str, image_store: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Call this once, offline, after running real inference on your 3-5
    curated demo images — NOT during the live pitch. See usage note below."""
    cache = _load_cache()
    key = compute_cache_key(image_ids, task, image_store)
    cache[key] = payload
    _save_cache(cache)
    logger.info(f"Cached real inference result for task={task}, key={key[:12]}...")


# -----------------------------------------------------------------------
# USAGE (run this manually, once, before the pitch, after ML models are
# actually ready — this is NOT called automatically anywhere):
#
#   from inference_cache import store_result
#   from tools import TOOL_REGISTRY
#
#   image_store = {...}          # populate with your 3-5 curated demo images
#   image_ids = [...]
#   task = "change_vqa"
#   real_payload = TOOL_REGISTRY["change_tool"]("What changed?", image_ids, image_store)
#   store_result(image_ids, task, image_store, real_payload)
#
# Repeat for each curated image / task pair you want pitch-day-safe.
# -----------------------------------------------------------------------