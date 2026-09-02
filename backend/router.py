import logging
from typing import List, Optional

logger = logging.getLogger("satquery.router")

# Valid FS-1 / FS-2 task vocabulary
VALID_TASKS = [
    "single_image_vqa",
    "visual_grounding",
    "change_vqa",
    "optical_sar_fusion"
]

def classify_task(query: str, image_ids: Optional[List[str]] = None) -> str:
    """
    Classifies the user query and image context into one of the 4 specialist remote sensing tasks.
    Can be powered by LangChain or deterministic keyword routing.
    """
    query_lower = query.lower()
    num_images = len(image_ids) if image_ids else 0

    # 1. Optical-SAR Fusion detection
    if any(k in query_lower for k in ["sar", "radar", "sentinel-1", "cloud", "fusion", "all-weather"]):
        return "optical_sar_fusion"

    # 2. Bi-temporal Change Detection
    if any(k in query_lower for k in ["change", "temporal", "before", "after", "difference", "t1", "t2", "growth", "expansion", "new building", "demolished"]):
        return "change_vqa"
    
    if num_images == 2 and not any(k in query_lower for k in ["sar", "radar"]):
        return "change_vqa"

    # 3. Visual Grounding (Bounding Box / Object Localization)
    if any(k in query_lower for k in ["where", "locate", "ground", "box", "highlight", "find", "detect position", "coordinates of"]):
        return "visual_grounding"

    # 4. Default: Single Image VQA
    return "single_image_vqa"

def get_tool_for_task(task: str) -> str:
    """Maps task vocabulary to specialist ML tool ID"""
    mapping = {
        "single_image_vqa": "vqa_tool",
        "visual_grounding": "grounding_tool",
        "change_vqa": "change_tool",
        "optical_sar_fusion": "fusion_tool"
    }
    return mapping.get(task, "vqa_tool")
