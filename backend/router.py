import os
import logging
import re
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
    Uses NVIDIA ChatNVIDIA LLM router if API key is provided, falling back to deterministic keyword routing.
    """
    # Try NVIDIA / LangChain LLM Classification if API key exists
    if os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"):
        try:
            llm_result = classify_with_langchain(query, image_ids)
            if llm_result in VALID_TASKS:
                logger.info(f"LLM router classified task: {llm_result}")
                return llm_result
        except Exception as e:
            logger.warning(f"LLM router fallback due to error: {e}")

    # Deterministic Heuristic Router
    return classify_heuristically(query, image_ids)

def classify_heuristically(query: str, image_ids: Optional[List[str]] = None) -> str:
    """Deterministic rule-based keyword & modality classifier"""
    query_lower = query.lower()
    num_images = len(image_ids) if image_ids else 0

    # 1. Optical-SAR Fusion detection
    if any(k in query_lower for k in ["sar", "radar", "sentinel-1", "cloud", "fusion", "all-weather", "pierce cloud"]):
        return "optical_sar_fusion"

    # 2. Bi-temporal Change Detection
    if any(k in query_lower for k in ["change", "temporal", "before", "after", "difference", "t1", "t2", "growth", "expansion", "new building", "demolished", "construction"]):
        return "change_vqa"
    
    if num_images == 2 and not any(k in query_lower for k in ["sar", "radar"]):
        return "change_vqa"

    # 3. Visual Grounding (Bounding Box / Object Localization)
    if any(k in query_lower for k in ["where", "locate", "ground", "box", "highlight", "find", "detect position", "coordinates of", "bounding box"]):
        return "visual_grounding"

    # 4. Default: Single Image VQA
    return "single_image_vqa"

def classify_with_langchain(query: str, image_ids: Optional[List[str]]) -> Optional[str]:
    """
    NVIDIA Nemotron LLM Router implementation using langchain_nvidia_ai_endpoints.ChatNVIDIA.
    Falls back to OpenAI / generic LangChain LLM if NVIDIA key is absent.
    """
    system_prompt = """You are an ISRO satellite remote-sensing agent router.
Classify the user query into EXACTLY ONE of these task categories:
- optical_sar_fusion: when user query mentions SAR radar, Sentinel-1, cloud penetration, or multi-modal fusion.
- change_vqa: when user query asks to detect changes, urban growth, or compare before/after (T1 vs T2) images.
- visual_grounding: when user query asks to locate, find, highlight, or draw a bounding box around objects/facilities.
- single_image_vqa: for general questions on satellite imagery, counting objects, or land use.

Output ONLY the exact category string name. Do not output markdown, reasoning, or extra words."""

    user_msg = f"User Query: \"{query}\"\nNumber of uploaded images: {len(image_ids) if image_ids else 0}\nCategory:"

    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_api_key and not nvidia_api_key.startswith("your_"):
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            client = ChatNVIDIA(
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                api_key=nvidia_api_key,
                temperature=0.0,
                top_p=0.95,
                max_tokens=64,
                model_kwargs={
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            response = client.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ])
            
            raw_text = response.content if isinstance(response.content, str) else str(response.content)
            raw_text = raw_text.strip().lower()

            # Match tasks in order of specificity
            for task in ["optical_sar_fusion", "change_vqa", "visual_grounding", "single_image_vqa"]:
                if task in raw_text:
                    return task
        except Exception as e:
            logger.warning(f"ChatNVIDIA classification failed: {e}")

    # Fallback to OpenAI if OPENAI_API_KEY is present
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            from langchain.prompts import PromptTemplate

            prompt = PromptTemplate.from_template(prompt_str)
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            chain = prompt | llm
            response = chain.invoke({"query": query, "num_images": len(image_ids) if image_ids else 0})
            result = response.content.strip().lower()
            return result if result in VALID_TASKS else None
        except Exception as e:
            logger.debug(f"OpenAI fallback classification exception: {e}")

    return None

def get_tool_for_task(task: str) -> str:
    """Maps task vocabulary to specialist ML tool ID"""
    mapping = {
        "single_image_vqa": "vqa_tool",
        "visual_grounding": "grounding_tool",
        "change_vqa": "change_tool",
        "optical_sar_fusion": "fusion_tool"
    }
    return mapping.get(task, "vqa_tool")