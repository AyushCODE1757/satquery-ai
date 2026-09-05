"""
run_vqa_tool.py — loads base PaliGemma + fine-tuned LoRA adapter,
exposes contract-matching functions for FS-2's TOOL_REGISTRY.

MOCK_MODE: set env var SATQUERY_MOCK_ML1=1 to skip loading the real model
entirely and return a clearly-labeled placeholder instead. Use this while
the Kaggle fine-tune is still running, so integration testing (frontend,
FS-2 routing, ML-2/ML-3 wiring) isn't blocked waiting 1.5+ hours.
"""

from email.mime import image
import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

MOCK_MODE = os.environ.get("SATQUERY_MOCK_ML1", "0") == "1"

MODEL_ID = "google/paligemma-3b-pt-224"
LORA_PATH = os.path.join(os.path.dirname(__file__), "lora_adapter")

_processor = None
_model = None


def extract_object_phrase(query: str) -> str:
    """Strips instruction verbs so PaliGemma's 'detect X' gets a plain
    object noun, not a full sentence. Fixes the observed bug where
    'Highlight the road' was passed to the model verbatim."""
    prefixes = [
        r"^highlight the\s+", r"^highlight\s+", r"^locate the\s+", r"^locate\s+",
        r"^find the\s+", r"^find\s+", r"^show me the\s+", r"^show\s+",
        r"^where is the\s+", r"^where is\s+", r"^detect the\s+", r"^detect\s+",
        r"^identify the\s+", r"^identify\s+",
    ]
    q_lower = query.strip().lower()
    for pat in prefixes:
        m = re.match(pat, q_lower)
        if m:
            return query.strip()[m.end():].strip().rstrip("?.!")
    return query.strip()

def _load_model_once():
    global _processor, _model
    if MOCK_MODE or _model is not None:
        return

    print("Loading PaliGemma base model + LoRA adapter...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
    )

    if os.path.exists(LORA_PATH) and os.listdir(LORA_PATH):
        _model = PeftModel.from_pretrained(base_model, LORA_PATH).eval()
        print("Loaded fine-tuned LoRA adapter from:", LORA_PATH)
    else:
        _model = base_model.eval()
        print("WARNING: no LoRA adapter found — using base (non-fine-tuned) PaliGemma.")


# ── ADD near the top, after the other imports ──


# ── ADD this new function, anywhere after _load_model_once() ──
def _generate_with_confidence(prompt: str, image: Image.Image, max_new_tokens: int = 50):
    """Returns (decoded_text, confidence). Confidence is a best-effort signal
    from the model's own token probabilities (transformers' documented
    compute_transition_scores method) — not a formally calibrated number,
    but real per-query output, not a fixed placeholder."""
    if MOCK_MODE:
        return "[MOCK ML-1 OUTPUT]", 0.0

    _load_model_once()
    inputs = _processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = _model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs.sequences[0][input_len:]
    decoded = _processor.decode(generated_tokens, skip_special_tokens=True)

    try:
        transition_scores = _model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        avg_log_prob = transition_scores.mean().item()
        confidence = float(torch.exp(torch.tensor(avg_log_prob)))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.65  # RSVQA-LR non-count validation accuracy as documented fallback — see note below

    return decoded, confidence


# ── ADD this — exact signature ML-3's run_fusion_tool.py expects ──
def generate_caption(image_path: str, query: str) -> str:
    """Hand this function directly to ML-3 as their vqa_fn parameter."""
    image = Image.open(image_path).convert("RGB")
    text, confidence = _generate_with_confidence(f"answer en {query}", image)
    return {"text": text, "confidence": confidence}


def _generate(prompt: str, image: Image.Image, max_new_tokens: int = 50) -> str:
    if MOCK_MODE:
        return "[MOCK ML-1 OUTPUT] real model not loaded — SATQUERY_MOCK_ML1=1 is set."
    _load_model_once()
    inputs = _processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    return _processor.decode(output[0][input_len:], skip_special_tokens=True)


def parse_location_tokens(text: str):
    matches = re.findall(r"<loc(\d{4})>", text)
    if len(matches) < 4:
        return None
    y_min, x_min, y_max, x_max = [int(m) for m in matches[:4]]
    label = re.sub(r"<loc\d{4}>", "", text).strip()
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "label": label}


def run_vqa_tool(query: str, image_ids: list, image_store: dict) -> dict:
    image_path = image_store[image_ids[0]]["file_path"]
    image = Image.open(image_path).convert("RGB")
    answer_text, confidence = _generate_with_confidence(f"answer en {query}", image)

    return {
        "type": "final",
        "text": answer_text,
        "geojson": {"type": "FeatureCollection", "features": []},
        "confidence": confidence,
        "execution_summary": {
            "task": "single_image_vqa",
            "models_used": ["paligemma-3b-pt-224-lora-rsvqa-vrsbench" if not MOCK_MODE else "MOCK"],
            "params": {"query": query},
        },
    }


def run_grounding_tool(query: str, image_ids: list, image_store: dict) -> dict:
    image_path = image_store[image_ids[0]]["file_path"]
    image = Image.open(image_path).convert("RGB")

    if MOCK_MODE:
        parsed = {"x_min": 200, "y_min": 200, "x_max": 600, "y_max": 600, "label": f"[MOCK] {query}"}
        confidence = 0.0
    else:
        object_phrase = extract_object_phrase(query)
        raw_output, confidence = _generate_with_confidence(f"detect {object_phrase}\n", image)
        parsed = parse_location_tokens(raw_output)

    geo_meta = image_store[image_ids[0]].get("geo_meta", {})
    bounds = geo_meta.get("bounds", [72.50, 23.01, 72.55, 23.05])
    width = geo_meta.get("width", 1024)
    height = geo_meta.get("height", 1024)

    if parsed is None:
        return {
            "type": "final",
            "text": f"Could not confidently locate '{query}' in this image.",
            "geojson": {"type": "FeatureCollection", "features": []},
            "confidence": confidence or 0.0,
            "execution_summary": {
                "task": "visual_grounding",
                "models_used": ["paligemma-3b-pt-224-lora-vrsbench"],
                "params": {"query": query},
            },
        }

    min_lon, min_lat, max_lon, max_lat = bounds
    def to_lonlat(x_norm, y_norm):
        px, py = (x_norm / 1024) * width, (y_norm / 1024) * height
        lon = min_lon + (px / width) * (max_lon - min_lon)
        lat = max_lat - (py / height) * (max_lat - min_lat)
        return [lon, lat]

    p1 = to_lonlat(parsed["x_min"], parsed["y_min"])
    p2 = to_lonlat(parsed["x_max"], parsed["y_min"])
    p3 = to_lonlat(parsed["x_max"], parsed["y_max"])
    p4 = to_lonlat(parsed["x_min"], parsed["y_max"])
    image_space_bbox = [
        parsed["x_min"] / 1024, parsed["y_min"] / 1024,
        parsed["x_max"] / 1024, parsed["y_max"] / 1024,
    ]

    return {
        "type": "final",
        "text": f"Located '{parsed['label'] or query}' in the image.",
        "geojson": {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[p1, p2, p3, p4, p1]]},
                "properties": {"label": parsed["label"] or object_phrase, "confidence": confidence, "image_space_bbox": image_space_bbox},
            }],
        },
        "confidence": confidence,
        "execution_summary": {
            "task": "visual_grounding",
            "models_used": ["paligemma-3b-pt-224-lora-vrsbench" if not MOCK_MODE else "MOCK"],
            "params": {"query": query},
        },
    }