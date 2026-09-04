"""
paligemma_inference.py — SatQuery PaliGemma inference.

Base model:
    google/paligemma-3b-pt-224

SatQuery LoRA adapter:
    Gojo67868/satquery-paligemma

The LoRA adapter is hosted on Hugging Face.

The runtime must have:
    - access to the gated PaliGemma base model
    - Hugging Face authentication
    - CUDA-capable GPU
"""

import torch
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

from peft import PeftModel


BASE_MODEL = "google/paligemma-3b-pt-224"
ADAPTER_MODEL = "Gojo67868/satquery-paligemma"


_model = None
_processor = None


def load_model():
    """
    Load PaliGemma base model and SatQuery LoRA adapter.

    The model is cached after the first load.
    """

    global _model
    global _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PaliGemma inference requires a CUDA-capable GPU "
            "for the current 4-bit NF4 configuration."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading PaliGemma base model...")

    base_model = (
        PaliGemmaForConditionalGeneration
        .from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
        )
    )

    print("Loading SatQuery LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_MODEL,
    )

    model.eval()

    processor = PaliGemmaProcessor.from_pretrained(
        BASE_MODEL,
    )

    _model = model
    _processor = processor

    print(
        "PaliGemma + SatQuery adapter loaded successfully."
    )

    return _model, _processor


def generate_caption(
    image_path: str,
    query: str,
) -> dict:
    """
    Run PaliGemma inference.

    Args:
        image_path:
            Input image path.

        query:
            VQA/query prompt.

    Returns:
        {
            "text": generated answer,
            "confidence": average probability
                         of generated tokens
        }
    """

    model, processor = load_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        text=query,
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
        if torch.is_tensor(value)
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=64,
            output_scores=True,
            return_dict_in_generate=True,
        )

    output_ids = output.sequences

    prompt_len = inputs[
        "input_ids"
    ].shape[1]

    generated_ids = output_ids[
        0
    ][prompt_len:]

    generated_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    # ---------------------------------------------------------
    # Token-level confidence
    # ---------------------------------------------------------

    token_confidences = []

    for step, scores in enumerate(
        output.scores
    ):

        if step >= len(generated_ids):
            break

        probs = torch.softmax(
            scores[0],
            dim=-1,
        )

        token_id = generated_ids[
            step
        ]

        token_probability = probs[
            token_id
        ].item()

        token_confidences.append(
            token_probability
        )

    confidence = (
        sum(token_confidences)
        / len(token_confidences)
        if token_confidences
        else 0.0
    )

    return {
        "text": generated_text,
        "confidence": float(confidence),
    }