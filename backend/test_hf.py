import os
from huggingface_hub import InferenceClient

MODEL_ID = "google/paligemma-3b-pt-224"

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"]
)

print("HF authentication loaded.")
print("Model:", MODEL_ID)