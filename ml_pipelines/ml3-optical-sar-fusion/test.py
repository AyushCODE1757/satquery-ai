from bigearthnet_vlm import (
    build_samples,
    BigEarthNetVLMDataset
)


# Change this if your dataset is mounted somewhere else.
DATASET_ROOT = "/kaggle/input/datasets/narendraaironi/bigearthnet-14k/BEN_14k"


# ---------------------------------------------------------
# Build samples
# ---------------------------------------------------------

train_samples = build_samples(
    DATASET_ROOT,
    "train"
)

val_samples = build_samples(
    DATASET_ROOT,
    "validation"
)

test_samples = build_samples(
    DATASET_ROOT,
    "test"
)


# ---------------------------------------------------------
# Create datasets
# ---------------------------------------------------------

train_dataset = BigEarthNetVLMDataset(
    train_samples
)

val_dataset = BigEarthNetVLMDataset(
    val_samples
)

test_dataset = BigEarthNetVLMDataset(
    test_samples
)


# ---------------------------------------------------------
# Verify sizes
# ---------------------------------------------------------

print("\n========== DATASET SIZES ==========")

print("Train:", len(train_dataset))
print("Val  :", len(val_dataset))
print("Test :", len(test_dataset))


# ---------------------------------------------------------
# Test one sample
# ---------------------------------------------------------

print("\n========== SAMPLE TEST ==========")

sample = train_dataset[0]

print("Keys:", sample.keys())
print("Image type:", type(sample["pixel_values"]))

print(
    "Image size:",
    sample["pixel_values"].size
)

print(
    "Prompt:",
    sample["prompt"]
)

print(
    "Target:",
    sample["target"]
)


print("\n========== TEST PASSED ==========")