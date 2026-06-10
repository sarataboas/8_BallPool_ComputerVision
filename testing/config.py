import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Task 1
DEVELOPMENT_SET = PROJECT_ROOT / "development_set"
INPUT_JSON      = PROJECT_ROOT / "example_json" / "input.json"
OUTPUT_EXAMPLE  = PROJECT_ROOT / "example_json" / "output_example.json"

# Task 2 & 3
DATASET_DIR    = PROJECT_ROOT / "8-Ball Pool.v3i.yolov11"
TRAIN_DIR      = DATASET_DIR / "train"
VALID_DIR      = DATASET_DIR / "valid"
TEST_DIR       = DATASET_DIR / "test"
DATA_YAML      = DATASET_DIR / "data.yaml"
MODELS_DIR      = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "testing" / "experiments"

# Task 2 — default experiment config (override per run)
DEFAULT_CFG = {
    "name":        "unnamed",
    "backbone":    "resnet18",
    "head":        "regression",   # "regression" | "classification"
    "input_size":  384,
    "lr":          1e-4,
    "epochs":      30,
    "batch_size":  16,
    "augment":     False,
    "crop_table":  False,          # full-frame vs table-crop ablation
    "weights_dir": str(MODELS_DIR),
}