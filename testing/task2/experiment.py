"""
Runs a single Task 2 experiment end-to-end:
    build model → train → evaluate → save .pth → append row to results.csv
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import csv
from typing import Any

from testing.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    MODELS_DIR, EXPERIMENTS_DIR,
    DEFAULT_CFG,
)
from testing.task2.architectures import get_model

# cnn_pipeline is the deliverable — import only the reusable pieces
from models.cnn_pipeline import PoolBallDataset, build_loaders, train_model, evaluate


RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
CSV_FIELDS  = [
    "name", "backbone", "head", "input_size", "lr", "epochs",
    "batch_size", "augment", "crop_table",
    "val_mae", "val_acc", "val_off1",
    "test_mae", "test_acc", "test_off1",
]


def run_experiment(cfg: dict[str, Any]) -> tuple[dict, dict]:
    """
    Runs one experiment with the given config (merged over DEFAULT_CFG).

    Returns:
        results: dict with val/test metrics
        history: dict with per-epoch train/val loss lists
    """
    pass


def _append_results_csv(row: dict) -> None:
    """Appends one result row to the persistent CSV log."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
