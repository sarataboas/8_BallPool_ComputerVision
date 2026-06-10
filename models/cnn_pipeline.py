"""
Task 2 — Ball Count Prediction pipeline (DELIVERABLE).

Standalone script: no imports from testing/.
All paths are passed as arguments.

Input JSON format:
    {"image_path": ["path/to/img1.jpg", ...]}

Output JSON format:
    [{"image_path": "path/to/img1.jpg", "num_balls": 10}, ...]

Usage:
    python models/cnn_pipeline.py --input input.json --output output.json --weights models/best.pth
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def count_balls_from_label(label_path: Path) -> int:
    """Ball count = annotations where class_id != 2 (Dot = rail marker)."""
    pass


class PoolBallDataset(Dataset):
    """
    Loads images from a split directory and returns (image_tensor, num_balls).
    If labels_dir is None, returns (image_tensor, image_path) for inference.
    """
    def __init__(self, images_dir: Path, labels_dir: Path | None = None,
                 transform=None):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx):
        pass


def build_loaders(train_dir: Path, valid_dir: Path, cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Builds train and validation DataLoaders from the given split directories."""
    pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BallCounterCNN(nn.Module):
    """
    Winning architecture (to be filled after experiments in testing/task2/).
    Placeholder until the best config is determined.
    """
    def __init__(self, head: str = "regression"):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_model(model: nn.Module, train_loader: DataLoader,
                valid_loader: DataLoader, cfg: dict) -> dict:
    """
    Trains the model for cfg["epochs"] epochs.
    Returns history dict: {"train_loss": [...], "val_loss": [...]}
    """
    pass


def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    """
    Runs inference on loader and returns metrics dict:
        {"mae": float, "acc": float, "off1": float,
         "gt": np.ndarray, "pred": np.ndarray, "paths": list}
    """
    pass


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(input_json: str, output_json: str, weights: str) -> None:
    """
    Loads weights, runs inference on images listed in input_json,
    writes predictions to output_json.
    """
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 — Ball count inference")
    parser.add_argument("--input",   required=True, help="Path to input JSON")
    parser.add_argument("--output",  required=True, help="Path to output JSON")
    parser.add_argument("--weights", required=True, help="Path to .pth weights file")
    args = parser.parse_args()

    run_inference(args.input, args.output, args.weights)
