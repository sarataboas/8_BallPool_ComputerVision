"""
Comparison-only model architectures for Task 2 experiments.
The winning architecture is promoted to models/cnn_pipeline.py.
"""

import torch.nn as nn
import torchvision.models as tv_models


def get_model(backbone: str, head: str, num_classes: int = 17) -> nn.Module:
    """
    Factory that returns a model given a backbone name and head type.

    Args:
        backbone:    "resnet18" | "resnet34" | "scratch"
        head:        "regression" (single float output) |
                     "classification" (num_classes outputs, counts 0-16)
        num_classes: number of output classes (only used when head="classification")
    """
    pass
