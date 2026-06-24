"""
Comparison-only model architectures for Task 2 experiments.
The winning architecture is promoted to models/cnn_pipeline.BallCounterCNN after experiments.
"""

import torch.nn as nn
import torchvision.models as tv_models


class _ScratchCNN(nn.Module):
    """Small CNN trained from scratch. Regularized baseline to quantify transfer learning value."""
    out_features = 256

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.net(x).flatten(1)   # (B, 256)


class _PoolModel(nn.Module):
    """Generic wrapper: backbone → flatten → head."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() > 2:
            x = x.flatten(1)
        return self.head(x)


def get_model(backbone: str, head: str, num_classes: int = 17) -> nn.Module:
    """
    Factory that returns a model for a given backbone and head type.

    Args:
        backbone:    "resnet18" | "resnet18_nopretrain" | "resnet34" | "scratch"
        head:        "regression"      → single float output
                     "classification"  → num_classes outputs (counts 0-16)
        num_classes: number of output classes (used only when head="classification")
    """
    if backbone == "resnet18":
        base        = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        features    = nn.Sequential(*list(base.children())[:-1])   # (B, 512, 1, 1)
        in_features = 512
        dropout     = 0.2
    elif backbone == "resnet18_nopretrain":
        base        = tv_models.resnet18(weights=None)
        features    = nn.Sequential(*list(base.children())[:-1])   # (B, 512, 1, 1)
        in_features = 512
        dropout     = 0.2
    elif backbone == "resnet34":
        base        = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
        features    = nn.Sequential(*list(base.children())[:-1])   # (B, 512, 1, 1)
        in_features = 512
        dropout     = 0.2
    elif backbone == "efficientnet_b0":
        base        = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()   # strip head; forward now returns (B, 1280)
        features    = base
        in_features = 1280
        dropout     = 0.2
    elif backbone == "scratch":
        features    = _ScratchCNN()
        in_features = _ScratchCNN.out_features
        dropout     = 0.4   # heavier regularization for scratch model
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")

    if head == "regression":
        head_module = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    elif head == "classification":
        head_module = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f"Unknown head: {head!r}")

    return _PoolModel(features, head_module)
