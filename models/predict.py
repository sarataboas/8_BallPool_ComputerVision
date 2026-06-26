"""
Task 2 — Ball Count Prediction (DELIVERABLE).

Standalone inference script: loads the trained CNN weights, reads an input
JSON listing image paths, and writes predicted ball counts to an output JSON.
Contains only what inference needs (model + transform + TTA loop) — no
training/evaluation code.

Input JSON:  {"image_path": ["path/to/img1.jpg", ...]}
Output JSON: [{"image_path": "path/to/img1.jpg", "num_balls": 10}, ...]

Usage:
    python models/predict.py --input models/input.json --output models/output.json --weights models/best.pth
"""

import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv_models

_INFER_SIZE = 384  # matches training input size of the winning model


class BallCounterCNN(nn.Module):
    """
    Winning architecture: ResNet18 backbone + regression head.
    Attribute named 'backbone' to match state-dict keys saved by training.
    weights=None here: the full state dict (backbone included) is loaded
    from disk right after construction, so downloading ImageNet weights
    just to overwrite them would be a pointless network dependency.
    """
    def __init__(self):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).flatten(1)
        return self.head(x)


def get_inference_transform(size: int = _INFER_SIZE) -> T.Compose:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return T.Compose([
        T.ToPILImage(),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def run_inference(input_json: str, output_json: str, weights: str) -> None:
    with open(input_json) as f:
        data = json.load(f)
    image_paths = data.get("image_path") or data.get("image_paths", [])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = BallCounterCNN()
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    model.to(device)

    transform = get_inference_transform()

    results = []
    with torch.no_grad():
        for path_str in image_paths:
            img = cv2.cvtColor(cv2.imread(str(path_str)), cv2.COLOR_BGR2RGB)
            img_t      = transform(img).unsqueeze(0).to(device)
            img_t_flip = torch.flip(img_t, dims=[3])   # horizontal flip

            # TTA: average raw outputs over original + flipped image before rounding.
            # Horizontal flip is valid because table orientation is arbitrary and the
            # model was trained with random horizontal flip.
            raw = (model(img_t) + model(img_t_flip)) / 2.0
            count = int(np.clip(round(raw.squeeze().item()), 0, 16))
            results.append({"image_path": path_str, "num_balls": count})

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 — Ball count inference (deliverable)")
    parser.add_argument("--input",   default="models/input.json", help="Path to input JSON")
    parser.add_argument("--output",  default="models/output.json", help="Path to output JSON")
    parser.add_argument("--weights", default="models/best.pth", help="Path to .pth weights file")
    args = parser.parse_args()
    run_inference(args.input, args.output, args.weights)
