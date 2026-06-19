"""
Task 2 — Ball Count Prediction pipeline (DELIVERABLE).

Standalone: no imports from testing/.
All paths are passed as arguments.

Input JSON:  {"image_path": ["path/to/img1.jpg", ...]}
Output JSON: [{"image_path": "path/to/img1.jpg", "num_balls": 10}, ...]

Usage:
    python models/cnn_pipeline.py --input input.json --output output.json --weights models/best.pth
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tv_models

_INFER_SIZE = 384  # matches training input size of the winning model


# ---------------------------------------------------------------------------
# Table crop preprocessing
# ---------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top, bottom = y_sorted[:2], y_sorted[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_table_roi(bgr: np.ndarray) -> np.ndarray:
    """
    Detects the table cloth, finds 4 corners, warps to a top-down view.
    Falls back to the original image if detection fails at any step.
    Adapted from image_processing_pipeline.py.
    """
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Sample cloth colour from image center
        center = hsv[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        h_vals, s_vals, v_vals = (center[:, :, c].reshape(-1) for c in range(3))
        valid = (s_vals > 50) & (v_vals > 50)
        if valid.sum() < 50:
            return bgr

        h_med = int(np.median(h_vals[valid]))
        mask = cv2.inRange(hsv,
                           np.array([max(0, h_med - 18), 70, 70]),
                           np.array([min(179, h_med + 18), 255, 255]))
        mask[:int(0.2 * h), :] = 0
        mask[int(0.9 * h):, :] = 0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        # Largest component closest to center
        n, labels, _, centroids = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return bgr
        cp = np.array([w / 2, h / 2])
        best = min(range(1, n), key=lambda i: np.linalg.norm(centroids[i] - cp))
        comp = np.uint8(labels == best) * 255

        # Contour → 4 corners
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return bgr
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        peri = cv2.arcLength(hull, True)
        corners = None
        for eps in [0.01, 0.02, 0.03]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                break
        if corners is None:
            corners = cv2.boxPoints(cv2.minAreaRect(hull))
        corners = _order_points(corners)

        # Reject tiny detections
        cx, cy = corners[:, 0], corners[:, 1]
        if 0.5 * abs(np.dot(cx, np.roll(cy, -1)) - np.dot(cy, np.roll(cx, -1))) < 1000:
            return bgr

        # Expand corners outward by 25px
        mcx, mcy = np.mean(cx), np.mean(cy)
        exp = []
        for px, py in corners:
            dx, dy = px - mcx, py - mcy
            n_ = np.sqrt(dx**2 + dy**2) + 1e-6
            exp.append([px + 25 * dx / n_, py + 25 * dy / n_])
        pts = np.array(exp, dtype=np.float32)

        # Warp to top-down, 2:1 pool table aspect ratio
        tl, tr, br, bl = pts
        max_w = int((np.linalg.norm(br - bl) + np.linalg.norm(tr - tl)) / 2)
        max_h = int(max_w * 2.0)
        max_w, max_h = int(max_w * 2.0), int(max_h * 2.0)
        dst = np.array([[0, 0], [max_w-1, 0], [max_w-1, max_h-1], [0, max_h-1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(bgr, M, (max_w, max_h), flags=cv2.INTER_CUBIC)

    except Exception:
        return bgr


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def count_balls_from_label(label_path: Path) -> int:
    """Ball count = annotation lines where class_id != 2 (Dot = rail marker)."""
    path = Path(label_path)
    if not path.exists():
        return 0
    lines = path.read_text().splitlines()
    return sum(1 for ln in lines if ln.strip() and not ln.startswith("2 "))


def get_transform(cfg: dict, train: bool) -> T.Compose:
    """Returns the torchvision transform pipeline for a given config and phase."""
    size = cfg["input_size"]
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    steps = [T.ToPILImage(), T.Resize((size, size))]

    if train:
        aug = cfg.get("augment", "none")
        if aug == "light":
            steps += [
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        elif aug == "heavy":
            steps += [
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]
        elif aug == "heavy_motivated":
            # Justified per domain analysis (Run 8):
            # - rotation ±10°: closes gap with external data's Roboflow ±15° rotation
            # - b/c/s=0.3: broadcast (vivid, high contrast) vs phone camera (flat, muted)
            # - hue=0.15: forces colour-invariant features; prevents table-colour shortcuts
            steps += [
                T.RandomHorizontalFlip(),
                T.RandomRotation(degrees=10, fill=0),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            ]

    steps += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(steps)


class PoolBallDataset(Dataset):
    """
    Loads images from split_dir/images/ and labels from split_dir/labels/.
    Returns (img_tensor, count_float, path_str) for each sample.
    count is -1 when labels_dir is None (inference mode).
    """
    def __init__(self, images_dir: Path, labels_dir: Optional[Path] = None,
                 transform=None, crop_table: bool = False):
        self.images     = sorted(Path(images_dir).glob("*.jpg"))
        self.labels_dir = Path(labels_dir) if labels_dir is not None else None
        self.transform  = transform
        self.crop_table = crop_table

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        bgr = cv2.imread(str(img_path))
        if self.crop_table:
            bgr = crop_table_roi(bgr)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)

        if self.labels_dir is not None:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            count = float(count_balls_from_label(label_path))
        else:
            count = -1.0

        return img, count, str(img_path)


def build_loader(split_dir: Path, cfg: dict, train: bool = False) -> DataLoader:
    """Builds a DataLoader for a single split directory."""
    transform = get_transform(cfg, train=train)
    ds = PoolBallDataset(
        images_dir=Path(split_dir) / "images",
        labels_dir=Path(split_dir) / "labels",
        transform=transform,
        crop_table=cfg.get("crop_table", False),
    )
    g = torch.Generator().manual_seed(cfg.get("seed", 42))
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=train,
        num_workers=0,
        generator=g if train else None,
    )


def build_loaders(train_dir: Path, valid_dir: Path,
                  cfg: dict) -> Tuple[DataLoader, DataLoader]:
    return build_loader(train_dir, cfg, train=True), build_loader(valid_dir, cfg, train=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BallCounterCNN(nn.Module):
    """
    Winning architecture — updated to final config after experiments.
    
    """
    def __init__(self):
        super().__init__()
        base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_model(model: nn.Module, train_loader: DataLoader,
                valid_loader: DataLoader, cfg: dict) -> dict:
    """
    Trains model, checkpoints on best valid MAE, saves weights.
    Returns history: {"train_loss": [...], "val_mae": [...]}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    head   = cfg.get("head", "regression")

    if head == "classification":
        criterion = nn.CrossEntropyLoss()
    elif cfg.get("loss", "smooth_l1") == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.get("lr_patience", 7), factor=0.5
    )

    history      = {"train_loss": [], "val_mae": []}
    best_val_mae = float("inf")
    best_state   = None
    weights_dir  = Path(cfg.get("weights_dir", "."))
    es_patience  = cfg.get("early_stopping_patience", None)
    epochs_no_improve = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0

        for imgs, counts, _ in train_loader:
            imgs = imgs.to(device)
            if head == "classification":
                targets = counts.long().to(device)
                loss = criterion(model(imgs), targets)
            else:
                targets = counts.float().unsqueeze(1).to(device)
                loss = criterion(model(imgs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)

        train_loss = total_loss / len(train_loader.dataset)
        val_mae    = evaluate(model, valid_loader, device)["mae"]
        scheduler.step(val_mae)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{cfg['epochs']}  "
                  f"train_loss={train_loss:.4f}  val_mae={val_mae:.4f}")

        if es_patience and epochs_no_improve >= es_patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {es_patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    weights_path = weights_dir / f"{cfg['name']}.pth"
    torch.save(model.state_dict(), str(weights_path))
    print(f"Best val_mae={best_val_mae:.4f}  →  saved {weights_path.name}")

    return history


def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    """
    Runs inference on a labeled DataLoader.
    Returns: {mae, acc, off1, gt (np.ndarray), pred (np.ndarray), paths (list)}.
    """
    model.eval()
    gt_list, pred_list, path_list = [], [], []

    with torch.no_grad():
        for imgs, counts, paths in loader:
            imgs = imgs.to(device)
            out  = model(imgs)

            if out.shape[-1] > 1:          # classification head
                preds = out.argmax(dim=1).float().cpu().numpy()
            else:                          # regression head
                preds = out.squeeze(1).cpu().numpy()

            preds = np.clip(np.round(preds), 0, 16).astype(int)
            gt    = counts.numpy().astype(int)

            gt_list.extend(gt.tolist())
            pred_list.extend(preds.tolist())
            path_list.extend(list(paths))

    gt   = np.array(gt_list)
    pred = np.array(pred_list)

    return {
        "mae":   float(np.mean(np.abs(pred - gt))),
        "acc":   float(np.mean(pred == gt)),
        "off1":  float(np.mean(np.abs(pred - gt) <= 1)),
        "gt":    gt,
        "pred":  pred,
        "paths": path_list,
    }


# ---------------------------------------------------------------------------
# Inference (deliverable entry point)
# ---------------------------------------------------------------------------

def run_inference(input_json: str, output_json: str, weights: str) -> None:
    with open(input_json) as f:
        data = json.load(f)
    image_paths = data.get("image_path") or data.get("image_paths", [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BallCounterCNN()
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    model.to(device)

    infer_cfg = {"input_size": _INFER_SIZE}
    transform = get_transform(infer_cfg, train=False)

    results = []
    with torch.no_grad():
        for path_str in image_paths:
            img   = cv2.cvtColor(cv2.imread(str(path_str)), cv2.COLOR_BGR2RGB)
            img_t = transform(img).unsqueeze(0).to(device)
            out   = model(img_t)
            if out.shape[-1] > 1:
                count = int(out.argmax(dim=1).item())
            else:
                count = int(np.clip(round(out.squeeze().item()), 0, 16))
            results.append({"image_path": path_str, "num_balls": count})

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {output_json}")


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
