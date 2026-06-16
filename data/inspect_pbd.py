"""
Inspect the pool-ball-detection-6lfd9 dataset (link 3) before integrating it.

Checks:
  1. Class map — which class is cue ball, which are counted balls
  2. Label sanity — counts per image look plausible
  3. Warp quality — applies crop_table_roi to sample images and shows before/after

Usage (from project root):
  python data/inspect_pbd.py
"""

import sys, random
from pathlib import Path

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.cnn_pipeline import crop_table_roi

# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent / "external" / "pool_ball_detection"
CUE_BALL_NAMES = {"cue_ball"}
# ---------------------------------------------------------------------------


def find_nested_root(base: Path) -> Path:
    """Datasets from Roboflow are often inside a named subfolder."""
    if (base / "data.yaml").exists():
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and (sub / "data.yaml").exists():
            return sub
    raise FileNotFoundError(f"No data.yaml found under {base}")


def load_class_map(root: Path) -> dict:
    with open(root / "data.yaml") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(cid): ("cue" if str(n) in CUE_BALL_NAMES else "ball")
            for cid, n in names.items()}


def count_from_label(label_path: Path, class_map: dict) -> int:
    if not label_path.exists():
        return -1
    return sum(
        1 for ln in label_path.read_text().splitlines()
        if ln.strip() and class_map.get(int(ln.split()[0]), "ball") == "ball"
    )


def main():
    if not DATASET_DIR.exists():
        print(f"ERROR: {DATASET_DIR} does not exist.")
        print("Download the dataset from Roboflow (YOLOv11 format) and extract it there.")
        return

    root = find_nested_root(DATASET_DIR)
    print(f"Dataset root : {root}")

    # ── 1. Class map ─────────────────────────────────────────────────────────
    class_map = load_class_map(root)
    print("\nClass map:")
    for cid, role in sorted(class_map.items()):
        marker = "EXCLUDE (cue ball)" if role == "cue" else "COUNT"
        print(f"  class {cid} → {marker}")

    # ── 2. Label sanity ───────────────────────────────────────────────────────
    train_dir = root / "train"
    if not (train_dir / "images").exists():
        train_dir = root
    images = sorted((train_dir / "images").glob("*.jpg"))
    labels_dir = train_dir / "labels"

    counts = [count_from_label(labels_dir / (p.stem + ".txt"), class_map)
              for p in images]
    valid_counts = [c for c in counts if c >= 0]

    print(f"\nTrain images : {len(images)}")
    print(f"With labels  : {len(valid_counts)}")
    print(f"Ball count   : min={min(valid_counts)}  max={max(valid_counts)}  "
          f"mean={np.mean(valid_counts):.1f}  median={np.median(valid_counts):.0f}")

    # distribution
    from collections import Counter
    dist = Counter(valid_counts)
    fig, ax = plt.subplots(figsize=(10, 3))
    keys = sorted(dist)
    ax.bar(keys, [dist[k] for k in keys], color='darkorange')
    ax.set_title(f"pool-ball-detection-6lfd9 — ball count distribution (train split, {len(valid_counts)} images)")
    ax.set_xlabel("Ball count (cue ball excluded)")
    ax.set_ylabel("# images")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "inspect_pbd_distribution.png", dpi=100)
    plt.show()

    # ── 3. Warp quality ───────────────────────────────────────────────────────
    random.seed(42)
    sample = random.sample(images, min(6, len(images)))

    n = len(sample)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 5))
    ok, fallback = 0, 0

    for i, p in enumerate(sample):
        bgr  = cv2.imread(str(p))
        crop = crop_table_roi(bgr.copy())
        fell_back = (crop.shape == bgr.shape and np.array_equal(crop, bgr))
        if fell_back:
            fallback += 1
        else:
            ok += 1

        cnt = count_from_label(labels_dir / (p.stem + ".txt"), class_map)

        axes[0, i].imshow(cv2.cvtColor(bgr,  cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Original  balls={cnt}\n{p.name[:28]}", fontsize=6)
        axes[0, i].axis('off')

        axes[1, i].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        status = "FALLBACK" if fell_back else f"warped {crop.shape[1]}×{crop.shape[0]}"
        axes[1, i].set_title(f"crop_table_roi\n[{status}]", fontsize=6, color='red' if fell_back else 'green')
        axes[1, i].axis('off')

    plt.suptitle("pool-ball-detection-6lfd9 — crop_table_roi preview (6 random train images)", fontsize=10)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "inspect_pbd_warp.png", dpi=100)
    plt.show()

    # Full warp success rate on all images
    print("\nRunning crop_table_roi on all train images (may take a minute)...")
    total_ok, total_fb = 0, 0
    for p in images:
        bgr  = cv2.imread(str(p))
        crop = crop_table_roi(bgr.copy())
        if crop.shape == bgr.shape and np.array_equal(crop, bgr):
            total_fb += 1
        else:
            total_ok += 1

    print(f"  Warp success : {total_ok}/{len(images)}  ({100*total_ok/len(images):.1f}%)")
    print(f"  Fallback     : {total_fb}/{len(images)}  ({100*total_fb/len(images):.1f}%)")
    print("\nSaved: data/inspect_pbd_distribution.png  and  data/inspect_pbd_warp.png")


if __name__ == "__main__":
    main()
