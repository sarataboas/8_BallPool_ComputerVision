"""
parameter_tuning.py — Random hyperparameter search for ball detection.

Pipeline per image:
    raw image → table contour (adaptive HSV) → ball mask (tunable HSV) → count blobs

Evaluation metrics:
    exact_match_acc : fraction of images where predicted count == ground truth
    mae             : mean absolute error of predicted ball counts

The search space is huge (~2.6 B exhaustive combinations) so we use random
sampling.  Best params are saved to a JSON checkpoint whenever a new best is
found, so you can kill the run early and still have a result.

Run from the project root:
    python testing/parameter_tuning.py
"""

import sys
import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import PROJECT_ROOT, DEVELOPMENT_SET
from utils import (
    detect_table_mask_adaptive,
    extract_main_table_component,
    extract_table_contour,
)

# ── Configuration ─────────────────────────────────────────────────────────────
METADATA_PATH  = Path(__file__).parent / "metadata.csv"
IMAGE_DIR      = DEVELOPMENT_SET
CHECKPOINT_PATH = PROJECT_ROOT / "tuning_best.json"

MAX_TRIALS  = 300
RANDOM_SEED = 42

# ── Parameter grid ────────────────────────────────────────────────────────────
param_grid = {
    "H_THRESH":           [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    "S_THRESH":           [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "V_THRESH":           [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "ERODE_KERNEL":       [7, 11, 15, 21],
    "OPEN_KERNEL":        [3, 5, 7],
    "CLOSE_KERNEL":       [3, 5, 7, 9],
    "MIN_RADIUS_SCALE":   [0.3, 0.4, 0.5, 0.6],
    "MAX_RADIUS_SCALE":   [1.6, 1.8, 2.0, 2.2],
    "CIRCULARITY_THRESH": [0.38, 0.42, 0.46, 0.50],
    "ASPECT_THRESH":      [0.3, 0.4, 0.5, 0.6],
    "DIST_THRESHOLD":     [0.4, 0.5, 0.6, 0.7],
    "DILATION_ITER":      [2, 3, 4],
    "S_MIN_VALID":        [20, 40, 60, 80],
    "V_MIN_VALID":        [20, 40, 60, 80],
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def parse_ball_types(s: str) -> List[int]:
    """Parse sparse ball-type strings like '[0,1,,3,,5,,,8,,,,,14,]'."""
    s = s.strip().lstrip("[").rstrip("]")
    result = []
    for token in s.split(","):
        token = token.strip()
        if token:
            try:
                result.append(int(token))
            except ValueError:
                pass
    return result


def load_metadata(path: Path) -> List[Dict]:
    """Load metadata.csv → list of {image_id, num_balls, ball_types}."""
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1:]:          # skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 2)
        if len(parts) < 2:
            continue
        rows.append({
            "image_id":   parts[0].strip(),
            "num_balls":  int(parts[1].strip()),
            "ball_types": parse_ball_types(parts[2]) if len(parts) > 2 else [],
        })
    return rows


def sample_random_params(grid: Dict, n: int, seed: int) -> List[Dict]:
    """Draw n unique random combinations from the grid."""
    rng     = random.Random(seed)
    keys    = list(grid.keys())
    seen    = set()
    samples = []
    max_attempts = n * 20
    attempts = 0
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        params = {k: rng.choice(grid[k]) for k in keys}
        sig = tuple(sorted(params.items()))
        if sig not in seen:
            seen.add(sig)
            samples.append(params)
    return samples


# ── Table detection ───────────────────────────────────────────────────────────

def detect_table(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Returns the table contour, or None if detection fails."""
    mask      = detect_table_mask_adaptive(bgr)
    component = extract_main_table_component(mask)
    return extract_table_contour(component)


# ── Ball mask (tunable) ───────────────────────────────────────────────────────

def get_ball_mask_tuned(bgr: np.ndarray, contour: np.ndarray, params: Dict) -> Optional[np.ndarray]:
    """Binary mask of ball candidates using tunable HSV deviation from cloth colour."""
    h, w = bgr.shape[:2]

    # Validity region = inside table contour, eroded inward
    valid = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(valid, [contour.astype(np.int32)], 255)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (ensure_odd(params["ERODE_KERNEL"]),) * 2)
    valid = cv2.erode(valid, ek, iterations=1)

    # Sample cloth colour from image centre
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    cy1, cy2 = int(h * 0.3), int(h * 0.7)
    cx1, cx2 = int(w * 0.3), int(w * 0.7)
    centre   = hsv[cy1:cy2, cx1:cx2]

    h_v = centre[:, :, 0].ravel()
    s_v = centre[:, :, 1].ravel()
    v_v = centre[:, :, 2].ravel()
    ok  = (s_v > params["S_MIN_VALID"]) & (v_v > params["V_MIN_VALID"])
    if ok.sum() < 50:
        return None

    h_med = int(np.median(h_v[ok]))
    s_med = int(np.median(s_v[ok]))
    v_med = int(np.median(v_v[ok]))

    # Per-pixel cloth deviation
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)

    h_diff = np.minimum(np.abs(H - h_med), 180 - np.abs(H - h_med))
    is_cloth = (
        (h_diff         < params["H_THRESH"]) &
        (np.abs(S - s_med) < params["S_THRESH"]) &
        (np.abs(V - v_med) < params["V_THRESH"])
    )

    ball_mask = (~is_cloth).astype(np.uint8) * 255
    ball_mask = cv2.bitwise_and(ball_mask, valid)

    ok_  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ensure_odd(params["OPEN_KERNEL"]),) * 2)
    ck_  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ensure_odd(params["CLOSE_KERNEL"]),) * 2)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN,  ok_)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, ck_)
    return ball_mask


# ── Ball counting ─────────────────────────────────────────────────────────────

def estimate_base_ball_radius(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    return max(4.0, math.sqrt(area) / 55.0) if area > 0 else 10.0


def count_balls_from_mask(ball_mask: Optional[np.ndarray],
                          contour: np.ndarray, params: Dict) -> int:
    if ball_mask is None:
        return 0

    base_r = estimate_base_ball_radius(contour)
    min_r  = params["MIN_RADIUS_SCALE"] * base_r
    max_r  = params["MAX_RADIUS_SCALE"] * base_r

    cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue

        aspect = min(bw, bh) / max(bw, bh)
        peri   = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        circ = (4.0 * np.pi * area) / (peri * peri)
        (_, _), radius = cv2.minEnclosingCircle(cnt)

        if aspect < params["ASPECT_THRESH"]:
            continue
        if circ < params["CIRCULARITY_THRESH"]:
            continue
        if radius < min_r or radius > max_r * 3:
            continue

        # Single ball
        if radius <= max_r:
            total += 1
            continue

        # Large blob — estimate merged ball count via distance transform peaks
        blob_mask = np.zeros_like(ball_mask)
        cv2.drawContours(blob_mask, [cnt], -1, 255, thickness=-1)
        dist     = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        if max_dist <= 0:
            continue

        peak = (dist > params["DIST_THRESHOLD"] * max_dist).astype(np.uint8) * 255
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        peak  = cv2.dilate(peak, dil_k, iterations=params["DILATION_ITER"])

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(peak)
        peak_count = sum(
            1 for i in range(1, n_labels)
            if stats[i, cv2.CC_STAT_AREA] >= 3
        )
        total += max(1, peak_count)

    return total


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_single_image(image_path: Path, gt_num_balls: int, params: Dict) -> Dict:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return {"ok": False, "pred_count": 0, "gt_count": gt_num_balls,
                "abs_error": gt_num_balls, "exact_match": 0,
                "reason": "image_not_found"}

    contour = detect_table(bgr)
    if contour is None:
        return {"ok": False, "pred_count": 0, "gt_count": gt_num_balls,
                "abs_error": gt_num_balls, "exact_match": 0,
                "reason": "table_not_detected"}

    mask  = get_ball_mask_tuned(bgr, contour, params)
    pred  = count_balls_from_mask(mask, contour, params)
    return {
        "ok":          True,
        "pred_count":  pred,
        "gt_count":    gt_num_balls,
        "abs_error":   abs(pred - gt_num_balls),
        "exact_match": int(pred == gt_num_balls),
        "reason":      "ok",
    }


def evaluate_dataset(metadata_rows: List[Dict], params: Dict) -> Dict:
    results = []
    for row in metadata_rows:
        path = IMAGE_DIR / row["image_id"]
        out  = evaluate_single_image(path, row["num_balls"], params)
        out["image_id"] = row["image_id"]
        results.append(out)

    n             = len(results)
    exact_matches = sum(r["exact_match"] for r in results)
    total_error   = sum(r["abs_error"]   for r in results)
    n_valid       = sum(r["ok"]          for r in results)
    return {
        "params":          params,
        "n_images":        n,
        "n_valid":         n_valid,
        "exact_match_acc": exact_matches / n if n else 0.0,
        "mae":             total_error   / n if n else float("inf"),
        "results":         results,
    }


def is_better(new: Dict, best: Optional[Dict]) -> bool:
    if best is None:
        return True
    if new["exact_match_acc"] != best["exact_match_acc"]:
        return new["exact_match_acc"] > best["exact_match_acc"]
    if new["mae"] != best["mae"]:
        return new["mae"] < best["mae"]
    return new["n_valid"] > best["n_valid"]


# ── Main tuning loop ──────────────────────────────────────────────────────────

def tune(metadata_path: Path = METADATA_PATH,
         image_dir:     Path = IMAGE_DIR,
         grid:          Dict = param_grid,
         max_trials:    int  = MAX_TRIALS,
         seed:          int  = RANDOM_SEED) -> Dict:

    metadata = load_metadata(metadata_path)
    samples  = sample_random_params(grid, max_trials, seed)

    best_score = None
    history    = []

    pbar = tqdm(samples, desc="Tuning", unit="trial")
    for i, params in enumerate(pbar, start=1):
        score = evaluate_dataset(metadata, params)

        history.append({
            "trial":           i,
            "params":          params,
            "exact_match_acc": score["exact_match_acc"],
            "mae":             score["mae"],
            "n_valid":         score["n_valid"],
        })

        if is_better(score, best_score):
            best_score = score

            # Save checkpoint immediately
            checkpoint = {
                "best_params":          best_score["params"],
                "best_exact_match_acc": best_score["exact_match_acc"],
                "best_mae":             best_score["mae"],
                "best_n_valid":         best_score["n_valid"],
                "n_images":             best_score["n_images"],
                "found_at_trial":       i,
                "history":              history,
                "best_results_per_image": best_score["results"],
            }
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)

            pbar.write(
                f"[Trial {i:03d}] NEW BEST  "
                f"acc={score['exact_match_acc']:.4f}  "
                f"mae={score['mae']:.4f}  "
                f"valid={score['n_valid']}/{score['n_images']}"
            )
        else:
            pbar.set_postfix(
                acc=f"{best_score['exact_match_acc']:.3f}",
                mae=f"{best_score['mae']:.3f}",
            )

    return checkpoint


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = tune()

    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(json.dumps(result["best_params"], indent=2))
    print(f"\nExact-match accuracy : {result['best_exact_match_acc']:.4f}")
    print(f"MAE                  : {result['best_mae']:.4f}")
    print(f"Valid images         : {result['best_n_valid']}/{result['n_images']}")
    print(f"Found at trial       : {result['found_at_trial']} / {MAX_TRIALS}")
    print(f"\nFull results saved to: {CHECKPOINT_PATH}")
