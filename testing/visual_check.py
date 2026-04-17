"""
visual_check.py — Quick visual sanity-check for the ball-detection pipeline.

Picks one image, shows the table detection step, then tries a small
H_THRESH × CIRCULARITY_THRESH grid and draws the detected blobs.
Green title = exact match with ground truth, Red = mismatch.

Run from the project root:
    python testing/visual_check.py
"""

import sys
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# allow importing siblings
sys.path.insert(0, str(Path(__file__).parent))

from config import DEVELOPMENT_SET
from utils import (
    detect_table_mask_adaptive,
    extract_main_table_component,
    extract_table_contour,
)
from parameter_tuning import (
    detect_table,
    get_ball_mask_tuned,
    count_balls_from_mask,
    load_metadata,
    estimate_base_ball_radius,
)

# ── Image to inspect (change this to any filename in development_set/) ────────
IMAGE_NAME = "34_png.rf.407889c8b2afe2a57c57461fa46b2d4f.jpg"
METADATA_PATH = Path(__file__).parent / "metadata.csv"

# ── Base parameters (everything fixed except the two we sweep) ────────────────
BASE_PARAMS = {
    "H_THRESH":           18,
    "S_THRESH":           50,
    "V_THRESH":           50,
    "ERODE_KERNEL":       11,
    "OPEN_KERNEL":         5,
    "CLOSE_KERNEL":        7,
    "MIN_RADIUS_SCALE":    0.4,
    "MAX_RADIUS_SCALE":    2.0,
    "CIRCULARITY_THRESH":  0.46,
    "ASPECT_THRESH":       0.5,
    "DIST_THRESHOLD":      0.5,
    "DILATION_ITER":       3,
    "S_MIN_VALID":        40,
    "V_MIN_VALID":        40,
}

# ── Grid axes ─────────────────────────────────────────────────────────────────
H_THRESH_VALUES = [10, 18, 26]          # columns
CIRC_VALUES     = [0.38, 0.46, 0.54]   # rows


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def annotate_blobs(bgr, contour, params):
    """Draw circles around detected blobs; return annotated RGB + predicted count."""
    ball_mask = get_ball_mask_tuned(bgr, contour, params)
    rgb = to_rgb(bgr.copy())
    if ball_mask is None:
        return rgb, 0

    pred_count = count_balls_from_mask(ball_mask, contour, params)
    base_r = estimate_base_ball_radius(contour)
    min_r  = params["MIN_RADIUS_SCALE"] * base_r
    max_r  = params["MAX_RADIUS_SCALE"] * base_r

    cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circ   = (4 * np.pi * area) / (peri ** 2)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0

        if (aspect   >= params["ASPECT_THRESH"] and
                circ >= params["CIRCULARITY_THRESH"] and
                min_r <= r <= max_r * 3):
            color = (0, 255, 0) if r <= max_r else (255, 165, 0)  # green=single, orange=merged
            cv2.circle(rgb, (int(cx), int(cy)), int(r), color, 2)

    return rgb, pred_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    image_path = DEVELOPMENT_SET / IMAGE_NAME
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"Could not load image: {image_path}")
        return

    # Ground-truth count from metadata
    metadata = load_metadata(str(METADATA_PATH))
    gt = next((r["num_balls"] for r in metadata if r["image_id"] == IMAGE_NAME), "?")

    # ── Panel 1: original + table contour ────────────────────────────────────
    mask_raw  = detect_table_mask_adaptive(bgr)
    component = extract_main_table_component(mask_raw)
    contour   = extract_table_contour(component)

    rgb_orig  = to_rgb(bgr)
    rgb_table = rgb_orig.copy()
    if contour is not None:
        cv2.drawContours(rgb_table, [contour.astype(np.int32)], -1, (255, 50, 50), 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(rgb_orig);   axes[0].set_title("Original image");          axes[0].axis("off")
    axes[1].imshow(rgb_table);  axes[1].set_title("Table detected (red contour)"); axes[1].axis("off")
    status = "OK" if contour is not None else "FAILED"
    plt.suptitle(f"{IMAGE_NAME}  |  GT balls = {gt}  |  Table detection: {status}", fontsize=11)
    plt.tight_layout()
    plt.show()

    if contour is None:
        print("Table not detected — cannot proceed with ball detection.")
        return

    # ── Panel 2: H_THRESH × CIRCULARITY_THRESH grid ──────────────────────────
    rows, cols = len(CIRC_VALUES), len(H_THRESH_VALUES)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    for ri, circ in enumerate(CIRC_VALUES):
        for ci, h_thresh in enumerate(H_THRESH_VALUES):
            params = {**BASE_PARAMS,
                      "H_THRESH": h_thresh,
                      "CIRCULARITY_THRESH": circ}

            annotated, pred = annotate_blobs(bgr, contour, params)
            ax = axes[ri][ci]
            ax.imshow(annotated)

            match   = (pred == gt)
            t_color = "green" if match else "red"
            ax.set_title(
                f"H_THRESH={h_thresh}  circ={circ}\n"
                f"pred={pred}   gt={gt}",
                color=t_color, fontsize=9
            )
            ax.axis("off")

    plt.suptitle(
        "Parameter sweep: H_THRESH (cols) × CIRCULARITY_THRESH (rows)\n"
        "Green title = exact match  |  circles: green=single ball, orange=merged blob",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
