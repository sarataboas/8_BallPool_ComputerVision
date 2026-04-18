import os
from pathlib import Path
import json
import logging
from typing import Any, Optional, Dict, Tuple, List

import cv2
import numpy as np

# ----------- CONFIGS --> all directories and variables used in the script -----------

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = 'example_json/input.json' 
TOPVIEW_RESULTS = 'results/topview_images/'
DETECTION_RESULTS = 'results/detection_images/'
OUTPUT_FILE = 'results/output.json'

# BGR colours for drawing each ball number on the detection image.
# Used only for visualization — one distinct colour per ball.
BALL_DRAW_COLOURS_BGR = {
    0:  (255, 255, 255),  # white  — cue ball
    1:  (0,   200, 255),  # yellow
    2:  (200,  50,   0),  # blue
    3:  (0,    0,  200),  # red
    4:  (130,  0,  130),  # purple
    5:  (0,   120, 255),  # orange
    6:  (0,   160,   0),  # green
    7:  (30,   80, 130),  # brown
    8:  (30,   30,  30),  # black
    9:  (180, 230, 255),  # yellow stripe
    10: (230, 150, 100),  # blue stripe
    11: (100, 100, 255),  # red stripe
    12: (200, 100, 200),  # purple stripe
    13: (100, 200, 255),  # orange stripe
    14: (100, 210, 100),  # green stripe
    15: (150, 160, 180),  # brown stripe
}

BALL_COLOURS = {
    1:  (18,  32,  130, 130),   # yellow solid  — high sat/val, mid hue
    2:  (95, 115,  100,  80),   # blue solid    — H ~100-110 from data
    3:  (0,   7,   120,  80),   # red solid     — low hue only
    4:  (100, 120,  80,  60),   # purple solid  — H ~105-108 from data
    5:  (8,   18,  120, 120),   # orange solid  — H ~8-18, high sat/val
    6:  (35,  95,   80,  50),   # green solid   — extended to H=95
    7:  (18,  32,   60, 60),    # brown solid   — lower sat than orange
    8:  (0,  180,   0,   0),    # black         — matched by low value
}

# Stripe balls have the same hue as their solid counterpart
STRIPE_MAP = {
    9:  1,   # yellow stripe
    10: 2,   # blue stripe
    11: 3,   # red stripe
    12: 4,   # purple stripe
    13: 5,   # orange stripe
    14: 6,   # green stripe
    15: 7,   # brown stripe
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# -------------------- Utility Functions -------------------------
def load_input_paths(input_json):
    """
    Loads the image paths in the input JSON file.
    Returns a list of Path objects. 
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    paths = data.get("image_path") or data.get("image_paths") or []
    return [PROJECT_ROOT / Path(p) for p in paths]


def imread_bgr(path):
    """Reads an image from `path` in the default openCV format (BGR)"""

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save_warped_image(image, original_path, output_dir):
    """Saves the warped image"""
    
    output_dir = PROJECT_ROOT / Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = original_path.stem + ".jpg"
    save_path = output_dir / filename

    success = cv2.imwrite(str(save_path), image)

    if success:
        logging.info(f"Saved: {save_path}")
    else:
        logging.error(f"Failed to save: {save_path}")


def save_detection_image(bgr, detections, ball_ids, original_path, output_dir):
    """
    Draws bounding boxes and ball numbers on the original image and saves it.

    Each box is coloured according to BALL_DRAW_COLOURS_BGR so balls are
    visually distinguishable at a glance. Unclassified balls (None) are
    drawn in grey with a '?' label.

    Args:
        bgr:           original BGR image
        detections:    list of dicts with 'x1','y1','x2','y2'
        ball_ids:      list of ball numbers (int or None), same order as detections
        original_path: Path of the source image (used to derive the filename)
        output_dir:    directory where the annotated image will be saved
    """
    output_dir = PROJECT_ROOT / Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis = bgr.copy()

    for det, ball_num in zip(detections, ball_ids):
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']

        colour = BALL_DRAW_COLOURS_BGR.get(ball_num, (160, 160, 160))  # grey for None
        label  = str(ball_num) if ball_num is not None else '?'

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        # Label background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx, ly = x1, max(y1 - 4, th + 4)
        cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly), colour, -1)

        # Label text in black for contrast
        cv2.putText(vis, label, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    filename  = original_path.stem + ".jpg"
    save_path = output_dir / filename
    success   = cv2.imwrite(str(save_path), vis)

    if success:
        logging.info(f"Saved detection image: {save_path}")
    else:
        logging.error(f"Failed to save detection image: {save_path}")


def save_output_json(data, output_file):
    """Saves the output results into a JSON file"""

    output_path = PROJECT_ROOT / Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logging.info(f"Output saved to: {output_path}")

# -------------------- Table Mask Detection -------------------------

def detect_table_mask_adaptive(bgr):
    """Detects the table mask by adaptively estimating the cloth colour from the image center."""

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    cx1, cy1 = int(w * 0.35), int(h * 0.35)
    cx2, cy2 = int(w * 0.65), int(h * 0.65)
    center = hsv[cy1:cy2, cx1:cx2]

    h_vals = center[:, :, 0].reshape(-1)
    s_vals = center[:, :, 1].reshape(-1)
    v_vals = center[:, :, 2].reshape(-1)

    valid = (s_vals > 50) & (v_vals > 50)
    if valid.sum() < 50:
        return None

    h_med = int(np.median(h_vals[valid]))
    low = max(0, h_med - 18)
    high = min(179, h_med + 18)

    mask = cv2.inRange(hsv, np.array([low, 70, 70]), np.array([high, 255, 255]))

    # remove likely noise bands
    mask[:int(0.2 * h), :] = 0
    mask[int(0.9 * h):, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def extract_main_table_component(mask):
    """ Given a mask, extract the largest connected component closest to the image center"""

    if mask is None:
        return None

    h, w = mask.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return None

    center_pt = np.array([w / 2, h / 2])
    best_label = None
    best_dist = float("inf")

    for i in range(1, num_labels):
        c = centroids[i]
        dist = np.linalg.norm(c - center_pt)
        if dist < best_dist:
            best_dist = dist
            best_label = i

    component_mask = np.uint8(labels == best_label) * 255
    return component_mask

# -------------------------- Table Contour ---------------------------

def extract_table_contour(component_mask):
    """Receives a binary mask of the main component and extracts its contour"""

    if component_mask is None:
        return None

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    contour = cv2.convexHull(contour)
    return contour


# -------------------------- Table Corners ---------------------------

def order_points(pts):
    """Orders 4 points in the order: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)

    y_sorted = pts[np.argsort(pts[:, 1])] # sort by y-coordinate to separate top and bottom points
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]

    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return ordered

def polygon_area(pts):
    """Fall back function: computes the area of a polygon given its vertices -> used in case of contour approximation failure, to filter out small contours"""
    pts = np.asarray(pts, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def contour_to_corners_refined(contour):
    """Refined contour to corners: tries multiple approximation thresholds, falls back to minAreaRect if no quadrilateral found, and filters out small contours."""
    
    if contour is None:
        return None

    peri = cv2.arcLength(contour, True)
    approx = None
    for eps in [0.01, 0.02, 0.03]:
        approx_candidate = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx_candidate) == 4:
            approx = approx_candidate
            break

    if approx is not None:
        corners = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect)

    corners = order_points(corners)

    if polygon_area(corners) < 1000:
        return None

    return corners


# -------------------------- Table Top-view ---------------------------

def expand_corners(corners, expand_px=20):
    """Push corners outward so rails/balls at edges aren't clipped."""
    
    cx = np.mean(corners[:, 0])
    cy = np.mean(corners[:, 1])
    expanded = []
    for (x, y) in corners:
        dx = x - cx
        dy = y - cy
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        expanded.append([x + expand_px * dx / norm,
                         y + expand_px * dy / norm])
    return np.array(expanded, dtype=np.float32)

def warp_table(bgr, corners):
    """Warps the image to a top-down view of the table using the detected corners."""

    pts = order_points(corners)
    pts_expanded = expand_corners(pts, expand_px=25)  # checked - Necessary ? 

    (tl, tr, br, bl) = pts_expanded 

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = int((widthA + widthB) / 2)

    # enforce pool table aspect ratio (2:1)
    aspect_ratio = 2.0 # width / height
    maxHeight = int(maxWidth * aspect_ratio)
 
    scale = 2.0
    maxWidth = int(maxWidth * scale)
    maxHeight = int(maxHeight * scale)

    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_expanded, dst)
    warped = cv2.warpPerspective(bgr, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    return warped, M


# -------------------------- Ball Detection --------------------------

def estimate_cloth_colour(bgr):
    """
    Estimates the cloth colour by sampling the HSV median from the image center.

    Returns:
        (h_med, s_med, v_med) or None if not enough valid pixels.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    cx1, cy1 = int(w * 0.3), int(h * 0.3)
    cx2, cy2 = int(w * 0.7), int(h * 0.7)
    center = hsv[cy1:cy2, cx1:cx2]

    h_vals = center[:, :, 0].reshape(-1)
    s_vals = center[:, :, 1].reshape(-1)
    v_vals = center[:, :, 2].reshape(-1)

    valid = (s_vals > 40) & (v_vals > 40)
    if valid.sum() < 50:
        return None

    return (
        int(np.median(h_vals[valid])),
        int(np.median(s_vals[valid])),
        int(np.median(v_vals[valid]))
    )


def compute_ball_score(bgr, cloth_hsv):
    """
    Computes a continuous per-pixel score [0, 1] representing how different
    each pixel is from the cloth colour.

    A score close to 1 means "very different from cloth" (likely a ball).
    A score close to 0 means "very similar to cloth" (background).

    Unlike a binary threshold, this preserves gradients — ball centres
    get consistently high scores while shadows/rails get moderate, noisy scores.

    Args:
        bgr:        original BGR image
        cloth_hsv:  (h_med, s_med, v_med) from estimate_cloth_colour()

    Returns:
        score map (float32, shape HxW, values in [0, 1])
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_med, s_med, v_med = cloth_hsv

    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    h_diff = np.abs(H - h_med)
    h_diff = np.minimum(h_diff, 180 - h_diff)
    s_diff = np.abs(S - s_med)
    v_diff = np.abs(V - v_med)

    # Normalize each channel by its threshold so score=1 means "at threshold"
    # We use thresholds from the original binary version as reference
    h_score = h_diff / 20.0
    s_score = s_diff / 35.0
    v_score = v_diff / 35.0

    # Average of the three channels, clipped to [0, 1]
    score = (h_score + s_score + v_score) / 3.0
    score = np.clip(score, 0.0, 1.0).astype(np.float32)

    return score


def build_valid_region(bgr, contour, erode_px=60):
    """
    Builds a binary mask of the table playing surface from the table contour,
    eroded inward to exclude rails.

    Args:
        bgr:       original BGR image
        contour:   table contour from extract_table_contour()
        erode_px:  erosion size in pixels

    Returns:
        valid_region: binary mask (uint8, 0/255)
    """
    h, w = bgr.shape[:2]
    valid_region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(valid_region, [contour], 255)

    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
    valid_region = cv2.erode(valid_region, erode_k, iterations=1)

    return valid_region


def score_to_ball_mask(score_map, valid_region, score_thr=0.5):
    """
    Converts the continuous score map to a binary ball mask by thresholding,
    restricted to the valid table region, with morphological cleanup.

    Args:
        score_map:    float32 score map from compute_ball_score()
        valid_region: binary mask from build_valid_region()
        score_thr:    pixels above this score are considered non-cloth (default 0.5)

    Returns:
        ball_mask: binary mask (uint8, 0/255)
    """
    binary = (score_map > score_thr).astype(np.uint8) * 255
    ball_mask = cv2.bitwise_and(binary, valid_region)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, k)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, k)

    return ball_mask


def get_ball_mask_original(bgr, contour):
    """
    Main entry point: produces a binary ball mask from the original image.

    Args:
        bgr:     original BGR image
        contour: table contour from extract_table_contour()

    Returns:
        ball_mask (uint8, 0/255) or None if cloth colour estimation failed.
    """
    cloth_hsv = estimate_cloth_colour(bgr)
    if cloth_hsv is None:
        return None

    score_map    = compute_ball_score(bgr, cloth_hsv)
    valid_region = build_valid_region(bgr, contour)
    ball_mask    = score_to_ball_mask(score_map, valid_region)

    return ball_mask

def extract_ball_bboxes_watershed_single(blob_mask, bgr, expected_r):
    h, w = bgr.shape[:2]

    min_area = int(np.pi * (expected_r * 0.4) ** 2)
    max_area = int(np.pi * (expected_r * 1.8) ** 2)

    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, seeds = cv2.threshold(dist_norm, 0.8, 1.0, cv2.THRESH_BINARY)
    seeds = seeds.astype(np.uint8)

    num_labels, markers = cv2.connectedComponents(seeds)
    markers = markers + 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(blob_mask, kernel, iterations=2)
    unknown = cv2.subtract(dilated, seeds * 255)
    markers[unknown == 255] = 0

    # Debug: state before watershed
    dist_vis   = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    seeds_vis  = (seeds * 255).astype(np.uint8)
    unknown_vis = unknown.copy()
    #print(f"    [watershed_single] seeds found: {num_labels - 1}")
    #show_many(
    #    [blob_mask, dist_vis, seeds_vis, unknown_vis],
    #    titles=["Blob mask", "Distance transform", "Seeds", "Unknown region"],
    #    cols=2, figsize=(12, 6)
    #)

    markers = cv2.watershed(bgr, markers)

    # Debug: watershed result
    watershed_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(2, num_labels + 1):
        colour = np.random.randint(60, 255, 3).tolist()
        watershed_vis[markers == label] = colour
    watershed_vis[markers == -1] = [255, 0, 0]

    detections = []
    for label in range(2, num_labels + 1):
        region = (markers == label).astype(np.uint8) * 255
        area   = cv2.countNonZero(region)

        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            #print(f"    [label {label}] REJECTED — no contour")
            continue

        cnt  = cnts[0]
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        peri   = cv2.arcLength(cnt, True)
        circ   = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
        (xc, yc), radius = cv2.minEnclosingCircle(cnt)
        fill_ratio = area / (np.pi * radius * radius) if radius > 0 else 0

        passes_area   = min_area < area < max_area
        passes_aspect = aspect >= 0.35
        passes_circ   = circ >= 0.35
        passes_fill   = fill_ratio >= 0.45

        if not (passes_area and passes_aspect and passes_circ and passes_fill):
            #print(f"    [label {label}] REJECTED — shape filters")
            continue

        roi_hsv = cv2.cvtColor(bgr[y:y+bh, x:x+bw], cv2.COLOR_BGR2HSV)
        v_p20   = float(np.percentile(roi_hsv[:, :, 2], 20))
        s_mean  = float(np.mean(roi_hsv[:, :, 1]))
        if v_p20 < 50 and s_mean < 60:
            #print(f"    [label {label}] REJECTED — pocket filter (v_p20={v_p20:.0f}, s_mean={s_mean:.0f})")
            continue

        detections.append({
            'cx': x + bw / 2, 'cy': y + bh / 2,
            'x1': x, 'y1': y, 'x2': x + bw, 'y2': y + bh
        })

    return detections


def extract_ball_bboxes_hybrid(ball_mask, bgr):
    if ball_mask is None:
        return []

    h, w = bgr.shape[:2]

    expected_r = min(h, w) * 0.025
    min_area   = int(np.pi * (expected_r * 0.4) ** 2)
    max_area   = int(np.pi * (expected_r * 0.7) ** 2)
    large_area = max_area * 20

    #print(f"expected_r={expected_r:.1f}  min_area={min_area}  max_area={max_area}  large_area={large_area}")

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(ball_mask)

    #print(f"Total blobs found: {num_labels - 1}")

    detections = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]

        # Very large blob
        if area > large_area:
            #print(f"  blob {i}: area={area} → SKIPPED (too large, rail/noise)")
            continue

        blob = (labels == i).astype(np.uint8) * 255

        # Large blob → watershed
        if area > max_area:
            #print(f"  blob {i}: area={area} → WATERSHED (merged balls)")
            merged = extract_ball_bboxes_watershed_single(blob, bgr, expected_r)
            #print(f"    watershed returned {len(merged)} detections")
            detections.extend(merged)
            continue

        # Small blob
        if area < min_area:
            continue

        # Normal blob → standard filters
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        peri = cv2.arcLength(cnts[0], True)
        circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

        passes_aspect = aspect >= 0.35
        passes_circ   = circ >= 0.35        # Alterar em cima tambem

        if not (passes_aspect and passes_circ):
            #print(" → REJECTED")
            continue

        cx, cy = centroids[i]
        detections.append({
            'cx': cx, 'cy': cy,
            'x1': x, 'y1': y,
            'x2': x + bw, 'y2': y + bh
        })
    return detections


# ---------------------- Ball Classification ---------------------------
def is_white_pixel(hsv_pixel, sat_thr=40, val_thr=180):
    """Returns True if the HSV pixel is likely to be white/light, based on low saturation and high value."""
    return hsv_pixel[1] < sat_thr and hsv_pixel[2] > val_thr


def get_white_fraction(ball_region_hsv, ball_region_mask):
    """
    Computes the fraction of ball pixels that are white/light.
    Used to distinguish: cue ball (>70%), stripes (>25%), solids (<25%).
    """
    ball_pixels = ball_region_hsv[ball_region_mask > 0]
    if len(ball_pixels) == 0:
        return 0.0

    white_pixels = np.sum(
        (ball_pixels[:, 1] < 90) & (ball_pixels[:, 2] > 140)
    )
    return white_pixels / len(ball_pixels)


def get_dominant_colour(ball_region_hsv, ball_region_mask):
    """
    Returns the median hue and saturation of the non-white ball pixels.
    White pixels (reflexos, stripes) are excluded to get the true ball colour.
    """
    ball_pixels = ball_region_hsv[ball_region_mask > 0]
    if len(ball_pixels) == 0:
        return None

    # Exclude white pixels (reflexos, stripe white band)
    non_white = ball_pixels[
        ~((ball_pixels[:, 1] < 40) & (ball_pixels[:, 2] > 180))
    ]
    if len(non_white) == 0:
        return None

    median_h = int(np.median(non_white[:, 0]))
    median_s = int(np.median(non_white[:, 1]))
    median_v = int(np.median(non_white[:, 2]))
    return (median_h, median_s, median_v)


def match_colour_to_ball(median_hsv):
    """
    Matches a median HSV colour to the closest solid ball number (1-8).
    The black ball (8) is matched by low value rather than hue.
    """
    if median_hsv is None:
        return None

    h, s, v = median_hsv

    # Black ball: very low value regardless of hue
    if v < 80:
        return 8

    for ball_num, (h_min, h_max, s_min, v_min) in BALL_COLOURS.items():
        if ball_num == 8:
            continue
        if h_min <= h <= h_max and s >= s_min and v >= v_min:
            return ball_num

    return None


def classify_ball(det, bgr, ball_mask):
    """
    Classifies a detected ball by analysing its colour and white fraction.
    """
    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']

    # Clamp to image bounds
    ih, iw = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    # Extract crops
    bgr_crop = bgr[y1:y2, x1:x2]
    mask_crop = ball_mask[y1:y2, x1:x2]
    hsv_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

    # Step 1: white fraction
    white_frac = get_white_fraction(hsv_crop, mask_crop)

    # Step 2: cue ball detection
    if white_frac > 0.65:
        return 0  # cue ball (white)

    # Step 3: dominant colour
    median_hsv = get_dominant_colour(hsv_crop, mask_crop)
    solid_num = match_colour_to_ball(median_hsv)

    if solid_num is None:
        return None

    # Step 4: solid vs stripe
    if white_frac > 0.15:
        # Has a significant white band — it's a stripe
        stripe_num = next(
            (k for k, v in STRIPE_MAP.items() if v == solid_num), None
        )
        return stripe_num
    else:
        return solid_num
    


# -------------------- Image Processing Complete Pipeline  -------------------
def main():
    input_data = load_input_paths(input_json=INPUT_FILE)
    print(len(input_data))

    output_results = []

    for path in input_data: 
        image_info = {}

        print()
        img_bgr = imread_bgr(path=path)
        image_info['image_path'] = str(path.relative_to(PROJECT_ROOT))
        
        h, w = img_bgr.shape[:2]

        mask = detect_table_mask_adaptive(bgr=img_bgr)
        component_mask = extract_main_table_component(mask=mask)
        contour = extract_table_contour(component_mask=component_mask)
        corners = contour_to_corners_refined(contour=contour)
        if corners is None:
            logging.warning(f"Skipping {path}, no corners detected")
            continue
        warped_image, _ = warp_table(bgr=img_bgr, corners=corners)
        save_warped_image(image=warped_image, original_path=path, output_dir=TOPVIEW_RESULTS)
        
        cloth_hsv    = estimate_cloth_colour(img_bgr)
        score_map    = compute_ball_score(img_bgr, cloth_hsv)
        valid_region = build_valid_region(img_bgr, contour)
        ball_mask_orig    = score_to_ball_mask(score_map, valid_region)

        detections = extract_ball_bboxes_hybrid(ball_mask_orig, img_bgr)

        # print(detections)
        image_info['num_balls'] = len(detections)
        balls = []
        for detected_ball in detections:
            # identify the ball number
            ball_id = classify_ball(det=detected_ball, bgr=img_bgr, ball_mask=ball_mask_orig)
            # get ball coordinates
            xmin = float(detected_ball['x1']) / w
            xmax = float(detected_ball['x2']) / w
            ymin = float(detected_ball['y1']) / h
            ymax = float(detected_ball['y2']) / h
            # write in the results dictionary
            ball_info = {
                "number": ball_id,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax
            }
            balls.append(ball_info)
        image_info['balls'] = balls
        output_results.append(image_info)

        # Save annotated detection image (bboxes + ball numbers)
        ball_ids = [b['number'] for b in balls]
        save_detection_image(
            bgr=img_bgr,
            detections=detections,
            ball_ids=ball_ids,
            original_path=path,
            output_dir=DETECTION_RESULTS
        )

    print(len(output_results))
    save_output_json(data=output_results, output_file=OUTPUT_FILE)
        


if __name__=='__main__':
    main()