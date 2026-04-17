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
OUTPUT_FILE = 'results/output.json'

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

def get_ball_mask_original(bgr, contour):
    """
    Applies the ball detection mask to the original (non-warped) image,
    restricting the analysis to the table playing surface.

    Instead of warping the image, we detect the table contour and use it
    as a validity mask — anything outside the table polygon is ignored.
    This avoids perspective distortion artifacts introduced by warping.

    Args:
        bgr: original BGR image

    Returns:
        ball_mask: binary mask (uint8, 0 or 255) in original image space,
                   or None if table detection failed
    """
    h, w = bgr.shape[:2]

    valid_region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(valid_region, [contour], 255)

    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    valid_region = cv2.erode(valid_region, erode_k, iterations=1)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    cx1, cy1 = int(w * 0.3), int(h * 0.3)
    cx2, cy2 = int(w * 0.7), int(h * 0.7)
    center = hsv[cy1:cy2, cx1:cx2]

    h_vals = center[:, :, 0].reshape(-1)
    s_vals = center[:, :, 1].reshape(-1)
    v_vals = center[:, :, 2].reshape(-1)

    valid = (s_vals > 40) & (v_vals > 40)
    if valid.sum() < 50:
        return None

    h_med = int(np.median(h_vals[valid]))
    s_med = int(np.median(s_vals[valid]))
    v_med = int(np.median(v_vals[valid]))

    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)

    h_diff = np.abs(H - h_med)
    h_diff = np.minimum(h_diff, 180 - h_diff)
    s_diff = np.abs(S - s_med)
    v_diff = np.abs(V - v_med)

    is_cloth = (h_diff < 20) & (s_diff < 35) & (v_diff < 35)

    ball_mask = (~is_cloth).astype(np.uint8) * 255
    ball_mask = cv2.bitwise_and(ball_mask, valid_region)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, k)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, k)

    return ball_mask

def extract_ball_bboxes_watershed(ball_mask, bgr):
    """Extracts ball bounding boxes from the ball mask using watershed segmentation: thresholding, distance transform, and shape filtering to find circular blobs corresponding to balls."""

    if ball_mask is None:
        return []

    h, w = bgr.shape[:2]

    # --- Step 1: Distance transform ---
    dist = cv2.distanceTransform(ball_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # --- Step 2: Threshold to find seeds ---
    _, seeds = cv2.threshold(dist_norm, 0.6, 1.0, cv2.THRESH_BINARY)
    seeds = seeds.astype(np.uint8)

    # --- Step 3: Marker map ---
    num_labels, markers = cv2.connectedComponents(seeds)
    markers = markers + 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(ball_mask, kernel, iterations=3)
    unknown = cv2.subtract(dilated, seeds * 255)
    markers[unknown == 255] = 0

    # --- Step 4: Watershed ---
    markers = cv2.watershed(bgr, markers)

    # --- Step 5: Extract regions ---
    expected_r = min(h, w) * 0.025
    min_area = int(np.pi * (expected_r * 0.4) ** 2)
    max_area = int(np.pi * (expected_r * 1.8) ** 2)

    detections = []

    for label in range(2, num_labels + 1):

        region = (markers == label).astype(np.uint8) * 255
        area = cv2.countNonZero(region)

        if not (min_area < area < max_area):
            continue

        contours_r, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        if not contours_r:
            continue

        cnt = contours_r[0]

        x, y, bw, bh = cv2.boundingRect(cnt)

        # --- Shape filters ---
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        if aspect < 0.35:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circ = 4 * np.pi * area / (peri * peri)
        if circ < 0.45:
            continue

        # --- NEW: fill ratio test ---
        (xc, yc), radius = cv2.minEnclosingCircle(cnt)

        if radius < 1:
            continue

        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area

        # Reject blobs that don't fill the circle well
        if fill_ratio < 0.55:
            continue

        # ---  NEW: Pocket removal (brightness filter) ---
        roi = bgr[y:y+bh, x:x+bw]

        if roi.size == 0:
            continue

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        v_vals = hsv_roi[:, :, 2].flatten()
        s_vals = hsv_roi[:, :, 1].flatten()

        v_p20 = np.percentile(v_vals, 20)
        s_mean = np.mean(s_vals)

        # Reject only if dark AND low saturation (pocket)
        if v_p20 < 50 and s_mean < 60:
            continue

        # --- Final detection ---
        cx = x + bw / 2
        cy = y + bh / 2

        detections.append({
            'cx': cx,
            'cy': cy,
            'x1': x,
            'y1': y,
            'x2': x + bw,
            'y2': y + bh
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
        

        ball_mask = get_ball_mask_original(bgr=img_bgr, contour=contour)
    
        detections = extract_ball_bboxes_watershed(ball_mask=ball_mask, bgr=img_bgr)
        # print(detections)
        image_info['num_balls'] = len(detections)
        balls = []
        for detected_ball in detections:
            # identify the ball number
            ball_id = classify_ball(det=detected_ball, bgr=img_bgr, ball_mask=ball_mask)
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

    print(len(output_results))
    save_output_json(data=output_results, output_file=OUTPUT_FILE)
        


if __name__=='__main__':
    main()