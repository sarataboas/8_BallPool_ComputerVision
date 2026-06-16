from pathlib import Path
import json
import math
import os
from collections import defaultdict, Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams["figure.figsize"] = (10, 6)


######################## Utility helper functions (shared: Task 1 + Task 2) ########################

def imread_rgb(path):
    ''' 
    Reads an image from `path` and converts it to RGB format (because openCV uses BGR by default) 
    '''
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imread_bgr(path):
    ''' 
    Reads an image from `path` in the default openCV format (BGR)
    '''
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def show(img, title=None, cmap=None, figsize=(8, 5)):
    ''' Displays the image.
    If the image has 2 dimensions (2D), it is displayed in grayscale using the provided colormap (cmap).
    If the image has 3 dimensions (3D), it is displayed in color.
    Adds an optional title and removes axes for better visualization
    '''
    plt.figure(figsize=figsize)
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

def show_many(images, titles=None, cols=3, figsize=(16, 10)):
    ''' 
    Displays multiple images in a grid format
    '''
    n = len(images)
    rows = int(math.ceil(n / cols))
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()

######################## Utility helper functions (Task 2 only) ########################

def find_dataset_root(base):
    """Finds the nested dataset root that contains data.yaml (Roboflow datasets extract into a named subfolder)."""
    from pathlib import Path
    base = Path(base)
    if (base / 'data.yaml').exists():
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and (sub / 'data.yaml').exists():
            return sub
    raise FileNotFoundError(f"No data.yaml found under {base}")


def count_ext_labels(label_path, class_map):
    """Counts ball annotations in a YOLO label file using a class_map dict (class_id → 'ball'|'cue')."""
    from pathlib import Path
    label_path = Path(label_path)
    if not label_path.exists():
        return -1
    return sum(
        1 for ln in label_path.read_text().splitlines()
        if ln.strip() and class_map.get(int(ln.split()[0]), 'ball') == 'ball'
    )


def cloth_center_hue(bgr):
    """Returns the median HSV hue of the cloth region at the image center. Returns -1 if the region is too dark/unsaturated."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    center = hsv[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
    h_v = center[:, :, 0].reshape(-1)
    s_v = center[:, :, 1].reshape(-1)
    v_v = center[:, :, 2].reshape(-1)
    valid = (s_v > 50) & (v_v > 50)
    return int(np.median(h_v[valid])) if valid.sum() >= 50 else -1


def tensor_to_display(tensor):
    """Converts an ImageNet-normalised CHW tensor to a HWC float32 numpy array ready for imshow."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy() * std + mean
    return np.clip(img, 0, 1)


def diagnose_crop_roi(bgr):
    """
    Runs crop_table_roi step-by-step and returns (status_str, corners_or_None).
    status_str is 'OK' on success or a 'FAIL: <reason>' string on the failing step.
    Useful for debugging which images the table detector cannot handle.
    """
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        center = hsv[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        h_vals = center[:, :, 0].reshape(-1)
        s_vals = center[:, :, 1].reshape(-1)
        v_vals = center[:, :, 2].reshape(-1)
        valid = (s_vals > 50) & (v_vals > 50)
        if valid.sum() < 50:
            return "FAIL: cloth colour (too few valid pixels)", None

        h_med = int(np.median(h_vals[valid]))
        mask = cv2.inRange(hsv,
                           np.array([max(0, h_med - 18), 70, 70]),
                           np.array([min(179, h_med + 18), 255, 255]))
        mask[:int(0.2 * h), :] = 0
        mask[int(0.9 * h):, :] = 0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if mask.sum() == 0:
            return "FAIL: mask empty after morphology", None

        n, labels, _, centroids = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return "FAIL: no connected components", None

        cp = np.array([w / 2, h / 2])
        best = min(range(1, n), key=lambda i: np.linalg.norm(centroids[i] - cp))
        comp = np.uint8(labels == best) * 255

        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "FAIL: no contours", None

        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        peri = cv2.arcLength(hull, True)
        corners = None
        for eps in [0.01, 0.02, 0.03]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                break
        if corners is None:
            corners = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)

        y_s = corners[np.argsort(corners[:, 1])]
        top, bot = y_s[:2], y_s[2:]
        tl, tr = top[np.argsort(top[:, 0])]
        bl, br = bot[np.argsort(bot[:, 0])]
        corners = np.array([tl, tr, br, bl], dtype=np.float32)

        cx, cy = corners[:, 0], corners[:, 1]
        area = 0.5 * abs(np.dot(cx, np.roll(cy, -1)) - np.dot(cy, np.roll(cx, -1)))
        if area < 1000:
            return f"FAIL: corners area too small ({area:.0f}px²)", None

        return "OK", corners

    except Exception as e:
        return f"FAIL: exception — {e}", None


######################## Utility helper functions (Task 1 only) ########################

def draw_boxes_rgb(img_rgb, boxes, labels=None, color=(255, 0, 0), thickness=2):
    '''
    Draws a bounding box on a RGB images.
    Boxes are expected in the format [x1, y1, x2, y2].
        - x1, y1: top-left corner of the box
        - x2, y2: bottom-right corner of the box
    Converts the image back to BGR format for drawing (default openCV color format) and converts it back to RGB before returning.
    Adds an optional label above the box if labels are provided.
    '''
    img = img_rgb.copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color[::-1], thickness)
        if labels is not None:
            cv2.putText(
                img_bgr, str(labels[i]), (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[::-1], 2, cv2.LINE_AA
            )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def normalize_box_xyxy(x, y, w, h, img_w, img_h):
    '''
    Normalizes the bounding box coordinates (x, y, w, h) to the range [0, 1] based on the image dimensions (img_w, img_h).
        - x, y: top-left corner of the box
        - w, h: width and height of the box 
    '''
    return [
        x / img_w,
        (x + w) / img_w,
        y / img_h,
        (y + h) / img_h
    ]

def denormalize_output_box(box, img_w, img_h):

    ''' 
    Denormalizes the bounding box coordinates from the range [0, 1] back to the image dimensions (img_w, img_h).
        - box: dictionary with keys "xmin", "xmax", "ymin", "ymax"
        - img_w, img_h: width and height of the image
    '''
    xmin = int(box["xmin"] * img_w)
    xmax = int(box["xmax"] * img_w)
    ymin = int(box["ymin"] * img_h)
    ymax = int(box["ymax"] * img_h)
    return [xmin, ymin, xmax, ymax]




######################## Load image list and optional ground truth (Task 1 only) ########################

def load_input_paths(input_json):
    '''
    Loads the image paths in the input JSON file.
    Returns a list of Path objects. 
    '''
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    paths = data.get("image_path") or data.get("image_paths") or []
    return [_PROJECT_ROOT / Path(p) for p in paths]

def load_output_example(output_json):
    ''' 
    Loads the output example JSON file.
    '''
    with open(output_json, "r", encoding="utf-8") as f:
        return json.load(f)
    




########################## Table detection helpers (Task 1 only) ##########################


def detect_table_mask_adaptive(bgr):
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
    ''' 
    Given a mask, extract the largest connected component closest to the image center
    '''
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




def extract_table_contour(component_mask):
    '''
    Receives a binary mask of the main component and extracts its contour
    '''
    if component_mask is None:
        return None

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    contour = cv2.convexHull(contour)
    return contour





def contour_to_corners_refined(contour):
    if contour is None:
        return None

    peri = cv2.arcLength(contour, True)
    print("Contour perimeter:", peri)

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