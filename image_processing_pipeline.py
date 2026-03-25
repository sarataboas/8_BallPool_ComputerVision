# Imports
import os
import json
import logging
from typing import Any, Optional, Dict, Tuple, List

import cv2
import numpy as np

# ----------- CONFIGS --> all directories and variables used in the script -----------
DEVELOPMENT_SET = 'development_set/'
INPUT_FILE = 'example_json/input.json'
TOPVIEW_RESULTS = 'results/topview_images/'
OUTPUT_FILE = 'results/output.json'

# Table output size after homography
WARPED_TABLE_WIDTH = 800
WARPED_TABLE_HEIGHT = 400

# Ball filtering parameters
MIN_BALL_AREA = 80
MAX_BALL_AREA = 5000
MIN_CIRCULARITY = 0.45

# HSV thresholds for table cloth --- TO DO: require tuning !!! (não sei o que são os valores)
TABLE_BLUE_LOWER = np.array([85, 40, 40])
TABLE_BLUE_UPPER = np.array([130, 255, 255])

TABLE_GREEN_LOWER = np.array([35, 30, 30])
TABLE_GREEN_UPPER = np.array([90, 255, 255])

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


# ------------- SCHEMAS --> define the input and output format --------------



# --------------- Load Input JSON (with the images) --> just loads the example file ---------------

# def load_example_input_json(input_file: str) -> json:
#     '''
#     Loads the JSON input file that contains all image paths in `DEVELOPMENT_SET`
#     '''
#     if not os.path.exists(input_file):
#         raise FileNotFoundError(f"Input file not found in {input_file}")
    
#     with open(input_file) as f:
#         input = json.loads(f)
#     return input

def create_input_json(img_dir: str) -> Dict[str, List[str]]:
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Input directory not found in {img_dir}")

    img_paths = []
    for img in os.listdir(img_dir): 
        if img.endswith('.jpg'):
            full_path = os.path.join(img_dir, img)
            img_paths.append(full_path)
    input_file = {
        "image_paths": img_paths
    }
    # print(len(img_paths))
    return input_file # TO DO: add here the input validation schema
    


# -------------------- Preprocessing + Task implementation -------------------------

def load_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path) # TO DO: confirm if this is the loading method of classes
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    return img

def save_json(data: dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# UTILS - TO DO: não vi estas funções 
def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points as:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def contour_to_corners(contour: np.ndarray) -> Optional[np.ndarray]:
    """
    Try to approximate table contour to 4 corners.
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        return order_points(pts)

    # fallback: use min area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_points(box)

def compute_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4.0 * np.pi * area) / (perimeter * perimeter)

def get_ball_mask(top_view_img: np.ndarray) -> np.ndarray:
    """
    Segment balls by removing table cloth color.
    """
    hsv = cv2.cvtColor(top_view_img, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, TABLE_BLUE_LOWER, TABLE_BLUE_UPPER)
    mask_green = cv2.inRange(hsv, TABLE_GREEN_LOWER, TABLE_GREEN_UPPER)
    table_mask = cv2.bitwise_or(mask_blue, mask_green)

    # Balls = not table
    balls_mask = cv2.bitwise_not(table_mask)

    # Clean mask
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    balls_mask = cv2.morphologyEx(balls_mask, cv2.MORPH_OPEN, kernel_open)
    balls_mask = cv2.morphologyEx(balls_mask, cv2.MORPH_CLOSE, kernel_close)

    return balls_mask

# ---------------------------------------------------------------------------------------------------


# Pipeline
# Image → Segment table → Warp → Segment balls → Detect → Classify → Output

def detect_table(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
        table_mask, largest_table_contour, ordered_4_corners
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, TABLE_BLUE_LOWER, TABLE_BLUE_UPPER)
    mask_green = cv2.inRange(hsv, TABLE_GREEN_LOWER, TABLE_GREEN_UPPER)
    table_mask = cv2.bitwise_or(mask_blue, mask_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return table_mask, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    corners = contour_to_corners(largest_contour)

    return table_mask, largest_contour, corners

def warp_table():
    pass

def detect_balls():
    pass

def classify_balls():
    pass

# Create and Return Output JSON

# Output directory with top-view images