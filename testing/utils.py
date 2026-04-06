from pathlib import Path
import json
import math
import os
from collections import defaultdict, Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import PROJECT_ROOT  

plt.rcParams["figure.figsize"] = (10, 6)


######################## Utility helper functions ########################

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




######################## Load image list and optional ground truth ########################

def load_input_paths(input_json):
    '''
    Loads the image paths in the input JSON file.
    Returns a list of Path objects. 
    '''
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    paths = data.get("image_path") or data.get("image_paths") or []
    return [PROJECT_ROOT / Path(p) for p in paths]

def load_output_example(output_json):
    ''' 
    Loads the output example JSON file.
    '''
    with open(output_json, "r", encoding="utf-8") as f:
        return json.load(f)