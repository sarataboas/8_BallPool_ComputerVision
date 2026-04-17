import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Load JSON ---
with open("output_example.json", "r") as f:
    data = json.load(f)

entry = data[0]
image_path = entry["image_path"]
balls = entry["balls"]

# --- Load image ---
bgr = cv2.imread(image_path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
ih, iw = bgr.shape[:2]

# --- Draw bounding boxes ---
vis = rgb.copy()
for ball in balls:
    x1 = int(ball["xmin"] * iw)
    x2 = int(ball["xmax"] * iw)
    y1 = int(ball["ymin"] * ih)
    y2 = int(ball["ymax"] * ih)

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, str(ball["number"]), (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

show(vis, title=f"GT boxes — {len(balls)} balls", figsize=(10, 7))