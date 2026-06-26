from pathlib import Path
import argparse
import json

import cv2
import yaml

from utils import load_detection_model
from analyze_clustering import nearest_neighbor_ratio


COLOR_CORRECT = (0, 200, 0)        # green
COLOR_MISCLASSIFIED = (0, 165, 255)  # orange
COLOR_MISSED = (0, 0, 255)         # red
COLOR_FALSE_POSITIVE = (255, 0, 255)  # magenta


def load_config(config_path: Path) -> dict:
    """
    Load configuration from a YAML file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list:
    """
    Load YOLO-format ground-truth boxes and convert to pixel xyxy.
    """
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        class_id, cx, cy, w, h = map(float, line.split())
        cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
        boxes.append((int(class_id), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))

    return boxes


def box_iou(box_a: tuple, box_b: tuple) -> float:
    """
    Intersection-over-union between two xyxy boxes.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_predictions(gt_boxes: list, pred_boxes: list, iou_threshold: float) -> dict:
    """
    Greedily match predictions to ground truth by descending confidence,
    using the same IoU>=threshold rule mAP itself uses for a true positive.

    Returns matched ground-truth indices and matched prediction indices,
    plus which matches are class-correct vs misclassified.
    """
    matched_gt = set()
    matched_pred = set()
    misclassified_pred = set()

    order = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i][5], reverse=True)

    for pi in order:
        pred = pred_boxes[pi]
        best_iou, best_gi = 0.0, None

        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = box_iou(pred[1:5], gt[1:5])
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_gi is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            if gt_boxes[best_gi][0] != pred[0]:
                misclassified_pred.add(pi)

    return {
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "misclassified_pred": misclassified_pred,
    }


def draw_box(image, box, color, label, offset=(0.0, 0.0), scale=1.0, font_scale=0.5):
    ox, oy = offset
    x1, y1, x2, y2 = (
        int((box[0] - ox) * scale), int((box[1] - oy) * scale),
        int((box[2] - ox) * scale), int((box[3] - oy) * scale),
    )
    thickness = max(1, round(2 * scale))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    label_y = y1 if y1 - text_h - 6 >= 0 else y2 + text_h + 6
    label_top = label_y - text_h - 6 if y1 - text_h - 6 >= 0 else y2
    cv2.rectangle(image, (x1, label_top), (x1 + text_w + 4, label_top + text_h + 6), color, -1)
    cv2.putText(
        image, label, (x1 + 2, label_top + text_h + 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_matches(image, gt_boxes, pred_boxes, match, names, offset=(0.0, 0.0), scale=1.0, font_scale=0.5):
    """
    Draw color-coded ground-truth/prediction boxes onto `image`, optionally
    translating by `offset` (in source-image pixels) and scaling - used both
    for the full annotated image (offset=0, scale=1) and for zoomed crops
    (offset=crop origin, scale=zoom factor) so labels stay legible either way.
    """
    for gi, gt in enumerate(gt_boxes):
        if gi not in match["matched_gt"]:
            draw_box(image, gt[1:5], COLOR_MISSED, f"MISSED:{names[gt[0]]}", offset, scale, font_scale)

    for pi, pred in enumerate(pred_boxes):
        class_id, x1, y1, x2, y2, conf = pred
        if pi not in match["matched_pred"]:
            draw_box(image, (x1, y1, x2, y2), COLOR_FALSE_POSITIVE, f"FP:{names[class_id]} {conf:.2f}", offset, scale, font_scale)
        elif pi in match["misclassified_pred"]:
            draw_box(image, (x1, y1, x2, y2), COLOR_MISCLASSIFIED, f"{names[class_id]} {conf:.2f}", offset, scale, font_scale)
        else:
            draw_box(image, (x1, y1, x2, y2), COLOR_CORRECT, f"{names[class_id]} {conf:.2f}", offset, scale, font_scale)

    return image


def build_error_zoom(raw_image, img_w, img_h, gt_boxes, pred_boxes, match, names,
                      padding_ratio=0.8, max_area_ratio=0.35, target_min_dim=640):
    """
    Crop tightly around just the erroring boxes (missed/misclassified/false
    positive) from the *raw* (unannotated) image, upscale, then draw boxes
    and labels at that upscaled resolution. Drawing after the resize (rather
    than stretching pre-drawn labels) keeps text legible even when several
    boxes sit only a few pixels apart. Returns None if there's nothing to
    zoom into, or if the error region is already most of the image.
    """
    error_boxes = []

    for gi, gt in enumerate(gt_boxes):
        if gi not in match["matched_gt"]:
            error_boxes.append(gt[1:5])

    for pi, pred in enumerate(pred_boxes):
        if pi not in match["matched_pred"] or pi in match["misclassified_pred"]:
            error_boxes.append(pred[1:5])

    if not error_boxes:
        return None

    x1 = min(b[0] for b in error_boxes)
    y1 = min(b[1] for b in error_boxes)
    x2 = max(b[2] for b in error_boxes)
    y2 = max(b[3] for b in error_boxes)

    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = box_w * padding_ratio, box_h * padding_ratio

    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(img_w, int(x2 + pad_x))
    crop_y2 = min(img_h, int(y2 + pad_y))

    crop_area = (crop_x2 - crop_x1) * (crop_y2 - crop_y1)
    if crop_area >= max_area_ratio * img_w * img_h:
        return None

    crop = raw_image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    scale = max(1.0, target_min_dim / min(crop.shape[0], crop.shape[1]))
    scale = min(scale, 6.0)
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    draw_matches(
        crop, gt_boxes, pred_boxes, match, names,
        offset=(crop_x1, crop_y1), scale=scale, font_scale=0.6,
    )

    return crop


def analyze_image(model, image_path: Path, label_path: Path, imgsz: int, conf: float, iou: float):
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    result = results[0]
    img_h, img_w = result.orig_shape

    gt_boxes = load_ground_truth(label_path, img_w, img_h)
    pred_boxes = [
        (int(box.cls[0].item()), *box.xyxy[0].tolist(), float(box.conf[0].item()))
        for box in result.boxes
    ]

    match = match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5)

    n_missed = len(gt_boxes) - len(match["matched_gt"])
    n_false_positive = len(pred_boxes) - len(match["matched_pred"])
    n_misclassified = len(match["misclassified_pred"])

    raw_image = cv2.imread(str(image_path))
    annotated = draw_matches(raw_image.copy(), gt_boxes, pred_boxes, match, model.names)
    zoom = build_error_zoom(raw_image, img_w, img_h, gt_boxes, pred_boxes, match, model.names)

    min_ratio, _ = nearest_neighbor_ratio(gt_boxes)

    return {
        "n_gt": len(gt_boxes),
        "n_missed": n_missed,
        "n_false_positive": n_false_positive,
        "n_misclassified": n_misclassified,
        "has_error": (n_missed + n_false_positive + n_misclassified) > 0,
        "min_ratio": min_ratio,
        "annotated_image": annotated,
        "zoom_image": zoom,
    }


def empty_totals() -> dict:
    return {"gt": 0, "missed": 0, "fp": 0, "miscls": 0, "n_images": 0, "images_with_errors": 0}


def add_to_totals(totals: dict, analysis: dict) -> None:
    totals["gt"] += analysis["n_gt"]
    totals["missed"] += analysis["n_missed"]
    totals["fp"] += analysis["n_false_positive"]
    totals["miscls"] += analysis["n_misclassified"]
    totals["n_images"] += 1
    if analysis["has_error"]:
        totals["images_with_errors"] += 1


def print_totals(label: str, totals: dict) -> None:
    print(f"\n--- {label}: {totals['n_images']} images, {totals['gt']} ground-truth balls ---")
    print(f"Missed: {totals['missed']}  False positives: {totals['fp']}  Misclassified: {totals['miscls']}")
    print(f"Images with at least one error: {totals['images_with_errors']}/{totals['n_images']}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw color-coded correct/missed/misclassified/false-positive boxes "
                    "on every image of a split, for visual error analysis."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions.")
    parser.add_argument(
        "--iou", type=float, default=None,
        help="NMS IoU threshold override for predictions. Defaults to the config's iou_threshold.",
    )
    parser.add_argument("--only-errors", action="store_true", help="Only save images that have at least one error.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save annotated images.")
    parser.add_argument(
        "--clustered-threshold", type=float, default=0.5,
        help="A test image is reported as 'clustered' if its closest ground-truth ball "
             "pair has nearest_neighbor_ratio <= this value (matches oversample_clustered.py).",
    )
    parser.add_argument(
        "--save-json", type=str, default=None,
        help="Optional path to save the overall/clustered/non_clustered totals as JSON.",
    )
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = current_dir / config_path
    config = load_config(config_path)

    dataset_yaml_path = current_dir / config["dataset_yaml"]
    dataset_root = Path(yaml.safe_load(dataset_yaml_path.read_text())["path"])
    if not dataset_root.is_absolute():
        # Ultralytics resolves a dataset yaml's relative "path:" against the
        # caller's working directory, which is always ball_detection/ for
        # these scripts (current_dir) - not against the yaml file's own
        # location.
        dataset_root = (current_dir / dataset_root).resolve()

    split_dir = {"val": "valid", "test": "test"}[args.split]
    images_dir = dataset_root / split_dir / "images"
    labels_dir = dataset_root / split_dir / "labels"

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = current_dir / model_path

    model = load_detection_model(model_path)
    imgsz = config.get("imgsz", 640)
    iou_threshold = args.iou if args.iou is not None else config.get("iou_threshold", 0.5)

    if args.output_dir is None:
        run_label = model_path.parent.parent.name if model_path.parent.name == "weights" else model_path.stem
        output_dir = current_dir / "outputs" / "error_analysis" / f"{run_label}_{args.split}"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = current_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    print(f"{'image':<55} {'gt':>3} {'missed':>6} {'fp':>3} {'miscls':>6}  cluster")

    totals_overall = empty_totals()
    totals_clustered = empty_totals()
    totals_non_clustered = empty_totals()

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        analysis = analyze_image(
            model, image_path, label_path,
            imgsz=imgsz, conf=args.conf, iou=iou_threshold,
        )

        is_clustered = (
            analysis["min_ratio"] is not None and analysis["min_ratio"] <= args.clustered_threshold
        )

        add_to_totals(totals_overall, analysis)
        add_to_totals(totals_clustered if is_clustered else totals_non_clustered, analysis)

        if analysis["has_error"]:
            print(
                f"{image_path.name:<55} {analysis['n_gt']:>3} "
                f"{analysis['n_missed']:>6} {analysis['n_false_positive']:>3} "
                f"{analysis['n_misclassified']:>6}  {'yes' if is_clustered else 'no'}"
            )

        if args.only_errors and not analysis["has_error"]:
            continue

        output_path = output_dir / f"{image_path.stem}_error.jpg"
        cv2.imwrite(str(output_path), analysis["annotated_image"])

        if analysis["zoom_image"] is not None:
            zoom_path = output_dir / f"{image_path.stem}_error_zoom.jpg"
            cv2.imwrite(str(zoom_path), analysis["zoom_image"])

    print_totals("Overall", totals_overall)
    print_totals(f"Clustered (min_ratio <= {args.clustered_threshold})", totals_clustered)
    print_totals("Non-clustered", totals_non_clustered)
    print(f"\nAnnotated images saved to: {output_dir}")
    print("Legend: green=correct  orange=misclassified  red=missed  magenta=false positive")

    if args.save_json:
        save_json_path = Path(args.save_json)
        if not save_json_path.is_absolute():
            save_json_path = current_dir / save_json_path
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "model": str(model_path),
                    "split": args.split,
                    "conf": args.conf,
                    "clustered_threshold": args.clustered_threshold,
                    "overall": totals_overall,
                    "clustered": totals_clustered,
                    "non_clustered": totals_non_clustered,
                },
                file, indent=4,
            )
        print(f"Stratified totals saved to: {save_json_path}")


if __name__ == "__main__":
    main()
