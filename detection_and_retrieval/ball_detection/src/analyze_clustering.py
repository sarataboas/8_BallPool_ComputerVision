from pathlib import Path
import argparse
import csv
import math

import cv2


def load_boxes(label_path: Path, img_w: int, img_h: int) -> list:
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


def box_center(box: tuple) -> tuple:
    _, x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_diagonal(box: tuple) -> float:
    _, x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)


def nearest_neighbor_ratio(boxes: list) -> tuple:
    """
    For each box, find the distance to its nearest neighbor (by center),
    normalized by the average diagonal of the pair.

    A ratio close to 1.0 means the two balls are touching; a ratio of
    ~2.7 or higher means they are clearly separated. Returns
    (min_ratio, avg_ratio) across all boxes in the image, or (None, None)
    if there are fewer than 2 boxes (no pair to measure).
    """
    if len(boxes) < 2:
        return None, None

    centers = [box_center(box) for box in boxes]
    diagonals = [box_diagonal(box) for box in boxes]

    ratios = []
    for i in range(len(boxes)):
        best_ratio = None
        for j in range(len(boxes)):
            if i == j:
                continue
            dist = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            avg_diag = (diagonals[i] + diagonals[j]) / 2
            ratio = dist / avg_diag if avg_diag > 0 else float("inf")
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
        ratios.append(best_ratio)

    return min(ratios), sum(ratios) / len(ratios)


def analyze_split(images_dir: Path, labels_dir: Path) -> list:
    """
    Compute clustering metrics for every image in a split.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )

    results = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: could not read image {image_path}, skipping")
            continue
        img_h, img_w = image.shape[:2]

        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = load_boxes(label_path, img_w, img_h)

        min_ratio, avg_ratio = nearest_neighbor_ratio(boxes)

        results.append({
            "image": image_path.name,
            "n_balls": len(boxes),
            "min_ratio": min_ratio,
            "avg_ratio": avg_ratio,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Rank YOLO-format dataset images by ball clustering "
                     "(touching/close balls vs well-separated balls)."
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to a YOLO dataset root containing train/valid/test split folders.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to analyze: train, valid, test, or 'all'.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="min_ratio at or below which an image is considered clustered/touching.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save per-image metrics as CSV.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    all_results = []
    for split in splits:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"

        if not images_dir.exists():
            print(f"Skipping missing split: {images_dir}")
            continue

        split_results = analyze_split(images_dir, labels_dir)
        for result in split_results:
            result["split"] = split
        all_results.extend(split_results)

    if not all_results:
        print("No images found.")
        return

    # Sort: images with a measurable pair first (most clustered first),
    # then images with fewer than 2 balls (not measurable) at the end.
    measurable = [r for r in all_results if r["min_ratio"] is not None]
    unmeasurable = [r for r in all_results if r["min_ratio"] is None]
    measurable.sort(key=lambda r: r["min_ratio"])

    clustered = [r for r in measurable if r["min_ratio"] <= args.threshold]

    print(f"Total images analyzed: {len(all_results)}")
    print(f"Images with >=2 balls (measurable): {len(measurable)}")
    print(f"Images with <2 balls (skipped, no pair): {len(unmeasurable)}")
    print(f"Clustered images (min_ratio <= {args.threshold}): {len(clustered)}")

    print(f"\nTop 15 most clustered images:")
    print(f"{'image':<55} {'split':<6} {'n_balls':<8} {'min_ratio':<10} {'avg_ratio':<10}")
    for result in measurable[:15]:
        print(
            f"{result['image']:<55} {result['split']:<6} {result['n_balls']:<8} "
            f"{result['min_ratio']:<10.3f} {result['avg_ratio']:<10.3f}"
        )

    print(f"\nTop 5 most separated images:")
    print(f"{'image':<55} {'split':<6} {'n_balls':<8} {'min_ratio':<10} {'avg_ratio':<10}")
    for result in measurable[-5:]:
        print(
            f"{result['image']:<55} {result['split']:<6} {result['n_balls']:<8} "
            f"{result['min_ratio']:<10.3f} {result['avg_ratio']:<10.3f}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["image", "split", "n_balls", "min_ratio", "avg_ratio"])
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
        print(f"\nPer-image metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
