from pathlib import Path
import argparse
import random
import shutil

import cv2
import numpy as np
import yaml

from analyze_clustering import load_boxes, nearest_neighbor_ratio


NEW_CLASS_NAMES = ["Black", "Cue", "Solid", "Striped"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def list_images(images_dir: Path):
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_split(source_split_dir: Path, destination_split_dir: Path) -> None:
    """
    Copy a whole split (images + labels) unchanged.
    """
    source_images_dir = source_split_dir / "images"
    source_labels_dir = source_split_dir / "labels"

    destination_images_dir = destination_split_dir / "images"
    destination_labels_dir = destination_split_dir / "labels"

    destination_images_dir.mkdir(parents=True, exist_ok=True)
    destination_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path in list_images(source_images_dir):
        shutil.copy2(image_path, destination_images_dir / image_path.name)

        label_path = source_labels_dir / f"{image_path.stem}.txt"
        destination_label_path = destination_labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, destination_label_path)
        else:
            destination_label_path.write_text("", encoding="utf-8")


def find_clustered_images(images_dir: Path, labels_dir: Path, threshold: float) -> list:
    """
    Return train images whose closest ball pair has min_ratio <= threshold,
    i.e. balls that are touching/almost touching (ratio ~1.0 = touching).
    """
    clustered = []

    for image_path in list_images(images_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = load_boxes(label_path, img_w, img_h)

        min_ratio, _ = nearest_neighbor_ratio(boxes)
        if min_ratio is not None and min_ratio <= threshold:
            clustered.append(image_path)

    return clustered


def flip_horizontal(image, label_lines: list) -> tuple:
    """
    Mirror the image left-right and flip only the x-center of each box.
    """
    flipped = cv2.flip(image, 1)

    new_lines = []
    for line in label_lines:
        class_id, cx, cy, w, h = line.split()
        new_cx = 1.0 - float(cx)
        new_lines.append(f"{class_id} {new_cx:.6f} {cy} {w} {h}")

    return flipped, new_lines


def jitter_photometric(image, rng: np.random.Generator):
    """
    Random brightness/contrast/saturation/noise jitter. Boxes are unaffected
    since pixel positions don't change.
    """
    alpha = rng.uniform(0.8, 1.2)  # contrast
    beta = rng.uniform(-25, 25)    # brightness
    img = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    noise = rng.normal(0, 6, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def jitter_geometric(image, label_lines: list, rng: np.random.Generator) -> tuple:
    """
    Small random rotation + scale + translation. This is the part that
    actually changes the spatial arrangement of touching balls (unlike flip,
    which is an exact mirror, or photometric jitter, which doesn't move
    anything) - giving the model new geometry for the same cluster, not just
    new colors.

    Boxes are recomputed as the axis-aligned bounding box of the
    rotated/scaled/translated corners, then clipped to the image. A box that
    loses more than half its area to clipping (e.g. pushed off-frame) is
    dropped rather than kept as a truncated, mislabeled box.
    """
    img_h, img_w = image.shape[:2]

    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-0.04, 0.04) * img_w
    ty = rng.uniform(-0.04, 0.04) * img_h

    matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    warped = cv2.warpAffine(image, matrix, (img_w, img_h), borderMode=cv2.BORDER_REPLICATE)

    new_lines = []
    for line in label_lines:
        class_id, cx, cy, bw, bh = line.split()
        cx, cy = float(cx) * img_w, float(cy) * img_h
        bw, bh = float(bw) * img_w, float(bh) * img_h

        corners = np.array([
            [cx - bw / 2, cy - bh / 2],
            [cx + bw / 2, cy - bh / 2],
            [cx - bw / 2, cy + bh / 2],
            [cx + bw / 2, cy + bh / 2],
        ])
        warped_corners = corners @ matrix[:, :2].T + matrix[:, 2]

        x1, y1 = warped_corners[:, 0].min(), warped_corners[:, 1].min()
        x2, y2 = warped_corners[:, 0].max(), warped_corners[:, 1].max()

        clipped_x1, clipped_y1 = max(0.0, x1), max(0.0, y1)
        clipped_x2, clipped_y2 = min(float(img_w), x2), min(float(img_h), y2)

        new_w, new_h = clipped_x2 - clipped_x1, clipped_y2 - clipped_y1
        if new_w <= 0 or new_h <= 0 or (new_w * new_h) < 0.5 * (bw * bh):
            continue

        new_cx = (clipped_x1 + clipped_x2) / 2 / img_w
        new_cy = (clipped_y1 + clipped_y2) / 2 / img_h
        new_lines.append(
            f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_w / img_w:.6f} {new_h / img_h:.6f}"
        )

    return warped, new_lines


def make_augmented_copy(image, label_lines: list, rng: np.random.Generator) -> tuple:
    out_image, out_lines = image, label_lines

    if rng.random() < 0.5:
        out_image, out_lines = flip_horizontal(out_image, out_lines)

    out_image, out_lines = jitter_geometric(out_image, out_lines, rng)
    out_image = jitter_photometric(out_image, rng)

    return out_image, out_lines


def oversample_train_split(
    source_train_dir: Path,
    destination_train_dir: Path,
    threshold: float,
    copies_per_image: int,
    seed: int,
) -> int:
    """
    Add `copies_per_image` augmented duplicates of every clustered training
    image (already copied unchanged by copy_split) to the destination train
    split. Returns the number of new images added.
    """
    images_dir = source_train_dir / "images"
    labels_dir = source_train_dir / "labels"

    clustered_images = find_clustered_images(images_dir, labels_dir, threshold)
    print(f"Clustered training images (min_ratio <= {threshold}): {len(clustered_images)}")

    destination_images_dir = destination_train_dir / "images"
    destination_labels_dir = destination_train_dir / "labels"

    rng = np.random.default_rng(seed)
    n_added = 0

    for image_path in clustered_images:
        image = cv2.imread(str(image_path))
        label_path = labels_dir / f"{image_path.stem}.txt"
        label_lines = [
            line for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        for copy_idx in range(copies_per_image):
            aug_image, aug_lines = make_augmented_copy(image, label_lines, rng)

            new_name = f"{image_path.stem}_clusteraug{copy_idx}"
            cv2.imwrite(str(destination_images_dir / f"{new_name}{image_path.suffix}"), aug_image)

            text = "\n".join(aug_lines)
            if text:
                text += "\n"
            (destination_labels_dir / f"{new_name}.txt").write_text(text, encoding="utf-8")

            n_added += 1

    return n_added


def create_data_yaml(destination_dataset_dir: Path) -> None:
    data = {
        "path": str(destination_dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(NEW_CLASS_NAMES),
        "names": NEW_CLASS_NAMES,
    }

    output_yaml_path = destination_dataset_dir / "data.yaml"
    with open(output_yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    print(f"Created data.yaml at: {output_yaml_path}")


def count_files(dataset_dir: Path) -> None:
    print("\nFinal dataset counts:")
    for split in ["train", "valid", "test"]:
        images_dir = dataset_dir / split / "images"
        n_images = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        print(f"  {split:<6} images={n_images}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a copy of a YOLO dataset where training images with "
                    "touching/clustered balls are oversampled with photometric "
                    "augmentation (flip + brightness/contrast/saturation/noise jitter). "
                    "valid and test are copied unchanged."
    )
    parser.add_argument("--dataset-dir", type=str, required=True, help="Source YOLO dataset root.")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination dataset root.")
    parser.add_argument("--threshold", type=float, default=0.5, help="min_ratio <= threshold counts as clustered.")
    parser.add_argument("--copies", type=int, default=3, help="Augmented copies to add per clustered image.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dataset_dir = Path(args.dataset_dir)
    if not source_dataset_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset_dir}")

    if args.output_dir is None:
        destination_dataset_dir = source_dataset_dir.parent / f"{source_dataset_dir.name}_oversampled_clusters"
    else:
        destination_dataset_dir = Path(args.output_dir)

    if destination_dataset_dir.exists():
        print(f"Removing old oversampled dataset: {destination_dataset_dir}")
        shutil.rmtree(destination_dataset_dir)
    destination_dataset_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        copy_split(
            source_split_dir=source_dataset_dir / split,
            destination_split_dir=destination_dataset_dir / split,
        )

    n_added = oversample_train_split(
        source_train_dir=source_dataset_dir / "train",
        destination_train_dir=destination_dataset_dir / "train",
        threshold=args.threshold,
        copies_per_image=args.copies,
        seed=args.seed,
    )

    create_data_yaml(destination_dataset_dir)
    count_files(destination_dataset_dir)

    print(f"\nAdded {n_added} augmented images to train.")
    print(f"Oversampled dataset created at: {destination_dataset_dir}")


if __name__ == "__main__":
    main()
