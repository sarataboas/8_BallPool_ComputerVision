"""
One-time script to split the dataset into train/valid/test according to
data/partition.csv (the authoritative split — do not use split_dataset.py).

All images currently live in train/images and train/labels. This moves each
image+label pair to its destination split folder based on the CSV.

Run from the project root:
    python data/apply_partition_split.py
"""

import csv
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "8-Ball Pool.v3i.yolov11"
PARTITION_CSV = PROJECT_ROOT / "data" / "partition.csv"

SRC_IMAGES = DATASET_DIR / "train" / "images"
SRC_LABELS = DATASET_DIR / "train" / "labels"

SPLITS = ("train", "valid", "test")


def load_partition() -> list[tuple[str, str]]:
    with open(PARTITION_CSV, newline="") as f:
        reader = csv.DictReader(f)
        return [(row["image_name"].strip(), row["partition"].strip()) for row in reader]


def move_pair(image_name: str, split: str) -> None:
    stem = Path(image_name).stem
    dest_images = DATASET_DIR / split / "images"
    dest_labels = DATASET_DIR / split / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    src_img = SRC_IMAGES / image_name
    src_lbl = SRC_LABELS / (stem + ".txt")

    if split == "train":
        return  # already in place

    if src_img.exists():
        shutil.move(str(src_img), str(dest_images / src_img.name))
    if src_lbl.exists():
        shutil.move(str(src_lbl), str(dest_labels / src_lbl.name))


def main() -> None:
    for split in ("valid", "test"):
        dest = DATASET_DIR / split / "images"
        if dest.exists() and any(dest.iterdir()):
            print(f"ERROR: {split}/ already has images. Remove it first to re-run.")
            return

    partition = load_partition()
    counts = {split: 0 for split in SPLITS}

    for image_name, split in partition:
        move_pair(image_name, split)
        counts[split] += 1

    print("Split complete (source: data/partition.csv)")
    for split in SPLITS:
        n_img = len(list((DATASET_DIR / split / "images").glob("*")))
        n_lbl = len(list((DATASET_DIR / split / "labels").glob("*")))
        print(f"  {split:<6} csv_rows={counts[split]:>3}  images={n_img:>3}  labels={n_lbl:>3}")


if __name__ == "__main__":
    main()
