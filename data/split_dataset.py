"""
One-time script to split the dataset into train/valid/test sets (80/10/10).
Uses stratified sampling by ball count so each split has a similar distribution.

Run from the project root:
    python data/split_dataset.py

NOTE: The Dot class (id=2) in YOLO labels are table rail markers, not balls.
Ball count = number of annotations where class_id != 2.
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "8-Ball Pool.v3i.yolov11"
TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_RATIO = 0.10
TEST_RATIO  = 0.10
SEED        = 42


def count_balls(label_path: Path) -> int:
    lines = label_path.read_text().splitlines()
    return sum(1 for l in lines if l.strip() and not l.startswith("2 "))


def collect_groups() -> dict[int, list[str]]:
    groups = defaultdict(list)
    for img_path in sorted(TRAIN_IMAGES.glob("*.jpg")):
        label_path = TRAIN_LABELS / (img_path.stem + ".txt")
        n = count_balls(label_path) if label_path.exists() else 0
        groups[n].append(img_path.stem)
    return groups


def stratified_split(groups: dict) -> tuple[list, list]:
    random.seed(SEED)
    valid_stems, test_stems = [], []

    for n_balls, stems in sorted(groups.items()):
        random.shuffle(stems)
        n = len(stems)
        n_valid = max(1, round(n * VALID_RATIO)) if n >= 5 else 0
        n_test  = max(1, round(n * TEST_RATIO))  if n >= 5 else 0
        n_test  = min(n_test, n - n_valid)
        valid_stems.extend(stems[:n_valid])
        test_stems.extend(stems[n_valid : n_valid + n_test])

    return valid_stems, test_stems


def move_stems(stems: list[str], split: str) -> None:
    dest_images = DATASET_DIR / split / "images"
    dest_labels = DATASET_DIR / split / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        src_img = TRAIN_IMAGES / (stem + ".jpg")
        src_lbl = TRAIN_LABELS / (stem + ".txt")
        if src_img.exists():
            shutil.move(str(src_img), str(dest_images / src_img.name))
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dest_labels / src_lbl.name))


def print_distribution(split: str, folder: Path) -> None:
    groups = defaultdict(int)
    for lbl in (folder / "labels").glob("*.txt"):
        n = count_balls(lbl)
        groups[n] += 1
    dist = " | ".join(f"{k}b:{v}" for k, v in sorted(groups.items()))
    total = sum(groups.values())
    print(f"  {split:<6} {total:>3} images  [{dist}]")


def main() -> None:
    for split in ("valid", "test"):
        dest = DATASET_DIR / split / "images"
        if dest.exists() and any(dest.iterdir()):
            print(f"ERROR: {split}/ already has images. Remove it first to re-run.")
            return

    groups = collect_groups()
    valid_stems, test_stems = stratified_split(groups)

    move_stems(valid_stems, "valid")
    move_stems(test_stems,  "test")

    total = sum(len(v) for v in groups.values())
    n_train = len(list(TRAIN_IMAGES.glob("*.jpg")))
    n_valid = len(valid_stems)
    n_test  = len(test_stems)

    print(f"\nSplit complete  (total={total}, seed={SEED})")
    print(f"  train  {n_train:>3} ({n_train/total:.0%})")
    print(f"  valid  {n_valid:>3} ({n_valid/total:.0%})")
    print(f"  test   {n_test:>3} ({n_test/total:.0%})")
    print("\nBall-count distribution per split:")
    print_distribution("train", DATASET_DIR / "train")
    print_distribution("valid", DATASET_DIR / "valid")
    print_distribution("test",  DATASET_DIR / "test")


if __name__ == "__main__":
    main()
