from pathlib import Path
import shutil
import yaml


# Class mapping from original dataset to new dataset
# Original:
# 0 = Black
# 1 = Cue
# 2 = Dot
# 3 = Solid
# 4 = Striped
#
# New:
# 0 = Black
# 1 = Cue
# 2 = Solid
# 3 = Striped

DOT_CLASS_ID = 2

CLASS_ID_MAPPING = {
    0: 0,  # Black
    1: 1,  # Cue
    3: 2,  # Solid
    4: 3,  # Striped
}

NEW_CLASS_NAMES = ["Black", "Cue", "Solid", "Striped"]


def process_label_file(source_label_path: Path, destination_label_path: Path):
    """
    Copy one YOLO label file while removing Dot annotations
    and remapping class IDs.
    """
    new_lines = []

    if source_label_path.exists():
        with open(source_label_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            class_id = int(parts[0])

            # Remove Dot labels
            if class_id == DOT_CLASS_ID:
                continue

            # Keep only known classes
            if class_id not in CLASS_ID_MAPPING:
                print(f"Warning: unknown class id {class_id} in {source_label_path}")
                continue

            new_class_id = CLASS_ID_MAPPING[class_id]

            # Replace old class id with new class id
            parts[0] = str(new_class_id)
            new_lines.append(" ".join(parts))

    destination_label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(destination_label_path, "w", encoding="utf-8") as file:
        for line in new_lines:
            file.write(line + "\n")


def copy_split_without_dot(source_split_dir: Path, destination_split_dir: Path):
    """
    Copy images and labels from one split: train, valid or test.
    """
    source_images_dir = source_split_dir / "images"
    source_labels_dir = source_split_dir / "labels"

    destination_images_dir = destination_split_dir / "images"
    destination_labels_dir = destination_split_dir / "labels"

    if not source_images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {source_images_dir}")

    if not source_labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {source_labels_dir}")

    destination_images_dir.mkdir(parents=True, exist_ok=True)
    destination_labels_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    image_paths = [
        path for path in source_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    print(f"Processing split: {source_split_dir.name}")
    print(f"Images found: {len(image_paths)}")

    for image_path in image_paths:
        # Copy image
        destination_image_path = destination_images_dir / image_path.name
        shutil.copy2(image_path, destination_image_path)

        # Process corresponding label
        source_label_path = source_labels_dir / f"{image_path.stem}.txt"
        destination_label_path = destination_labels_dir / f"{image_path.stem}.txt"

        process_label_file(source_label_path, destination_label_path)


def create_data_yaml(destination_dataset_dir: Path):
    """
    Create a new data.yaml for the dataset without Dot.
    """
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

    print(f"Created new data.yaml at: {output_yaml_path}")


def main():
    # This script is inside ball_detection/src
    # parents[2] points to detection_and_retrieval/
    detection_and_retrieval_dir = Path(__file__).resolve().parents[2]

    data_dir = detection_and_retrieval_dir.parent / "data"

    source_dataset_dir = data_dir / "8-Ball Pool.v3i.yolov11"
    destination_dataset_dir = data_dir / "8-Ball Pool.v3i.yolov11_no_dot"

    if not source_dataset_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset_dir}")

    if destination_dataset_dir.exists():
        print(f"Removing old dataset without Dot: {destination_dataset_dir}")
        shutil.rmtree(destination_dataset_dir)

    destination_dataset_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        copy_split_without_dot(
            source_split_dir=source_dataset_dir / split,
            destination_split_dir=destination_dataset_dir / split,
        )

    # Copy README files if they exist
    for readme_name in ["README.dataset.txt", "README.roboflow.txt"]:
        source_readme = source_dataset_dir / readme_name
        if source_readme.exists():
            shutil.copy2(source_readme, destination_dataset_dir / readme_name)

    create_data_yaml(destination_dataset_dir)

    print("\nDataset without Dot created successfully.")
    print(f"Original dataset: {source_dataset_dir}")
    print(f"New dataset: {destination_dataset_dir}")


if __name__ == "__main__":
    main()