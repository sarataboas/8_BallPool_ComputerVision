from pathlib import Path
import shutil
import yaml


NEW_CLASS_NAMES = ["Black", "Cue", "Solid", "Striped"]

# Dataset novo Roboflow:
# normalmente vem com classes:
# 0  = Black / 8-ball
# 1  = Cue / white ball
# 2-8 = Solid balls
# 9-15 = Striped balls
#
# Novo formato do nosso projeto:
# 0 = Black
# 1 = Cue
# 2 = Solid
# 3 = Striped

ROBOFLOW_TO_OUR_CLASSES = {
    0: 0,   # Black
    1: 1,   # Cue

    2: 2,   # Solid
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,

    9: 3,   # Striped
    10: 3,
    11: 3,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
}


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def list_images(images_dir: Path):
    return [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_original_split(source_split_dir: Path, destination_split_dir: Path) -> None:
    """
    Copia um split inteiro do dataset original sem Dot.
    Isto preserva exatamente as imagens/labels de train, valid e test originais.
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

    image_paths = list_images(source_images_dir)

    print(f"Copying original {source_split_dir.name}: {len(image_paths)} images")

    for image_path in image_paths:
        label_path = source_labels_dir / f"{image_path.stem}.txt"

        shutil.copy2(image_path, destination_images_dir / image_path.name)

        destination_label_path = destination_labels_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            shutil.copy2(label_path, destination_label_path)
        else:
            destination_label_path.write_text("", encoding="utf-8")


def convert_roboflow_label(source_label_path: Path, destination_label_path: Path) -> None:
    """
    Converte labels YOLO do dataset Roboflow de 16 classes para 4 classes.
    """
    new_lines = []

    if source_label_path.exists():
        lines = source_label_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            old_class_id = int(parts[0])

            if old_class_id not in ROBOFLOW_TO_OUR_CLASSES:
                print(f"Warning: unknown class id {old_class_id} in {source_label_path}")
                continue

            new_class_id = ROBOFLOW_TO_OUR_CLASSES[old_class_id]
            parts[0] = str(new_class_id)

            new_lines.append(" ".join(parts))

    destination_label_path.parent.mkdir(parents=True, exist_ok=True)

    text = "\n".join(new_lines)
    if text:
        text += "\n"

    destination_label_path.write_text(text, encoding="utf-8")


def add_roboflow_split_to_train(
    source_split_dir: Path,
    destination_train_dir: Path,
    prefix: str,
) -> None:
    """
    Adiciona imagens do dataset Roboflow APENAS ao train.
    As labels são convertidas para o nosso formato de 4 classes.
    """
    source_images_dir = source_split_dir / "images"
    source_labels_dir = source_split_dir / "labels"

    if not source_images_dir.exists():
        print(f"Skipping missing split: {source_split_dir}")
        return

    if not source_labels_dir.exists():
        print(f"Skipping split without labels: {source_split_dir}")
        return

    destination_images_dir = destination_train_dir / "images"
    destination_labels_dir = destination_train_dir / "labels"

    destination_images_dir.mkdir(parents=True, exist_ok=True)
    destination_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(source_images_dir)

    print(f"Adding Roboflow {source_split_dir.name} to TRAIN: {len(image_paths)} images")

    for image_path in image_paths:
        # Prefixo para evitar nomes repetidos entre datasets
        new_image_name = f"{prefix}_{source_split_dir.name}_{image_path.name}"
        new_label_name = f"{Path(new_image_name).stem}.txt"

        destination_image_path = destination_images_dir / new_image_name
        destination_label_path = destination_labels_dir / new_label_name

        source_label_path = source_labels_dir / f"{image_path.stem}.txt"

        shutil.copy2(image_path, destination_image_path)
        convert_roboflow_label(source_label_path, destination_label_path)


def create_data_yaml(destination_dataset_dir: Path) -> None:
    data = {
        "path": str(destination_dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 4,
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
        labels_dir = dataset_dir / split / "labels"

        n_images = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        n_labels = len(list(labels_dir.glob("*"))) if labels_dir.exists() else 0

        print(f"  {split:<6} images={n_images:<5} labels={n_labels:<5}")


def main() -> None:
    # Este script está em:
    # detection_and_retrieval/ball_detection/src/
    ball_detection_dir = Path(__file__).resolve().parents[1]

    # Sobe até à raiz do projeto:
    # 8_BallPool_ComputerVision/
    project_root = ball_detection_dir.parents[1]

    data_dir = project_root / "data"

    original_dataset_dir = data_dir / "8-Ball Pool.v3i.yolov11_no_dot"

    roboflow_dataset_dir = data_dir / "Pool Balls Detection.v13-v13.yolov11"

    destination_dataset_dir = data_dir / "8-Ball Pool.v3i.yolov11_no_dot_plus_poolballs_train"

    if not original_dataset_dir.exists():
        raise FileNotFoundError(f"Original no-dot dataset not found: {original_dataset_dir}")

    if not roboflow_dataset_dir.exists():
        raise FileNotFoundError(f"Roboflow dataset not found: {roboflow_dataset_dir}")

    if destination_dataset_dir.exists():
        print(f"Removing old combined dataset: {destination_dataset_dir}")
        shutil.rmtree(destination_dataset_dir)

    destination_dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copia o dataset antigo completo.
    # Isto mantém valid e test exatamente iguais ao dataset anterior.
    for split in ["train", "valid", "test"]:
        copy_original_split(
            source_split_dir=original_dataset_dir / split,
            destination_split_dir=destination_dataset_dir / split,
        )

    # 2. Junta o dataset novo SOMENTE ao train.
    destination_train_dir = destination_dataset_dir / "train"

    for split in ["train", "valid", "test"]:
        add_roboflow_split_to_train(
            source_split_dir=roboflow_dataset_dir / split,
            destination_train_dir=destination_train_dir,
            prefix="poolballs_v13",
        )

    # 3. Cria data.yaml
    create_data_yaml(destination_dataset_dir)

    count_files(destination_dataset_dir)

    print("\nCombined dataset created successfully.")
    print(f"Original dataset: {original_dataset_dir}")
    print(f"Roboflow dataset: {roboflow_dataset_dir}")
    print(f"New dataset: {destination_dataset_dir}")
    print("\nImportant: valid and test were copied only from the original dataset.")


if __name__ == "__main__":
    main()