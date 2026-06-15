from pathlib import Path
import shutil
import yaml
import torch
from ultralytics import YOLO


def load_config(config_path: Path) -> dict:
    """
    Load training configuration from a YAML file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def main():
    # This script is inside ball_detection/src
    # parents[1] points to ball_detection/
    current_dir = Path(__file__).resolve().parents[1]

    # Load config.yaml
    config_path = current_dir / "configs" / "config.yaml"
    config = load_config(config_path)

    # Paths from config
    dataset_yaml = current_dir / config["dataset_yaml"]
    base_model = config["base_model"]
    project_dir = current_dir / config["project_dir"]
    experiment_name = config["experiment_name"]
    trained_model_path = current_dir / config["trained_model"]

    # Training parameters
    epochs = config["epochs"]
    imgsz = config["imgsz"]
    batch = config["batch"]

    # Check paths
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    # Choose device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLO model
    model = YOLO(str(base_model))

    # Train model
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project_dir),
        name=experiment_name,
        device=device,
        workers=0,
        exist_ok=True,
    )

    # Path where YOLO saves the training run
    run_dir = project_dir / experiment_name

    # Path where YOLO saves the best model
    best_model_from_run = run_dir / "weights" / "best.pt"

    if not best_model_from_run.exists():
        raise FileNotFoundError(
            f"Training finished, but best.pt was not found at: {best_model_from_run}"
        )

    # Create weights folder if it does not exist
    trained_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove old final model if it already exists
    if trained_model_path.exists():
        trained_model_path.unlink()

    # Copy best.pt to weights/best.pt
    shutil.copy(best_model_from_run, trained_model_path)

    print("\nTraining completed successfully.")
    print(f"Training run saved at: {run_dir}")
    print(f"Best model from run: {best_model_from_run}")
    print(f"Copied final model to: {trained_model_path}")


if __name__ == "__main__":
    main()