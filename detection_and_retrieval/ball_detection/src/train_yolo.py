from pathlib import Path
import shutil

import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
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


def save_training_plots(run_dir: Path, output_dir: Path) -> None:
    """
    Save extra training plots from YOLO results.csv.

    This creates:
    - losses.png
    - metrics.png
    - results.csv copy
    """
    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        print(f"Warning: results.csv not found at: {results_csv}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    # YOLO sometimes saves column names with extra spaces
    df.columns = df.columns.str.strip()

    # Save a clean copy of the raw CSV
    shutil.copy(results_csv, output_dir / "results.csv")

    # YOLO usually has an epoch column.
    # If not, use the row index.
    if "epoch" in df.columns:
        x_values = df["epoch"]
        x_label = "Epoch"
    else:
        x_values = df.index
        x_label = "Epoch"

    # -------------------------
    # Loss plot
    # -------------------------
    loss_columns = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]

    available_loss_columns = [col for col in loss_columns if col in df.columns]

    if available_loss_columns:
        plt.figure(figsize=(10, 6))

        for col in available_loss_columns:
            plt.plot(x_values, df[col], label=col)

        plt.xlabel(x_label)
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "losses.png", dpi=300)
        plt.close()
    else:
        print("Warning: no YOLO loss columns found in results.csv")

    # -------------------------
    # Metrics plot
    # -------------------------
    metric_columns = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    available_metric_columns = [col for col in metric_columns if col in df.columns]

    if available_metric_columns:
        plt.figure(figsize=(10, 6))

        for col in available_metric_columns:
            plt.plot(x_values, df[col], label=col)

        plt.xlabel(x_label)
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "metrics.png", dpi=300)
        plt.close()
    else:
        print("Warning: no YOLO metric columns found in results.csv")

    print(f"Extra training plots saved to: {output_dir}")


def copy_full_experiment(run_dir: Path, experiments_dir: Path, experiment_name: str) -> Path:
    """
    Copy the full YOLO run folder into experiments/experiment_name.

    This saves everything from the run:
    - weights
    - results.csv
    - results.png
    - confusion matrices
    - train/val example images
    - args.yaml
    - etc.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"YOLO run folder not found: {run_dir}")

    experiments_dir.mkdir(parents=True, exist_ok=True)
    experiment_backup_dir = experiments_dir / experiment_name

    # If the experiment backup already exists, remove it to avoid mixing old and new files
    if experiment_backup_dir.exists():
        shutil.rmtree(experiment_backup_dir)

    shutil.copytree(run_dir, experiment_backup_dir)

    print(f"Full experiment copied to: {experiment_backup_dir}")

    return experiment_backup_dir


def main():
    # This script is inside ball_detection/src
    # parents[1] points to ball_detection/
    current_dir = Path(__file__).resolve().parents[1]

    # Load config.yaml
    config_path = current_dir / "configs" / "config.yaml"
    config = load_config(config_path)

    # Paths from config
    dataset_yaml = current_dir / config["dataset_yaml"]
    base_model = current_dir / config["base_model"]
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

    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")

    # Choose device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nTraining configuration:")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Base model: {base_model}")
    print(f"Project dir: {project_dir}")
    print(f"Experiment name: {experiment_name}")
    print(f"Final trained model path: {trained_model_path}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")

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

    # Create trained models folder if it does not exist
    trained_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove old final model if it already exists
    if trained_model_path.exists():
        trained_model_path.unlink()

    # Copy best.pt to final trained model path
    shutil.copy(best_model_from_run, trained_model_path)

    # Save extra clean plots to outputs/training/<experiment_name>
    training_output_dir = current_dir / "outputs" / "training" / experiment_name
    save_training_plots(run_dir, training_output_dir)

    # Copy full YOLO run to experiments/<experiment_name>
    experiments_dir = current_dir / "experiments"
    experiment_backup_dir = copy_full_experiment(
        run_dir=run_dir,
        experiments_dir=experiments_dir,
        experiment_name=experiment_name,
    )

    print("\nTraining completed successfully.")
    print(f"YOLO run saved at: {run_dir}")
    print(f"Full experiment backup saved at: {experiment_backup_dir}")
    print(f"Best model from run: {best_model_from_run}")
    print(f"Copied final trained model to: {trained_model_path}")
    print(f"Extra plots saved at: {training_output_dir}")


if __name__ == "__main__":
    main()