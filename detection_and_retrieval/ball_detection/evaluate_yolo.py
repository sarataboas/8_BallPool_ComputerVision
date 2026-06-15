from pathlib import Path
import argparse
import json
import yaml

from ultralytics import YOLO


def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def save_json(data: dict, output_path: Path):
    """
    Save dictionary as JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def evaluate_model(model_path: Path, dataset_yaml: Path, split: str, imgsz: int, batch: int):
    """
    Evaluate YOLO model on a selected dataset split.

    split can be:
    - val
    - test
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        verbose=True,
    )

    return metrics


def extract_metrics(metrics, split: str) -> dict:
    """
    Extract the most important detection metrics from YOLO validation output.
    """
    results = {
        "split": split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO model for ball detection."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained YOLO model. If not provided, uses trained_model from config.yaml."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on: val or test."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation metrics JSON."
    )

    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent

    config_path = current_dir / "config.yaml"
    config = load_config(config_path)

    dataset_yaml = current_dir / config["dataset_yaml"]

    if args.model is None:
        model_path = current_dir / config["trained_model"]
    else:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = current_dir / model_path

    imgsz = config.get("imgsz", 640)
    batch = config.get("batch", 4)

    metrics = evaluate_model(
        model_path=model_path,
        dataset_yaml=dataset_yaml,
        split=args.split,
        imgsz=imgsz,
        batch=batch,
    )

    results = extract_metrics(metrics, args.split)

    if args.output is None:
        output_path = current_dir / "outputs" / f"evaluation_{args.split}.json"
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = current_dir / output_path

    save_json(results, output_path)

    print("\nEvaluation completed successfully.")
    print(f"Split: {args.split}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"mAP@50: {results['map50']:.4f}")
    print(f"mAP@50-95: {results['map50_95']:.4f}")
    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()