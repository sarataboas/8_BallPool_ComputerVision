from pathlib import Path
import argparse
import json
import yaml

from utils import load_detection_model


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


def evaluate_model(
    model_path: Path,
    dataset_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    project_dir: Path,
    run_name: str,
):
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

    model = load_detection_model(model_path)

    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        plots=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    return metrics


def extract_metrics(metrics, split: str) -> dict:
    """
    Extract detection metrics from YOLO validation output, including
    per-class breakdown and inference speed.
    """
    per_class = {}
    for idx, class_id in enumerate(metrics.box.ap_class_index):
        class_name = metrics.names[int(class_id)]
        per_class[class_name] = {
            "precision": float(metrics.box.p[idx]),
            "recall": float(metrics.box.r[idx]),
            "ap50": float(metrics.box.ap50[idx]),
            "ap50_95": float(metrics.box.ap[idx]),
        }

    results = {
        "split": split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
        "speed_ms_per_image": {
            "preprocess": metrics.speed.get("preprocess", 0.0),
            "inference": metrics.speed.get("inference", 0.0),
            "postprocess": metrics.speed.get("postprocess", 0.0),
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO model for ball detection."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML, e.g. configs/experiment02_no_dot.yaml",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained YOLO model. If not provided, uses trained_model from --config.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on: val or test.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation metrics JSON.",
    )

    args = parser.parse_args()

    # This script is inside ball_detection/src
    # parents[1] points to ball_detection/
    current_dir = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = current_dir / config_path
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

    # If the model comes from experiments/<name>/weights/best.pt, reuse <name>
    # to label the eval run; otherwise fall back to the model file's stem.
    if model_path.parent.name == "weights":
        model_label = model_path.parent.parent.name
    else:
        model_label = model_path.stem

    project_dir = current_dir / "outputs" / "eval_runs"
    run_name = f"{model_label}_{args.split}"

    metrics = evaluate_model(
        model_path=model_path,
        dataset_yaml=dataset_yaml,
        split=args.split,
        imgsz=imgsz,
        batch=batch,
        project_dir=project_dir,
        run_name=run_name,
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
    print(f"Inference speed: {results['speed_ms_per_image']['inference']:.2f} ms/image")

    print("\nPer-class metrics:")
    for class_name, class_metrics in results["per_class"].items():
        print(
            f"  {class_name:<10} "
            f"P={class_metrics['precision']:.4f}  "
            f"R={class_metrics['recall']:.4f}  "
            f"AP50={class_metrics['ap50']:.4f}  "
            f"AP50-95={class_metrics['ap50_95']:.4f}"
        )

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()