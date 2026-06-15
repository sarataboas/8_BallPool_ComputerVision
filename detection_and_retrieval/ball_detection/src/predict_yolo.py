from pathlib import Path
import argparse
import json

from ball_detection import BallDetector


def save_json(results, output_path: Path):
    """
    Save detection results to a JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


def is_image_file(path: Path) -> bool:
    """
    Check if a file is an image.
    """
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    return path.suffix.lower() in valid_extensions


def predict_single_image(detector: BallDetector, image_path: Path, output_dir: Path):
    """
    Run prediction on a single image.
    """
    print(f"Processing image: {image_path.name}")

    # Detection results in dictionary format
    result = detector.detect_image(str(image_path))

    # Save image with predicted bounding boxes
    prediction_image_path = detector.save_prediction_image(
        image_path=str(image_path),
        output_dir=str(output_dir),
    )

    result["prediction_image"] = str(prediction_image_path)

    return result


def predict_folder(detector: BallDetector, folder_path: Path, output_dir: Path):
    """
    Run prediction on all images inside a folder.
    """
    image_paths = [
        path for path in folder_path.iterdir()
        if path.is_file() and is_image_file(path)
    ]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in folder: {folder_path}")

    all_results = []

    for image_path in image_paths:
        result = predict_single_image(detector, image_path, output_dir)
        all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO ball detection on one image or a folder of images."
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image or to a folder containing images.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained YOLO model. If not provided, uses weights/best.pt from config.yaml.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Directory where prediction images will be saved.",
    )

    parser.add_argument(
        "--json-output",
        type=str,
        default="outputs/predictions/predictions.json",
        help="Path where JSON results will be saved.",
    )

    args = parser.parse_args()

    # This script is inside ball_detection/src
    # parents[1] points to ball_detection/
    current_dir = Path(__file__).resolve().parents[1]

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = current_dir / source_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = current_dir / output_dir

    json_output = Path(args.json_output)
    if not json_output.is_absolute():
        json_output = current_dir / json_output

    if args.model is not None:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = current_dir / model_path

        detector = BallDetector(model_path=str(model_path))
    else:
        detector = BallDetector()

    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    if source_path.is_file():
        if not is_image_file(source_path):
            raise ValueError(f"Source file is not a valid image: {source_path}")

        results = [
            predict_single_image(detector, source_path, output_dir)
        ]

    elif source_path.is_dir():
        results = predict_folder(detector, source_path, output_dir)

    else:
        raise ValueError(f"Source must be an image or a folder: {source_path}")

    save_json(results, json_output)

    print("\nPrediction completed successfully.")
    print(f"Images saved to: {output_dir}")
    print(f"JSON results saved to: {json_output}")


if __name__ == "__main__":
    main()