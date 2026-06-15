from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
from ultralytics import YOLO


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


class BallDetector:
    """
    Ball detector using a trained YOLO model.

    This class receives an image and returns:
    - total number of detected balls
    - bounding boxes
    - class names
    - confidence scores
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        # This script is inside ball_detection/src
        # parents[1] points to ball_detection/
        current_dir = Path(__file__).resolve().parents[1]

        # Load config
        if config_path is None:
            config_path = current_dir / "configs" / "config.yaml"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = current_dir / config_path

        self.config = load_config(config_path)

        # Use model path from argument or from config.yaml
        if model_path is None:
            model_path = current_dir / self.config["trained_model"]
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = current_dir / model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}\n"
                "You need to train the model first and save it as weights/best.pt."
            )

        self.model_path = model_path
        self.model = YOLO(str(model_path))

        self.confidence_threshold = self.config.get("confidence_threshold", 0.25)
        self.iou_threshold = self.config.get("iou_threshold", 0.5)

    def detect_image(self, image_path: str) -> Dict[str, Any]:
        """
        Detect balls in a single image.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        dict
            Dictionary containing image name, number of balls and detections.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        result = results[0]
        detections = self._parse_result(result)

        return {
            "image": image_path.name,
            "num_balls": len(detections),
            "detections": detections,
        }

    def detect_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Detect balls in all images from a folder.

        Parameters
        ----------
        folder_path : str
            Path to folder containing images.

        Returns
        -------
        list
            List with detection results for each image.
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = [
            path for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ]

        all_results = []

        for image_path in image_paths:
            image_result = self.detect_image(str(image_path))
            all_results.append(image_result)

        return all_results

    def save_prediction_image(self, image_path: str, output_dir: str) -> Path:
        """
        Run detection on one image and save the image with bounding boxes.

        Parameters
        ----------
        image_path : str
            Path to input image.

        output_dir : str
            Directory where the prediction image will be saved.

        Returns
        -------
        Path
            Path to saved prediction image.
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            save=False,
            verbose=False,
        )

        result = results[0]
        plotted_image = result.plot()

        output_path = output_dir / f"{image_path.stem}_pred.jpg"

        import cv2
        cv2.imwrite(str(output_path), plotted_image)

        return output_path

    def _parse_result(self, result) -> List[Dict[str, Any]]:
        """
        Convert YOLO result into a clean list of detections.
        """
        detections = []

        if result.boxes is None:
            return detections

        class_names = result.names

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": round(confidence, 4),
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                ],
            }

            detections.append(detection)

        return detections


if __name__ == "__main__":
    detector = BallDetector()

    print(
        "BallDetector is ready.\n"
        "Use detector.detect_image(image_path) for one image or "
        "detector.detect_folder(folder_path) for multiple images."
    )