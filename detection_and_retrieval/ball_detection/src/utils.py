from pathlib import Path

from ultralytics import RTDETR, YOLO


def load_detection_model(model_path: Path):
    """
    Load a YOLO or RT-DETR model from a checkpoint path. Both share the same
    Ultralytics train()/val()/predict() API, so callers can use either
    interchangeably - the architecture is picked from the filename
    ("rtdetr" vs everything else).
    """
    if "rtdetr" in Path(model_path).stem.lower():
        return RTDETR(str(model_path))
    return YOLO(str(model_path))
