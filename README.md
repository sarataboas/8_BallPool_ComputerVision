# 8_BallPool_ComputerVision

## Data

**Source**: [Roboflow — bachelorthesis/8-ball-pool-l530o](https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o) (v3, 247 images)
**Format**: YOLOv11 — one `.txt` label per image, one annotation per line (`class_id cx cy w h`)
**Classes**: Black(0), Cue(1), Dot(2), Solid(3), Striped(4)
**Ball count**: lines where `class_id != 2` — class 2 ("Dot") are table rail markers, not balls

**Split**: run `data/split_dataset.py` once to create the stratified 80/10/10 split.
- Train: 199 images
- Valid: 24 images
- Test:  24 images

Stratification is by ball count. Groups with fewer than 5 images stay entirely in train. Seed=42.

**Extra datasets** (optional, for training augmentation — document in report if used):
- https://universe.roboflow.com/nidacorian-protonmail-com/pool-billiard
- https://universe.roboflow.com/mark-dj0yk/pool-balls-detection-srlqi
- https://universe.roboflow.com/pool-ball-detection/pool-ball-detection-6lfd9

---

## Task 1

**Goal**: detect balls, identify their numbers, and produce a top-view of the table.
**Deliverable**: `image_processing_pipeline.py` (single file, OpenCV only)
**Dev notebooks**: `testing/task1/task1.ipynb`, `testing/task1/evaluation.ipynb`
**Data**: `development_set/` (50 images), `data/ground_truth.csv`

Input/output JSON format defined in `example_json/`.

---

## Task 2

**Goal**: predict the total number of balls on the table from a single image (CNN-based).
**Deliverable**: `models/cnn_pipeline.py` + `models/best.pth`

Input JSON:
```json
{"image_path": ["path/to/img1.jpg", "..."]}
```

Output JSON:
```json
[{"image_path": "path/to/img1.jpg", "num_balls": 10}, ...]
```

**Experiments**: `testing/task2.ipynb` — one section per run, all results persisted to `testing/experiments/results.csv`
**Metrics**: MAE (headline), exact accuracy, off-by-1 accuracy — on valid and test splits

| File | Role |
|---|---|
| `models/cnn_pipeline.py` | Standalone deliverable — no imports from `testing/` |
| `testing/task2/architectures.py` | `get_model(backbone, head)` factory for comparison runs |
| `testing/task2/experiment.py` | `run_experiment(cfg)` — build → train → eval → save → log |
| `testing/task2/viz.py` | `plot_history`, `plot_error_distribution`, `show_worst_predictions` |
| `testing/experiments/results.csv` | Persistent experiment log |
| `testing/config.py` | Shared paths + `DEFAULT_CFG` |

---

## Task 3

**Goal**: ball detection (object detection model) + table retrieval (find most similar table from training data).
**Deliverable**: notebook `detection_and_retrieval/results.ipynb`

| File | Role |
|---|---|
| `detection_and_retrieval/ball_detection.py` | Detection model (YOLO / DETR) |
| `detection_and_retrieval/table_retrieval.py` | Retrieval pipeline |
