"""
Runs a single Task 2 experiment end-to-end:
    build model → train → evaluate → save .pth → append row to results.csv
"""

import csv
import sys
from pathlib import Path
from typing import Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ..config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    MODELS_DIR, EXPERIMENTS_DIR,
    DEFAULT_CFG,
)
from .architectures import get_model
from models.cnn_pipeline import build_loaders, build_loader, train_model, evaluate


RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
_CSV_FIELDS = [
    "name", "backbone", "head", "input_size", "loss", "optimizer",
    "lr", "lr_patience", "epochs", "batch_size", "augment", "crop_table",
    "val_mae", "val_acc", "val_off1",
    "test_mae", "test_acc", "test_off1",
]


def run_experiment(cfg: dict) -> Tuple[dict, dict]:
    """
    Runs one experiment with the given config merged over DEFAULT_CFG.

    Returns:
        results: {"valid": metrics_dict, "test": metrics_dict, "cfg": cfg}
        history: {"train_loss": [...], "val_mae": [...]}
    """
    cfg = {**DEFAULT_CFG, **cfg}
    torch.manual_seed(cfg.get("seed", 42))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"  Run : {cfg['name']}")
    print(f"  {cfg['backbone']} | head={cfg['head']} | size={cfg['input_size']} "
          f"| aug={cfg['augment']} | device={device}")
    print(f"{'='*60}")

    model = get_model(cfg["backbone"], cfg["head"])

    train_loader, valid_loader = build_loaders(TRAIN_DIR, VALID_DIR, cfg)
    test_loader                = build_loader(TEST_DIR,  cfg, train=False)

    history = train_model(model, train_loader, valid_loader, cfg)

    val_metrics  = evaluate(model, valid_loader, device)
    test_metrics = evaluate(model, test_loader,  device)

    print(f"\n  [Valid]  MAE={val_metrics['mae']:.3f}  "
          f"Acc={val_metrics['acc']:.2%}  Off1={val_metrics['off1']:.2%}")
    print(f"  [Test]   MAE={test_metrics['mae']:.3f}  "
          f"Acc={test_metrics['acc']:.2%}  Off1={test_metrics['off1']:.2%}")

    results = {"valid": val_metrics, "test": test_metrics, "cfg": cfg}
    _append_results_csv({**cfg,
        "val_mae":  round(val_metrics["mae"],  4),
        "val_acc":  round(val_metrics["acc"],  4),
        "val_off1": round(val_metrics["off1"], 4),
        "test_mae": round(test_metrics["mae"],  4),
        "test_acc": round(test_metrics["acc"],  4),
        "test_off1":round(test_metrics["off1"], 4),
    })

    return results, history


def _append_results_csv(row: dict) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
