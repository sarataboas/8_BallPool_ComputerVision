"""
Task 2 — Evaluation script.

Compares model predictions (output JSON) against ground-truth CSV and writes
a metrics CSV with one row per image plus a summary row at the end.

Usage:
    python models/eval.py \
        --predictions models/output.json \
        --ground-truth data/ground_truth.csv \
        --output models/json_metrics.csv
"""

import argparse
import csv
import json
from pathlib import Path


def evaluate(predictions_json: str, ground_truth_csv: str, output_csv: str) -> None:
    # ── Load predictions ───────────────────────────────────────────────────────
    with open(predictions_json) as f:
        pred_data = json.load(f)
    preds = {Path(r["image_path"]).name: int(r["num_balls"]) for r in pred_data}

    # ── Load ground truth ──────────────────────────────────────────────────────
    gt = {}
    with open(ground_truth_csv) as f:
        for row in csv.DictReader(f):
            gt[row["image_id"]] = int(row["num_balls"])

    # ── Match and compute per-image metrics ───────────────────────────────────
    rows = []
    unmatched = []
    for name, pred in sorted(preds.items()):
        if name not in gt:
            unmatched.append(name)
            continue
        true     = gt[name]
        error    = pred - true
        abs_err  = abs(error)
        rows.append({
            "image_id":   name,
            "true_count": true,
            "pred_count": pred,
            "error":      error,
            "abs_error":  abs_err,
            "exact":      int(abs_err == 0),
            "off1":       int(abs_err <= 1),
        })

    if unmatched:
        print(f"Warning: {len(unmatched)} predictions had no ground-truth match: {unmatched}")

    n         = len(rows)
    mae       = sum(r["abs_error"] for r in rows) / n
    exact     = sum(r["exact"]     for r in rows) / n
    off1      = sum(r["off1"]      for r in rows) / n
    mean_err  = sum(r["error"]     for r in rows) / n

    # ── Write CSV ──────────────────────────────────────────────────────────────
    fields = ["image_id", "true_count", "pred_count", "error", "abs_error", "exact", "off1"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        # Summary row
        writer.writerow({
            "image_id":   f"SUMMARY ({n} images)",
            "true_count": "",
            "pred_count": "",
            "error":      f"{mean_err:+.4f}",
            "abs_error":  f"{mae:.4f}",
            "exact":      f"{exact:.2%}",
            "off1":       f"{off1:.2%}",
        })

    print(f"Images evaluated : {n}")
    print(f"MAE              : {mae:.4f}")
    print(f"Exact accuracy   : {exact:.2%}")
    print(f"Off-by-1         : {off1:.2%}")
    print(f"Mean signed error: {mean_err:+.4f}")
    print(f"Metrics written  : {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 — Evaluation")
    parser.add_argument("--predictions",  default="models/output.json",
                        help="Path to model output JSON")
    parser.add_argument("--ground-truth", default="data/ground_truth.csv",
                        help="Path to ground-truth CSV")
    parser.add_argument("--output",       default="models/json_metrics.csv",
                        help="Path for output metrics CSV")
    args = parser.parse_args()

    evaluate(args.predictions, args.ground_truth, args.output)
