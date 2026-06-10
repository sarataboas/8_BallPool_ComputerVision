"""
Visualisation helpers for Task 2 experiment results.
Display only — no training logic here.
"""

import numpy as np
import matplotlib.pyplot as plt

from ..utils import imread_rgb, show_many


def plot_history(history: dict, cfg: dict = None) -> None:
    """Train loss and validation MAE curves for a single experiment."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], color="steelblue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, history["val_mae"], color="darkorange")
    axes[1].set_title("Validation MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")

    if cfg:
        fig.suptitle(cfg["name"], fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_scatter(results: dict, split: str = "valid") -> None:
    """Predicted vs true count scatter with y=x diagonal. Colour = |error|."""
    data = results[split]
    gt, pred = data["gt"], data["pred"]
    errors = np.abs(pred - gt)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(gt, pred, c=errors, cmap="Reds", s=60, edgecolors="k", linewidths=0.4)
    plt.colorbar(sc, label="|error|")
    plt.plot([0, 16], [0, 16], "b--", linewidth=1, label="y = x")
    plt.xlim(-0.5, 16.5)
    plt.ylim(-0.5, 16.5)
    plt.xlabel("True count")
    plt.ylabel("Predicted count")
    plt.title(f"Predicted vs True — {split}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_by_count(results: dict, split: str = "valid") -> None:
    """MAE per true ball count (bar) with sample count overlay (line)."""
    data = results[split]
    gt, pred = data["gt"], data["pred"]
    counts = sorted(set(gt.tolist()))

    mae_per_count = [
        float(np.mean(np.abs(pred[gt == c] - c))) if (gt == c).any() else 0.0
        for c in counts
    ]
    n_per_count = [(gt == c).sum() for c in counts]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(counts, mae_per_count, color="steelblue", alpha=0.8)
    ax1.set_xlabel("True ball count")
    ax1.set_ylabel("MAE", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(counts, n_per_count, "o--", color="darkorange", label="# images")
    ax2.set_ylabel("# images", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    plt.title(f"MAE by true count — {split}")
    plt.tight_layout()
    plt.show()


def plot_error_distribution(results: dict, split: str = "valid") -> None:
    """Histogram of prediction errors (pred − true)."""
    errors = results[split]["pred"] - results[split]["gt"]
    lo, hi = int(errors.min()) - 1, int(errors.max()) + 1

    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=range(lo, hi + 2), edgecolor="black",
             color="steelblue", align="left")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="no error")
    plt.xlabel("Prediction error (pred − true)")
    plt.ylabel("# images")
    plt.title(f"Error distribution — {split}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_worst_predictions(results: dict, split: str = "valid", k: int = 8) -> None:
    """Gallery of the k images with the largest absolute prediction error."""
    data   = results[split]
    gt, pred, paths = data["gt"], data["pred"], data["paths"]
    errors = np.abs(pred - gt)
    idx    = np.argsort(errors)[::-1][:k]

    imgs   = [imread_rgb(paths[i]) for i in idx]
    titles = [f"pred={pred[i]}  true={gt[i]}  |err|={errors[i]}" for i in idx]
    show_many(imgs, titles=titles, cols=4, figsize=(16, 4 * ((k + 3) // 4)))
