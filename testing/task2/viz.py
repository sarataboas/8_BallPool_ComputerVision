"""
Visualisation helpers for Task 2 experiment results.
All functions are display-only — no training logic here.
"""

import numpy as np
import matplotlib.pyplot as plt

from testing.utils import show_many, imread_rgb


def plot_history(history: dict) -> None:
    """Plots train vs val loss curves for a single experiment."""
    pass


def plot_error_distribution(results: dict) -> None:
    """Histogram of (pred - gt) errors across the evaluation split."""
    pass


def show_worst_predictions(results: dict, k: int = 8) -> None:
    """Displays the k images with the largest |pred - gt| error."""
    pass
