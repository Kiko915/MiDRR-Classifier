"""Metric helpers wrapping scikit-learn for the MiDRR-Classifier project."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 (macro-averaged).

    Args:
        y_true: Ground-truth preparedness labels.
        y_pred: Model-predicted labels.

    Returns:
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``.  All values are floats in [0, 1].
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list[str]] = None,
) -> None:
    """Print a full per-class classification report to stdout.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of class names for display.
    """
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot and optionally save a seaborn confusion-matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered class names for axis ticks.
        save_path: If provided, the PNG is saved to this path instead of
            being displayed interactively.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("MiDRR-Classifier — Confusion Matrix")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)
