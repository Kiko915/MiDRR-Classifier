"""Metric helpers wrapping scikit-learn for the MiDRR-Classifier project."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Full evaluation report as a JSON-serialisable dict (Phase 6).

    Unlike :func:`compute_classification_metrics` (macro headline numbers
    only), this returns the complete picture the manuscript's Chapter 1/3
    objectives call for: overall accuracy, **both** macro- and
    weighted-averaged precision/recall/F1, a **per-class** breakdown with
    support, and the confusion matrix.

    Macro vs weighted, in plain terms:
        * *macro* averages the three per-class scores with equal weight — it
          treats LOW (the rare class) as just as important as MODERATE.
        * *weighted* averages them by how many samples each class has — it
          reflects overall hit-rate but can hide a class the model ignores.
    Reporting both is what makes a small-N, imbalanced result honest.

    Args:
        y_true: Ground-truth labels.
        y_pred: Model-predicted labels.
        labels: Ordered class names (``LABEL_CLASSES``) — fixes both the
            per-class order and the confusion-matrix axis order.

    Returns:
        ``{"accuracy", "macro", "weighted", "per_class", "confusion_matrix",
        "n_samples", "labels"}`` — every value a plain float/int/list so it
        drops straight into ``metrics.json``.
    """
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {
            "precision": float(report["macro avg"]["precision"]),
            "recall": float(report["macro avg"]["recall"]),
            "f1": float(report["macro avg"]["f1-score"]),
        },
        "weighted": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": list(labels), "matrix": cm.tolist()},
        "n_samples": int(len(y_true)),
        "labels": list(labels),
    }


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


def update_metrics_json(path: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge *updates* into the metrics JSON at *path* and write it back.

    Uses merge (not overwrite) so different phases can own different top-level
    keys — e.g. training writes ``"cv"``/``"tuning"``, and Phase 6 evaluation
    later adds ``"test"`` to the same file without clobbering them.

    Args:
        path: Destination JSON path (typically ``config.metrics_path``).
        updates: Top-level keys to set/replace.

    Returns:
        The full merged dict that was written.
    """
    existing: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            try:
                existing = json.load(fh)
            except json.JSONDecodeError:
                existing = {}

    existing.update(updates)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)
    return existing


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list[str]] = None,
) -> None:
    """Print a full per-class classification report to stdout.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of class names.  Passed as BOTH ``labels`` and
            ``target_names`` — omitting ``labels`` lets sklearn fall back to
            alphabetical class order while still applying *labels* as display
            names, which silently mislabels rows when the class order is not
            alphabetical (``LABEL_CLASSES`` is HIGH/MODERATE/LOW, not sorted).
    """
    print(
        classification_report(
            y_true, y_pred, labels=labels, target_names=labels, zero_division=0
        )
    )


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
