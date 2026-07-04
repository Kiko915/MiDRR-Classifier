"""Manuscript figure generation from ``models/metrics.json`` (Phase 9).

Run from the project root::

    python -m midrr_classifier.reporting
    # or with explicit paths:
    python -m midrr_classifier.reporting --metrics models/metrics.json --output-dir models/figures

Why this module reads only ``metrics.json``
--------------------------------------------
Phases 5/6 were built so that *every* number a Chapter 4 figure needs —
CV mean/std, the confusion matrix, per-class precision/recall/F1, and both
importance rankings — already lives in ``metrics.json``
(:func:`~midrr_classifier.utils.metrics.update_metrics_json`). This module
never reloads the model or re-runs a prediction; it only reads that one file.
That is what makes the figures deterministic and instantly reproducible: run
``train.py``/``evaluate.py`` once, then re-generate every figure as many
times as the manuscript needs without touching the model again.

Four artifacts, matching the Phase 9 checklist:

* :func:`plot_confusion_matrix_figure` — heatmap from ``metrics["test"]``.
* :func:`plot_feature_importance_figure` — permutation vs. Gini side-by-side,
  from ``metrics["importances"]``.
* :func:`render_per_class_table` — precision/recall/F1/support table
  (+ macro/weighted rows), written as CSV and Markdown.
* :func:`plot_cv_variance_figure` — mean ± std bars (with per-fold scatter)
  from ``metrics["cv"]``.

:func:`generate_all_figures` runs whichever of the four are possible given
what keys are actually present in ``metrics.json`` (e.g. ``train.py`` alone
only populates ``"cv"``; the confusion matrix and importances need
``evaluate.py`` to have run too).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import numpy as np

from midrr_classifier.config import MiDRRConfig, load_config
from midrr_classifier.utils.logging_utils import get_logger, setup_root_logging

logger = get_logger(__name__)


def load_metrics(metrics_path: str) -> dict[str, Any]:
    """Load ``metrics.json``, failing with an actionable message if absent.

    Args:
        metrics_path: Path to the metrics JSON written by ``train.py`` /
            ``evaluate.py`` (typically ``config.metrics_path``).

    Returns:
        The parsed metrics dict.

    Raises:
        FileNotFoundError: If no file exists at ``metrics_path`` yet.
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"No metrics.json found at '{metrics_path}'. Run "
            "`python -m midrr_classifier.train` and "
            "`python -m midrr_classifier.evaluate` first to produce it."
        )
    with open(metrics_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_confusion_matrix_figure(metrics: dict[str, Any], save_path: str) -> str:
    """Render the confusion-matrix heatmap from ``metrics["test"]`` alone.

    Args:
        metrics: The full metrics dict (see :func:`load_metrics`).
        save_path: PNG destination.

    Returns:
        ``save_path``, for chaining into a results dict.

    Raises:
        KeyError: If evaluation hasn't run yet (no ``"test"`` key).
    """
    if "test" not in metrics:
        raise KeyError(
            "metrics.json has no 'test' key — run `python -m "
            "midrr_classifier.evaluate` before generating the confusion matrix."
        )

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.switch_backend("Agg")  # figures are always saved to disk, never shown

    test = metrics["test"]
    cm = test["confusion_matrix"]
    labels = cm["labels"]
    matrix = np.array(cm["matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Confusion Matrix (n={test['n_samples']}, accuracy={test['accuracy']:.1%})"
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_feature_importance_figure(metrics: dict[str, Any], save_path: str) -> str:
    """Render permutation vs. Gini importance side-by-side.

    Both panels share the same feature order (sorted by permutation
    importance) so a reader can visually cross-check whether the two
    methods agree — the whole point of running both (Gini is biased toward
    high-cardinality/continuous features; permutation isn't).

    Args:
        metrics: The full metrics dict.
        save_path: PNG destination.

    Returns:
        ``save_path``.

    Raises:
        KeyError: If no ``"importances"`` key is present.
    """
    if "importances" not in metrics:
        raise KeyError(
            "metrics.json has no 'importances' key — run `python -m "
            "midrr_classifier.evaluate` before generating this figure."
        )

    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")  # figures are always saved to disk, never shown

    imp = metrics["importances"]
    order = imp["ranking_permutation"]
    y_pos = np.arange(len(order))

    perm_means = [imp["permutation"][f]["mean"] for f in order]
    perm_stds = [imp["permutation"][f]["std"] for f in order]
    gini_vals = [imp["gini"][f] for f in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(order) + 2))

    axes[0].barh(y_pos, perm_means, xerr=perm_stds, color="#4C72B0", capsize=3)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean accuracy drop (± std)")
    axes[0].set_title("Permutation Importance")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(y_pos, gini_vals, color="#55A868")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(order)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Gini importance")
    axes[1].set_title("Gini (Mean Decrease in Impurity)")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("Feature Importance — Permutation vs. Gini Cross-Check")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_cv_variance_figure(metrics: dict[str, Any], save_path: str) -> str:
    """Render mean ± std bars for every grouped-CV metric, with per-fold scatter.

    The scatter overlay shows the raw fold scores behind each bar — the
    "how much would you trust this number" picture the std alone doesn't
    convey (five bunched dots vs. five spread-out ones look very different
    even at the same std).

    Args:
        metrics: The full metrics dict.
        save_path: PNG destination.

    Returns:
        ``save_path``.

    Raises:
        KeyError: If no ``"cv"`` key is present.
    """
    if "cv" not in metrics:
        raise KeyError(
            "metrics.json has no 'cv' key — run `python -m midrr_classifier.train` "
            "(with cross-validation enabled) before generating this figure."
        )

    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")  # figures are always saved to disk, never shown

    cv = metrics["cv"]
    metric_names = list(cv["metrics"].keys())
    means = [cv["metrics"][m]["mean"] for m in metric_names]
    stds = [cv["metrics"][m]["std"] for m in metric_names]
    x_pos = np.arange(len(metric_names))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x_pos, means, yerr=stds, capsize=5, color="#4C72B0", alpha=0.85, zorder=2)

    rng = np.random.default_rng(0)  # fixed seed: jitter is cosmetic, not data
    for i, m in enumerate(metric_names):
        folds = cv["metrics"][m]["per_fold"]
        jitter = rng.uniform(-0.08, 0.08, size=len(folds))
        ax.scatter(
            np.full(len(folds), x_pos[i]) + jitter,
            folds,
            color="black",
            s=18,
            zorder=3,
            alpha=0.7,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, rotation=15)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Grouped {cv['n_splits']}-Fold CV — Mean ± Std (dots = individual folds)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def _rows_to_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render a list of dict rows as a GitHub-flavoured Markdown table."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_per_class_table(
    metrics: dict[str, Any],
    save_path_csv: Optional[str] = None,
    save_path_md: Optional[str] = None,
):
    """Build the per-class precision/recall/F1/support table (+ macro/weighted rows).

    Args:
        metrics: The full metrics dict.
        save_path_csv: If given, write the table there as CSV.
        save_path_md: If given, write the table there as a Markdown table
            (paste-ready for the Chapter 4 draft).

    Returns:
        A :class:`pandas.DataFrame` with columns
        ``class, precision, recall, f1, support`` — one row per label plus
        ``macro avg`` and ``weighted avg``.

    Raises:
        KeyError: If no ``"test"`` key is present.
    """
    if "test" not in metrics:
        raise KeyError(
            "metrics.json has no 'test' key — run `python -m "
            "midrr_classifier.evaluate` before generating this table."
        )

    import pandas as pd

    test = metrics["test"]
    # Row order must come from the "labels" *list* (["HIGH","MODERATE","LOW"]),
    # never from iterating the per_class dict directly: update_metrics_json
    # persists metrics.json with `sort_keys=True` for deterministic diffs, which
    # silently alphabetizes per_class's keys on disk (-> HIGH, LOW, MODERATE) even
    # though the "labels" list is untouched by sort_keys (it only reorders dict
    # keys, not list contents). Same bug class as the printed-report/predict_proba
    # label swaps fixed earlier — this is what it looks like after a JSON round-trip.
    labels = test.get("labels", list(test["per_class"].keys()))
    rows = [{"class": label, **test["per_class"][label]} for label in labels]
    for avg_name in ("macro", "weighted"):
        avg = test[avg_name]
        rows.append(
            {
                "class": f"{avg_name} avg",
                "precision": avg["precision"],
                "recall": avg["recall"],
                "f1": avg["f1"],
                "support": test["n_samples"],
            }
        )

    df = pd.DataFrame(rows, columns=["class", "precision", "recall", "f1", "support"])

    if save_path_csv:
        os.makedirs(os.path.dirname(save_path_csv) or ".", exist_ok=True)
        df.to_csv(save_path_csv, index=False)

    if save_path_md:
        os.makedirs(os.path.dirname(save_path_md) or ".", exist_ok=True)
        with open(save_path_md, "w", encoding="utf-8") as fh:
            fh.write(_rows_to_markdown(rows, list(df.columns)))

    return df


def generate_all_figures(
    config_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict[str, str]:
    """Generate every figure/table that the current ``metrics.json`` supports.

    Runs whichever of the four Phase 9 artifacts its required key(s) are
    present for — e.g. right after ``train.py`` (no ``evaluate.py`` yet),
    only the CV-variance plot is possible, and that's not an error.

    Args:
        config_path: Optional YAML config (for default paths).
        metrics_path: Override ``config.metrics_path``.
        output_dir: Override ``config.figures_dir``.

    Returns:
        Dict mapping artifact name to the file path written. Possible keys:
        ``confusion_matrix``, ``feature_importance``, ``cv_variance``,
        ``per_class_table_csv``, ``per_class_table_md``.
    """
    cfg = load_config(config_path)
    metrics_path = metrics_path or cfg.metrics_path
    output_dir = output_dir or cfg.figures_dir

    metrics = load_metrics(metrics_path)
    os.makedirs(output_dir, exist_ok=True)

    saved: dict[str, str] = {}

    if "test" in metrics:
        saved["confusion_matrix"] = plot_confusion_matrix_figure(
            metrics, os.path.join(output_dir, "confusion_matrix.png")
        )
        render_per_class_table(
            metrics,
            save_path_csv=os.path.join(output_dir, "per_class_metrics.csv"),
            save_path_md=os.path.join(output_dir, "per_class_metrics.md"),
        )
        saved["per_class_table_csv"] = os.path.join(output_dir, "per_class_metrics.csv")
        saved["per_class_table_md"] = os.path.join(output_dir, "per_class_metrics.md")
    else:
        logger.warning(
            "No 'test' key in metrics.json — skipping confusion matrix + "
            "per-class table (run evaluate.py)."
        )

    if "importances" in metrics:
        saved["feature_importance"] = plot_feature_importance_figure(
            metrics, os.path.join(output_dir, "feature_importance.png")
        )
    else:
        logger.warning(
            "No 'importances' key in metrics.json — skipping feature-importance "
            "figure (run evaluate.py)."
        )

    if "cv" in metrics:
        saved["cv_variance"] = plot_cv_variance_figure(
            metrics, os.path.join(output_dir, "cv_variance.png")
        )
    else:
        logger.warning(
            "No 'cv' key in metrics.json — skipping CV-variance figure (run train.py)."
        )

    logger.info("Generated %d artifact(s) into %s", len(saved), output_dir)
    return saved


if __name__ == "__main__":
    setup_root_logging()

    parser = argparse.ArgumentParser(
        description="Generate manuscript figures/tables from metrics.json (Phase 9)."
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--metrics",
        default=None,
        dest="metrics_path",
        help="Path to metrics.json (default: from config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write figures/tables into (default: models/figures).",
    )
    args = parser.parse_args()

    saved_paths = generate_all_figures(
        config_path=args.config,
        metrics_path=args.metrics_path,
        output_dir=args.output_dir,
    )
    print("\n=== Phase 9 figures generated ===")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")
