"""Model evaluation entry-point for the MiDRR-Classifier.

Run from the project root::

    python -m midrr_classifier.evaluate
    # or with explicit paths:
    python -m midrr_classifier.evaluate \
        --model models/midrr_rf.pkl \
        --test-csv data/processed/test.csv

Produces:
- A per-class classification report printed to stdout.
- A confusion-matrix PNG saved to ``models/confusion_matrix.png``.
"""

from __future__ import annotations

import argparse
import os

from midrr_classifier.config import load_config
from midrr_classifier.data_ingestion import load_feature_table
from midrr_classifier.data_schema import LABEL_CLASSES
from midrr_classifier.model_definition import MiDRRClassifier
from midrr_classifier.utils.logging_utils import get_logger, setup_root_logging
from midrr_classifier.utils.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    print_classification_report,
)

logger = get_logger(__name__)


def evaluate_model(
    model_path: str,
    test_csv_path: str,
    config_path: str | None = None,
) -> dict[str, float]:
    """Load a trained model and evaluate it on the test set.

    Args:
        model_path: Path to the serialised ``.pkl`` model artifact.
        test_csv_path: Path to the test-split feature CSV.
        config_path: Optional path to YAML config (for column names).

    Returns:
        Dictionary of evaluation metrics (accuracy, precision, recall,
        f1) as returned by
        :func:`~midrr_classifier.utils.metrics.compute_classification_metrics`.
    """
    cfg = load_config(config_path)

    # ------------------------------------------------------------------
    # Load model and test data
    # ------------------------------------------------------------------
    classifier = MiDRRClassifier(cfg)
    classifier.load(model_path)

    test_df = load_feature_table(test_csv_path)
    X_test = test_df[cfg.feature_cols].to_numpy()
    y_test = test_df[cfg.label_col].to_numpy()

    # ------------------------------------------------------------------
    # Predict and compute metrics
    # ------------------------------------------------------------------
    y_pred = classifier.predict(X_test)

    metrics = compute_classification_metrics(y_test, y_pred)
    logger.info(
        "Evaluation — accuracy: %.4f  precision: %.4f  recall: %.4f  F1: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    print("\n=== MiDRR-Classifier Evaluation ===")
    print_classification_report(y_test, y_pred, labels=LABEL_CLASSES)

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm_path = cfg.confusion_matrix_path
    os.makedirs(os.path.dirname(cm_path) or ".", exist_ok=True)
    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=LABEL_CLASSES,
        save_path=cm_path,
    )
    logger.info("Confusion matrix saved to %s", cm_path)

    return metrics


if __name__ == "__main__":
    setup_root_logging()

    parser = argparse.ArgumentParser(description="Evaluate the MiDRR-Classifier.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the model .pkl file (default: from config).",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        dest="test_csv",
        help="Path to the test feature CSV (default: data/processed/test.csv).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (optional).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model or cfg.model_path
    test_csv = args.test_csv or os.path.join(cfg.processed_data_dir, "test.csv")

    evaluate_model(model_path, test_csv, config_path=args.config)
