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
- The full metrics + global feature importances merged into
  ``models/metrics.json`` (alongside the ``cv``/``tuning`` keys train.py
  writes), so every manuscript figure regenerates deterministically.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from midrr_classifier.config import load_config
from midrr_classifier.data_ingestion import load_feature_table
from midrr_classifier.data_schema import LABEL_CLASSES
from midrr_classifier.explainability import compute_global_importances
from midrr_classifier.model_definition import MiDRRClassifier
from midrr_classifier.utils.logging_utils import get_logger, setup_root_logging
from midrr_classifier.utils.metrics import (
    compute_full_metrics,
    plot_confusion_matrix,
    print_classification_report,
    update_metrics_json,
)

logger = get_logger(__name__)

# Features BFP domain intuition expects to rank highly (dev-plan Phase 6).
# Used only for a soft sanity-check log line — never a hard assertion.
_EXPECTED_TOP_FEATURES = ("decision_latency", "hazard_avoidance_ratio")


def _log_importance_sanity_check(ranking: list[str]) -> None:
    """Soft check: do BFP-expected features land in the top half of the ranking?

    This is the dev-plan Phase 6 "sanity check importances against domain
    intuition" step.  It only logs — a surprising ranking on synthetic data is
    informative, not an error, so it must never fail the evaluation.
    """
    top_half = set(ranking[: max(1, len(ranking) // 2)])
    for feat in _EXPECTED_TOP_FEATURES:
        if feat in top_half:
            logger.info("Sanity check OK: '%s' ranks in the top half.", feat)
        else:
            rank = ranking.index(feat) + 1 if feat in ranking else -1
            logger.warning(
                "Sanity check: BFP-expected feature '%s' ranked %d/%d "
                "(below the top half) — worth a look on real data.",
                feat, rank, len(ranking),
            )


def evaluate_model(
    model_path: str,
    test_csv_path: str,
    config_path: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Load a trained model and evaluate it on the held-out test set.

    Produces the full Phase 6 report: accuracy, macro- and weighted-averaged
    precision/recall/F1, a per-class breakdown, the confusion matrix, and
    global feature importances (Gini + permutation) as a cross-check on the
    per-session SHAP values used at serving time.

    Args:
        model_path: Path to the serialised ``.pkl`` model artifact.
        test_csv_path: Path to the test-split feature CSV.
        config_path: Optional path to YAML config (for column names/paths).
        persist: If ``True``, merge the results into ``config.metrics_path``
            and save the confusion-matrix PNG.  Set ``False`` for a dry run.

    Returns:
        ``{"test": {...full metrics...}, "importances": {...gini/permutation...}}``.
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
    # Predict and compute the full metrics report
    # ------------------------------------------------------------------
    y_pred = classifier.predict(X_test)

    test_metrics = compute_full_metrics(y_test, y_pred, labels=LABEL_CLASSES)
    logger.info(
        "Evaluation — accuracy: %.4f  macro-F1: %.4f  weighted-F1: %.4f",
        test_metrics["accuracy"],
        test_metrics["macro"]["f1"],
        test_metrics["weighted"]["f1"],
    )

    print("\n=== MiDRR-Classifier Evaluation ===")
    print_classification_report(y_test, y_pred, labels=LABEL_CLASSES)

    # ------------------------------------------------------------------
    # Global feature importances (cross-check on per-session SHAP)
    # ------------------------------------------------------------------
    importances = compute_global_importances(
        classifier.model,
        X_test,
        y_test,
        feature_cols=cfg.feature_cols,
        random_state=cfg.random_state,
    )
    logger.info(
        "Top features — permutation: %s | Gini: %s",
        importances["ranking_permutation"][:3],
        importances["ranking_gini"][:3],
    )
    _log_importance_sanity_check(importances["ranking_permutation"])

    result: dict[str, Any] = {"test": test_metrics, "importances": importances}

    # ------------------------------------------------------------------
    # Persist: confusion matrix PNG + metrics.json (merged)
    # ------------------------------------------------------------------
    if persist:
        cm_path = cfg.confusion_matrix_path
        os.makedirs(os.path.dirname(cm_path) or ".", exist_ok=True)
        plot_confusion_matrix(y_test, y_pred, labels=LABEL_CLASSES, save_path=cm_path)
        logger.info("Confusion matrix saved to %s", cm_path)

        update_metrics_json(cfg.metrics_path, result)
        logger.info("Metrics merged into %s", cfg.metrics_path)

    return result


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
