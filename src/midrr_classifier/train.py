"""Training entry-point for the MiDRR-Classifier.

Run from the project root::

    python -m midrr_classifier.train
    # or, with a custom config file:
    python -m midrr_classifier.train --config config.yaml

The script will raise :class:`FileNotFoundError` until a real feature
table CSV is placed in ``data/processed/``.  This is expected — the
scaffold is intentionally data-free.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score

from midrr_classifier.config import load_config
from midrr_classifier.data_ingestion import load_feature_table, split_train_test
from midrr_classifier.model_definition import MiDRRClassifier
from midrr_classifier.utils.logging_utils import get_logger, setup_root_logging

logger = get_logger(__name__)


def main(config_path: str | None = None) -> None:
    """Train and save the MiDRR Random Forest classifier.

    Args:
        config_path: Optional path to a YAML config file.  If
            ``None``, default :class:`~midrr_classifier.config.MiDRRConfig`
            values are used.
    """
    setup_root_logging()
    cfg = load_config(config_path)
    logger.info("Starting MiDRR-Classifier training pipeline.")

    # ------------------------------------------------------------------
    # 1. Load processed feature table
    # ------------------------------------------------------------------
    feature_csv = os.path.join(cfg.processed_data_dir, "features.csv")
    df = load_feature_table(feature_csv)

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    train_df, test_df = split_train_test(
        df,
        test_size=cfg.test_size,
        stratify_col=cfg.label_col,
        random_state=cfg.random_state,
    )

    X_train = train_df[cfg.feature_cols].to_numpy()
    y_train = train_df[cfg.label_col].to_numpy()
    X_test = test_df[cfg.feature_cols].to_numpy()
    y_test = test_df[cfg.label_col].to_numpy()

    # ------------------------------------------------------------------
    # 3. Build and fit model
    # ------------------------------------------------------------------
    classifier = MiDRRClassifier(cfg)
    classifier.build_model()
    classifier.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Quick sanity-check on the training set
    # ------------------------------------------------------------------
    train_preds = classifier.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    logger.info("Training accuracy: %.4f", train_acc)

    # ------------------------------------------------------------------
    # 5. Save model artifact
    # ------------------------------------------------------------------
    os.makedirs(cfg.models_dir, exist_ok=True)
    classifier.save(cfg.model_path)
    logger.info("Training complete.  Model saved to %s", cfg.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MiDRR-Classifier.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (optional).",
    )
    args = parser.parse_args()
    main(config_path=args.config)
