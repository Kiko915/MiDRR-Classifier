"""Training entry-point for the MiDRR-Classifier.

Run from the project root::

    python -m midrr_classifier.train                # split + grouped CV + fit + save
    python -m midrr_classifier.train --tune         # also grid-search hyperparameters
    python -m midrr_classifier.train --tune --apply-best   # ...and fit the winner
    python -m midrr_classifier.train --no-cv        # skip CV (fastest; just fit)
    python -m midrr_classifier.train --config config.yaml

What it does (Phase 5)
----------------------
1. Loads ``data/processed/features.csv`` and makes a group-aware, stratified
   train/test split (no ``player_id`` leaks across the two).
2. Persists ``train.csv`` / ``test.csv`` so ``evaluate.py`` scores the *same*
   held-out players.
3. Runs **grouped k-fold cross-validation** on the train split and reports the
   mean +/- std of accuracy and macro-F1 — the honest "how good is this model"
   number, not a single lucky split.  This is recorded as the baseline floor.
4. Optionally grid-searches hyperparameters (``--tune``) and, with
   ``--apply-best``, fits the winning configuration.
5. Fits the final model on the full train split and saves it.
6. Merges every number above into ``models/metrics.json`` (Phase 6 evaluation
   later adds the held-out test scores to the same file).

The script raises :class:`FileNotFoundError` until a real (or synthetic)
feature table CSV is placed in ``data/processed/``.
"""

from __future__ import annotations

import argparse
import os

from sklearn.metrics import accuracy_score

from midrr_classifier.config import MiDRRConfig, load_config
from midrr_classifier.data_ingestion import load_feature_table, split_train_test
from midrr_classifier.model_definition import MiDRRClassifier
from midrr_classifier.model_selection import (
    cross_validate_grouped,
    tune_hyperparameters,
)
from midrr_classifier.utils.logging_utils import get_logger, setup_root_logging
from midrr_classifier.utils.metrics import update_metrics_json

logger = get_logger(__name__)

# Hyperparameters the tuning search is allowed to lock back into the config.
_TUNABLE_FIELDS = (
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "class_weight",
)


def _apply_best_params(cfg: MiDRRConfig, best_params: dict) -> None:
    """Overwrite the tunable config fields in-place with search winners."""
    for field in _TUNABLE_FIELDS:
        if field in best_params:
            setattr(cfg, field, best_params[field])
    logger.info("Applied tuned hyperparameters to config for final fit.")


def _current_hyperparameters(cfg: MiDRRConfig) -> dict:
    """Snapshot the hyperparameters that will produce the saved model."""
    return {field: getattr(cfg, field) for field in _TUNABLE_FIELDS}


def main(
    config_path: str | None = None,
    do_cv: bool = True,
    do_tune: bool = False,
    apply_best: bool = False,
) -> dict:
    """Train, cross-validate, optionally tune, and save the classifier.

    Args:
        config_path: Optional path to a YAML config file.
        do_cv: Run grouped k-fold CV on the train split (the baseline floor).
        do_tune: Grid-search hyperparameters via grouped CV.
        apply_best: If tuning ran, fit the final model on the winning params
            (otherwise the search is reported but the config defaults are used).

    Returns:
        The metrics dict written to ``models/metrics.json``.
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
    # 2. Group-aware, stratified train / test split (persist both)
    # ------------------------------------------------------------------
    train_df, test_df = split_train_test(
        df,
        test_size=cfg.test_size,
        stratify_col=cfg.label_col,
        random_state=cfg.random_state,
    )

    os.makedirs(cfg.processed_data_dir, exist_ok=True)
    train_path = os.path.join(cfg.processed_data_dir, "train.csv")
    test_path = os.path.join(cfg.processed_data_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Persisted splits: %s / %s", train_path, test_path)

    X_train = train_df[cfg.feature_cols].to_numpy()
    y_train = train_df[cfg.label_col].to_numpy()
    groups_train = train_df["player_id"].to_numpy()

    report: dict = {}

    # ------------------------------------------------------------------
    # 3. Grouped cross-validation (baseline floor: mean +/- std)
    # ------------------------------------------------------------------
    if do_cv:
        report["cv"] = cross_validate_grouped(X_train, y_train, groups_train, cfg)

    # ------------------------------------------------------------------
    # 4. Optional hyperparameter search
    # ------------------------------------------------------------------
    if do_tune:
        tuning = tune_hyperparameters(X_train, y_train, groups_train, cfg)
        report["tuning"] = tuning
        if apply_best:
            _apply_best_params(cfg, tuning["best_params"])

    # ------------------------------------------------------------------
    # 5. Fit final model on the full train split
    # ------------------------------------------------------------------
    classifier = MiDRRClassifier(cfg)
    classifier.build_model()
    classifier.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, classifier.predict(X_train))
    logger.info("Final-fit training accuracy: %.4f", train_acc)
    report["final_fit"] = {
        "train_accuracy": float(train_acc),
        "hyperparameters": _current_hyperparameters(cfg),
        "n_train_rows": int(len(y_train)),
        "n_train_players": int(train_df["player_id"].nunique()),
    }

    # ------------------------------------------------------------------
    # 6. Persist model + metrics
    # ------------------------------------------------------------------
    os.makedirs(cfg.models_dir, exist_ok=True)
    classifier.save(cfg.model_path)
    update_metrics_json(cfg.metrics_path, report)
    logger.info(
        "Training complete. Model -> %s | metrics -> %s",
        cfg.model_path,
        cfg.metrics_path,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MiDRR-Classifier.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (optional).",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip grouped cross-validation (just split, fit, save).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Grid-search hyperparameters using grouped CV.",
    )
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="With --tune, fit the final model on the winning hyperparameters.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        do_cv=not args.no_cv,
        do_tune=args.tune,
        apply_best=args.apply_best,
    )
