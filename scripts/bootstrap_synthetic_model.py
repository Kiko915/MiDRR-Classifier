"""Bootstrap a synthetic-trained model artifact for container builds / demos.

Why this exists
----------------
``models/*.pkl`` is (correctly) gitignored — it's a build artifact, not source.
That means a fresh clone (or a Docker build) has no ``models/midrr_rf.pkl`` for
the API to serve, and real Necookie telemetry isn't in yet (Phase 3 is still
blocked — see CLAUDE.md). This script fills that gap by generating synthetic
sessions, engineering features, and training a model, all from code already in
this repo (:mod:`midrr_classifier.synth`, ``train.py``) — no binary artifact
needs to live in git history, and the container image stays self-contained.

**This produces a placeholder model.** It reflects fabricated gameplay, not
real students — see :mod:`midrr_classifier.synth` and the ``label_source``
column. Replace it by retraining on real data once Phase 3/4 lands.

Run:
    python scripts/bootstrap_synthetic_model.py

Produces (paths follow ``MiDRRConfig`` defaults, relative to the cwd):
    data/processed/features.csv
    data/processed/train.csv / test.csv
    models/midrr_rf.pkl
    models/metrics.json
"""

from __future__ import annotations

import os

from midrr_classifier import train
from midrr_classifier.config import MiDRRConfig
from midrr_classifier.feature_engineering import build_feature_table
from midrr_classifier.synth import generate_logs
from midrr_classifier.utils.logging_utils import get_logger, setup_root_logging

logger = get_logger(__name__)

# Fixed seed so every image build produces the same placeholder model —
# rebuilding the container shouldn't silently change what it serves.
_SYNTHETIC_SEED = 42


def main() -> None:
    setup_root_logging()
    cfg = MiDRRConfig()

    logger.warning(
        "Bootstrapping a SYNTHETIC-trained placeholder model (seed=%d). "
        "Predictions reflect fabricated gameplay, not real students — "
        "replace with a model retrained on real telemetry before treating "
        "them as meaningful (see docs/MiDRR_ML_Development_Plan.md Phase 3/4).",
        _SYNTHETIC_SEED,
    )

    os.makedirs(cfg.processed_data_dir, exist_ok=True)
    feature_table = build_feature_table(generate_logs(seed=_SYNTHETIC_SEED))
    features_path = os.path.join(cfg.processed_data_dir, "features.csv")
    feature_table.to_csv(features_path, index=False)
    logger.info("Wrote %d synthetic feature rows -> %s", len(feature_table), features_path)

    train.main(do_cv=True, do_tune=False)
    logger.info("Bootstrap complete: %s", cfg.model_path)


if __name__ == "__main__":
    main()
