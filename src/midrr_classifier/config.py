"""Central configuration for the MiDRR-Classifier project.

All tuneable knobs live here so that train.py, evaluate.py, and
inference.py share a single source of truth.  The config can be
constructed from defaults or overridden with a YAML file::

    cfg = load_config("config.yaml")

YAML format example::

    raw_data_dir: data/raw
    processed_data_dir: data/processed
    models_dir: models
    n_estimators: 200
    max_depth: 10
    random_state: 0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MiDRRConfig:
    """Holds all runtime configuration for the MiDRR-Classifier.

    Attributes:
        raw_data_dir: Directory containing raw gameplay CSV logs.
        processed_data_dir: Directory for feature-engineered CSVs.
        models_dir: Directory where trained model artifacts are saved.
        n_estimators: Number of trees in the Random Forest.
        max_depth: Maximum tree depth (None = unlimited).
        class_weight: Passed straight to ``RandomForestClassifier`` to handle
            class imbalance (e.g. ``"balanced"`` or ``None``).
        random_state: Seed for reproducibility.
        label_col: Name of the target column in the feature table.
        feature_cols: Ordered list of feature column names fed to the
            model.  Must match the output of
            :func:`~midrr_classifier.feature_engineering.build_feature_table`.
        test_size: Fraction of data reserved for the test split.
        turso_database_url: libSQL connection URL for the Turso `sessions`
            table (Phase 2.5 ingestion adapter). ``None`` = CSV-only mode.
        turso_auth_token: Turso auth token. Defaults to the
            ``TURSO_AUTH_TOKEN`` env var so the token is never hardcoded.
    """

    # Paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    models_dir: str = "models"

    # Random Forest hyperparameters
    # Locked to the BFP-revised simulation design spec (2026-07-01 diagrams).
    n_estimators: int = 100
    max_depth: Optional[int] = 8
    class_weight: Optional[str] = "balanced"
    random_state: int = 42

    # Column configuration
    label_col: str = "preparedness_level"
    feature_cols: list[str] = field(
        default_factory=lambda: [
            "decision_latency",
            "spray_accuracy",
            "path_efficiency_ratio",
            "hazard_avoidance_ratio",
            "evacuation_time",
            "interaction_frequency",
            "resource_utilization",
            "panic_proxy",
            "situational_awareness",
        ]
    )

    # Train / test split
    test_size: float = 0.3

    # Turso Cloud DB (libSQL) — live ingestion source (Phase 2.5).
    # Leave both None to use CSV-only ingestion (data_ingestion.load_raw_logs).
    turso_database_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("TURSO_DATABASE_URL")
    )
    turso_auth_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("TURSO_AUTH_TOKEN")
    )

    @property
    def model_path(self) -> str:
        """Canonical path to the serialised model artifact."""
        return os.path.join(self.models_dir, "midrr_rf.pkl")

    @property
    def confusion_matrix_path(self) -> str:
        """Path for the saved confusion-matrix PNG."""
        return os.path.join(self.models_dir, "confusion_matrix.png")


def load_config(path: Optional[str] = None) -> MiDRRConfig:
    """Build a :class:`MiDRRConfig`, optionally merging overrides from YAML.

    Args:
        path: Path to a YAML file whose top-level keys override the
            dataclass defaults.  Any key not present in the YAML uses
            the default value.  Pass ``None`` (default) to use only
            the built-in defaults.

    Returns:
        A fully populated :class:`MiDRRConfig` instance.

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        KeyError: If the YAML contains an unknown config key.
    """
    cfg = MiDRRConfig()

    if path is None:
        return cfg

    import yaml  # deferred so pyyaml is optional for non-YAML users

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        overrides: dict = yaml.safe_load(fh) or {}

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown config key '{key}' in {path}")
        setattr(cfg, key, value)

    return cfg
