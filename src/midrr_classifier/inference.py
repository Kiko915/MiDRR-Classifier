"""Inference interface for the MiDRR-Classifier.

This module is designed to be the **integration boundary** between
the trained ML model and whatever runtime calls it — a Minecraft
plugin, a Flask/FastAPI microservice, or a batch classification job.

Usage example::

    import pandas as pd
    from midrr_classifier.inference import predict_preparedness

    features = pd.Series({
        "evacuation_time": 42.0,
        "decision_delay": 3.5,
        "path_efficiency_ratio": 0.72,
        "hazard_avoidance_ratio": 0.85,
        "interaction_frequency": 0.12,
        "panic_proxy": 15.3,
    })
    label = predict_preparedness(features)
    print(label)  # "HIGH", "MODERATE", or "LOW"

Integration notes
-----------------
For real-time Minecraft integration, wrap :func:`predict_preparedness`
in a lightweight HTTP endpoint (e.g. Flask ``/predict`` POST route).
The Minecraft server plugin (Spigot/Paper/Fabric) POSTs a JSON payload
of features immediately after a simulation run ends and receives the
predicted preparedness label in the response body.

Deployment and REST scaffolding are **out of scope** for this repo and
will be implemented in a separate ``midrr-api`` service.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from midrr_classifier.config import MiDRRConfig, load_config
from midrr_classifier.explainability import compute_shap_values
from midrr_classifier.model_definition import MiDRRClassifier
from midrr_classifier.utils.logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_cached_classifier(model_path: str, config_path: str | None) -> MiDRRClassifier:
    """Load and cache the classifier so repeated calls avoid disk I/O.

    The cache is keyed on ``(model_path, config_path)`` so reloading
    happens automatically if the path changes (e.g. during testing).

    Args:
        model_path: Path to the serialised ``.pkl`` artifact.
        config_path: Optional YAML config path.

    Returns:
        A loaded :class:`~midrr_classifier.model_definition.MiDRRClassifier`.
    """
    cfg = load_config(config_path)
    classifier = MiDRRClassifier(cfg)
    classifier.load(model_path)
    return classifier


def predict_preparedness(
    features_row: pd.Series,
    model_path: str | None = None,
    config_path: str | None = None,
) -> str:
    """Predict the disaster preparedness level from a feature vector.

    This is the primary API for external callers.  It accepts a single
    row of engineered features (as produced by
    :func:`~midrr_classifier.feature_engineering.build_feature_table`)
    and returns one of ``{"HIGH", "MODERATE", "LOW"}``.

    Args:
        features_row: A :class:`pandas.Series` whose index contains
            the feature names listed in
            :attr:`~midrr_classifier.config.MiDRRConfig.feature_cols`.
        model_path: Path to a trained model ``.pkl`` file.  Defaults
            to the path in the default :class:`~midrr_classifier.config.MiDRRConfig`.
        config_path: Optional YAML config path for column definitions.

    Returns:
        Predicted preparedness level: ``"HIGH"``, ``"MODERATE"``, or
        ``"LOW"``.

    Raises:
        FileNotFoundError: If the model file does not exist.
        KeyError: If *features_row* is missing a required feature.
    """
    cfg = load_config(config_path)
    resolved_model_path = model_path or cfg.model_path

    classifier = _get_cached_classifier(resolved_model_path, config_path)

    # Select and order features to match training column order
    feature_vector = features_row[cfg.feature_cols].to_numpy().reshape(1, -1)
    prediction: str = classifier.predict(feature_vector)[0]

    logger.debug("predict_preparedness → %s", prediction)
    return prediction


def predict_preparedness_full(
    features_row: pd.Series,
    model_path: str | None = None,
    config_path: str | None = None,
) -> dict:
    """Extended prediction returning label, class probabilities, and feature importances.

    Use this function in the REST API layer.  The plain
    :func:`predict_preparedness` is kept for callers that only need the label.

    Args:
        features_row: A :class:`pandas.Series` with the six engineered feature
            values (see :attr:`~midrr_classifier.config.MiDRRConfig.feature_cols`).
        model_path: Path to a trained model ``.pkl`` file.
        config_path: Optional YAML config path.

    Returns:
        A dict with three keys:

        - ``"label"`` – predicted class string (``"HIGH"``, ``"MODERATE"``, ``"LOW"``)
        - ``"probabilities"`` – ``{class: float}`` confidence per class, sums to 1.0
        - ``"feature_importances"`` – ``{feature_name: float}`` global Gini importance
          per feature, also sums to 1.0.  Replace with SHAP in Phase 6.

    Raises:
        FileNotFoundError: If the model file does not exist.
        KeyError: If *features_row* is missing a required feature.
    """
    cfg = load_config(config_path)
    resolved_model_path = model_path or cfg.model_path

    classifier = _get_cached_classifier(resolved_model_path, config_path)

    feature_vector = features_row[cfg.feature_cols].to_numpy().reshape(1, -1)

    label: str = classifier.predict(feature_vector)[0]
    proba = classifier.predict_proba(feature_vector)[0]
    classes: list[str] = classifier.model.classes_.tolist()

    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}

    # Global Gini importance — same for every student, kept for training diagnostics.
    global_importances = {
        feat: float(imp)
        for feat, imp in zip(cfg.feature_cols, classifier.model.feature_importances_)
    }

    # Per-student SHAP values — different for every prediction.
    # Positive = pushed toward the predicted label; negative = pushed away (a strength).
    shap_contributions = compute_shap_values(
        rf_model=classifier.model,
        feature_vector=feature_vector,
        classes=classes,
        predicted_label=label,
        feature_cols=cfg.feature_cols,
    )

    logger.debug("predict_preparedness_full → %s (proba=%s)", label, probabilities)
    return {
        "label": label,
        "probabilities": probabilities,
        "shap_values": shap_contributions,
        "feature_importances": global_importances,
    }
