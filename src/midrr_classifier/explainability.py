"""SHAP-based per-session explainability for the MiDRR-Classifier.

Why SHAP instead of the default sklearn feature_importances_?
- sklearn's Gini importance is a GLOBAL average across all training samples.
  Every student gets the same ranking regardless of their own behaviour.
- SHAP (SHapley Additive exPlanations) gives a DIFFERENT score per student.
  It answers: "for THIS specific prediction, how much did each feature
  push the result toward the predicted label?"

This matters for adaptive feedback: a student who evacuated quickly but
panicked should hear different advice from one who was calm but slow.

The SHAP value sign tells you direction:
  + positive  → this feature pushed the prediction toward the predicted class
  - negative  → this feature pushed the prediction AWAY from the predicted class
                (i.e. it was actually a strength the model detected)

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
NeurIPS 2017. https://arxiv.org/abs/1705.07874
"""

from __future__ import annotations

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

# Module-level cache: model object id → TreeExplainer.
# TreeExplainer analyses the full tree structure at creation time, so building it
# once per model load (not once per request) keeps inference fast.
_explainer_cache: dict[int, shap.TreeExplainer] = {}


def _get_explainer(rf_model: RandomForestClassifier) -> shap.TreeExplainer:
    key = id(rf_model)
    if key not in _explainer_cache:
        _explainer_cache[key] = shap.TreeExplainer(rf_model)
    return _explainer_cache[key]


def compute_shap_values(
    rf_model: RandomForestClassifier,
    feature_vector: np.ndarray,
    classes: list[str],
    predicted_label: str,
    feature_cols: list[str],
) -> dict[str, float]:
    """Return per-feature SHAP attribution scores for a single prediction.

    Args:
        rf_model: The fitted sklearn RandomForestClassifier (from MiDRRClassifier.model).
        feature_vector: Shape ``(1, n_features)`` — the single student's feature row.
        classes: Ordered class names from ``rf_model.classes_.tolist()``.
        predicted_label: The class the model predicted for this student.
        feature_cols: Feature names in training column order.

    Returns:
        ``{feature_name: shap_value}`` for the predicted class only.

        Positive value  → feature pushed prediction toward ``predicted_label``.
        Negative value  → feature pushed prediction away  from ``predicted_label``
                          (this feature was a relative strength for the student).

    Note:
        SHAP 0.40+ returns a 3-D array ``(samples, features, classes)`` for
        multi-class trees.  Older versions returned a list of 2-D arrays.
        Both shapes are handled here for forward/backward compatibility.
    """
    explainer = _get_explainer(rf_model)
    raw = explainer.shap_values(feature_vector)

    class_idx = classes.index(predicted_label)

    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        # SHAP >= 0.40 multi-class: shape (n_samples, n_features, n_classes)
        values = raw[0, :, class_idx]
    elif isinstance(raw, list):
        # SHAP < 0.40 multi-class: list of (n_samples, n_features) arrays
        values = raw[class_idx][0]
    else:
        # Binary or unexpected shape — use as-is
        values = raw[0] if raw.ndim == 2 else raw

    return {feat: float(v) for feat, v in zip(feature_cols, values)}
