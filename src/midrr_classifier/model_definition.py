"""MiDRR-Classifier model wrapper.

This module implements the :class:`MiDRRClassifier`, a thin wrapper
around :class:`sklearn.ensemble.RandomForestClassifier` that:

* Reads hyper-parameters from a :class:`~midrr_classifier.config.MiDRRConfig`.
* Provides a clean :meth:`save` / :meth:`load` interface using
  :mod:`joblib`.
* Keeps training logic out of model construction so the same class can
  be used for both batch training and real-time inference.

Research context
----------------
The MiDRR-Classifier is the ML backend of the thesis project
"Minecraft as a Platform for AI-Enhanced Disaster Risk Simulation and
Adaptive Educational Preparedness."  It predicts a student's disaster
preparedness level (High / Moderate / Low) from behavioural features
derived from their gameplay in a Minecraft fire or earthquake scenario.
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from midrr_classifier.config import MiDRRConfig
from midrr_classifier.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MiDRRClassifier:
    """Random Forest classifier for Minecraft disaster-preparedness prediction.

    Wraps :class:`sklearn.ensemble.RandomForestClassifier` and ties
    it to the project's configuration object so hyper-parameters are
    always consistent between training and evaluation.

    Example::

        from midrr_classifier.config import MiDRRConfig
        from midrr_classifier.model_definition import MiDRRClassifier

        clf = MiDRRClassifier(MiDRRConfig())
        clf.build_model()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        clf.save("models/midrr_rf.pkl")

    Attributes:
        config: The project configuration driving hyper-parameters.
        model: The underlying :class:`~sklearn.ensemble.RandomForestClassifier`
            instance (``None`` until :meth:`build_model` is called).
    """

    def __init__(self, config: MiDRRConfig) -> None:
        self.config = config
        self.model: Optional[RandomForestClassifier] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_model(self) -> RandomForestClassifier:
        """Instantiate and attach the Random Forest using config hyper-parameters.

        Returns:
            The newly created :class:`~sklearn.ensemble.RandomForestClassifier`.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,  # handle potential class imbalance
        )
        logger.info(
            "Built RandomForestClassifier (n_estimators=%d, max_depth=%s, "
            "class_weight=%s, random_state=%d)",
            self.config.n_estimators,
            self.config.max_depth,
            self.config.class_weight,
            self.config.random_state,
        )
        return self.model

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MiDRRClassifier":
        """Fit the model on training data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target label array of shape ``(n_samples,)``.

        Returns:
            ``self`` for method chaining.

        Raises:
            RuntimeError: If :meth:`build_model` has not been called yet.
        """
        self._require_model()
        logger.info("Fitting model on %d samples.", len(y))
        self.model.fit(X, y)  # type: ignore[union-attr]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict preparedness labels for *X*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of predicted labels, e.g. ``["High", "Low", ...]``.
        """
        self._require_model()
        return self.model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for *X*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, n_classes)`` with class
            probabilities in the order given by
            :attr:`model.classes_`.
        """
        self._require_model()
        return self.model.predict_proba(X)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted model to disk using :mod:`joblib`.

        Args:
            path: Destination file path (e.g. ``"models/midrr_rf.pkl"``).
                  Parent directory must exist.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._require_model()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> "MiDRRClassifier":
        """Load a serialised model from *path* and attach it to ``self``.

        Args:
            path: Path to a ``.pkl`` file previously written by
                  :meth:`save`.

        Returns:
            ``self`` for method chaining.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Model is not initialised. Call build_model() first."
            )
