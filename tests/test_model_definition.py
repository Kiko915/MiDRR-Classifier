"""Tests for midrr_classifier.model_definition.

All tests use synthetic data so no real CSV or model file is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from midrr_classifier.config import MiDRRConfig
from midrr_classifier.model_definition import MiDRRClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> MiDRRConfig:
    return MiDRRConfig(n_estimators=10, max_depth=3, random_state=7)


@pytest.fixture()
def classifier(config: MiDRRConfig) -> MiDRRClassifier:
    clf = MiDRRClassifier(config)
    clf.build_model()
    return clf


@pytest.fixture()
def tiny_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Minimal labelled dataset with 3 classes for fit/predict checks."""
    rng = np.random.default_rng(42)
    X = rng.random((30, 6))
    y = np.array(["High"] * 10 + ["Moderate"] * 10 + ["Low"] * 10)
    return X, y


# ---------------------------------------------------------------------------
# build_model tests
# ---------------------------------------------------------------------------


def test_build_model_returns_rf(classifier: MiDRRClassifier) -> None:
    assert isinstance(classifier.model, RandomForestClassifier)


def test_model_n_estimators(config: MiDRRConfig, classifier: MiDRRClassifier) -> None:
    assert classifier.model.n_estimators == config.n_estimators


def test_model_max_depth(config: MiDRRConfig, classifier: MiDRRClassifier) -> None:
    assert classifier.model.max_depth == config.max_depth


def test_model_random_state(config: MiDRRConfig, classifier: MiDRRClassifier) -> None:
    assert classifier.model.random_state == config.random_state


# ---------------------------------------------------------------------------
# fit / predict tests
# ---------------------------------------------------------------------------


def test_fit_predict_shape(
    classifier: MiDRRClassifier,
    tiny_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = tiny_dataset
    classifier.fit(X, y)
    preds = classifier.predict(X)
    assert preds.shape == (len(X),), "predict() output length must match input."


def test_predict_valid_labels(
    classifier: MiDRRClassifier,
    tiny_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = tiny_dataset
    classifier.fit(X, y)
    preds = classifier.predict(X)
    assert set(preds).issubset({"High", "Moderate", "Low"})


def test_predict_proba_shape(
    classifier: MiDRRClassifier,
    tiny_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = tiny_dataset
    classifier.fit(X, y)
    proba = classifier.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert proba.shape[1] == 3  # three preparedness classes


def test_predict_proba_sums_to_one(
    classifier: MiDRRClassifier,
    tiny_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = tiny_dataset
    classifier.fit(X, y)
    proba = classifier.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)


# ---------------------------------------------------------------------------
# Error-state tests
# ---------------------------------------------------------------------------


def test_predict_without_build_raises(config: MiDRRConfig) -> None:
    clf = MiDRRClassifier(config)  # build_model NOT called
    with pytest.raises(RuntimeError, match="build_model"):
        clf.predict(np.zeros((1, 6)))


def test_fit_without_build_raises(
    config: MiDRRConfig,
    tiny_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    clf = MiDRRClassifier(config)
    X, y = tiny_dataset
    with pytest.raises(RuntimeError, match="build_model"):
        clf.fit(X, y)


def test_load_nonexistent_raises(config: MiDRRConfig) -> None:
    clf = MiDRRClassifier(config)
    with pytest.raises(FileNotFoundError):
        clf.load("nonexistent_path/model.pkl")
