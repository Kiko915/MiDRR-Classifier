"""Tests for midrr_classifier.model_selection (Phase 5 grouped CV + tuning).

Uses a small, well-separated synthetic dataset with an explicit ``groups``
array (multiple sessions per player) so the group-aware behaviour is exercised
without needing real data or the full synth pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from midrr_classifier.config import MiDRRConfig
from midrr_classifier.model_selection import (
    DEFAULT_PARAM_GRID,
    cross_validate_grouped,
    tune_hyperparameters,
)


def _make_grouped_dataset(
    players_per_class: int = 12, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separable 9-feature dataset; each player has 2 sessions (fire+quake)."""
    rng = np.random.default_rng(seed)
    centers = {"HIGH": 0.8, "MODERATE": 0.5, "LOW": 0.2}
    X, y, groups = [], [], []
    for cls, center in centers.items():
        for p in range(players_per_class):
            pid = f"{cls[:1]}{p:02d}"
            for _ in range(2):  # two sessions -> same player_id, must not leak
                X.append(rng.normal(center, 0.05, size=9))
                y.append(cls)
                groups.append(pid)
    return np.array(X), np.array(y), np.array(groups)


@pytest.fixture()
def cfg() -> MiDRRConfig:
    return MiDRRConfig(n_estimators=25, max_depth=None, cv_folds=3, random_state=0)


@pytest.fixture()
def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _make_grouped_dataset()


# ---------------------------------------------------------------------------
# cross_validate_grouped
# ---------------------------------------------------------------------------


def test_cv_reports_all_metrics(cfg, dataset) -> None:
    X, y, groups = dataset
    result = cross_validate_grouped(X, y, groups, cfg)
    assert result["n_splits"] == 3
    for name in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
        assert name in result["metrics"]
        assert set(result["metrics"][name]) == {"mean", "std", "per_fold"}


def test_cv_per_fold_length_matches_splits(cfg, dataset) -> None:
    X, y, groups = dataset
    result = cross_validate_grouped(X, y, groups, cfg)
    assert len(result["metrics"]["accuracy"]["per_fold"]) == 3


def test_cv_scores_in_unit_interval(cfg, dataset) -> None:
    X, y, groups = dataset
    result = cross_validate_grouped(X, y, groups, cfg)
    for stats in result["metrics"].values():
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["std"] >= 0.0


def test_cv_learns_separable_data(cfg, dataset) -> None:
    # Data is deliberately separable; a working forest should score high.
    X, y, groups = dataset
    result = cross_validate_grouped(X, y, groups, cfg)
    assert result["metrics"]["f1_macro"]["mean"] > 0.8


def test_cv_respects_n_splits_override(cfg, dataset) -> None:
    X, y, groups = dataset
    result = cross_validate_grouped(X, y, groups, cfg, n_splits=4)
    assert result["n_splits"] == 4
    assert len(result["metrics"]["accuracy"]["per_fold"]) == 4


def test_cv_is_json_serializable(cfg, dataset) -> None:
    import json

    X, y, groups = dataset
    json.dumps(cross_validate_grouped(X, y, groups, cfg))  # must not raise


# ---------------------------------------------------------------------------
# tune_hyperparameters
# ---------------------------------------------------------------------------


def test_tune_returns_best_from_grid(cfg, dataset) -> None:
    X, y, groups = dataset
    grid = {"n_estimators": [10, 25], "max_depth": [2, None]}
    result = tune_hyperparameters(X, y, groups, cfg, param_grid=grid)

    assert result["best_params"]["n_estimators"] in grid["n_estimators"]
    assert result["best_params"]["max_depth"] in grid["max_depth"]
    assert result["n_candidates"] == 4
    assert result["refit_metric"] == "f1_macro"
    assert 0.0 <= result["best_score"] <= 1.0


def test_tune_top_is_sorted_desc(cfg, dataset) -> None:
    X, y, groups = dataset
    grid = {"n_estimators": [10, 25], "max_depth": [2, 4, None]}
    result = tune_hyperparameters(X, y, groups, cfg, param_grid=grid)
    scores = [entry["mean_f1_macro"] for entry in result["top"]]
    assert scores == sorted(scores, reverse=True)


def test_tune_best_matches_top_of_ranking(cfg, dataset) -> None:
    X, y, groups = dataset
    grid = {"n_estimators": [10, 25], "max_depth": [2, None]}
    result = tune_hyperparameters(X, y, groups, cfg, param_grid=grid)
    assert result["top"][0]["mean_f1_macro"] == pytest.approx(result["best_score"])


def test_tune_is_json_serializable(cfg, dataset) -> None:
    import json

    X, y, groups = dataset
    grid = {"n_estimators": [10], "max_depth": [None]}
    json.dumps(tune_hyperparameters(X, y, groups, cfg, param_grid=grid))


def test_default_grid_keys_are_valid_rf_params(cfg) -> None:
    # Guard against a typo'd grid key that GridSearchCV would reject at fit time.
    from sklearn.ensemble import RandomForestClassifier

    valid = RandomForestClassifier().get_params()
    for key in DEFAULT_PARAM_GRID:
        assert key in valid
