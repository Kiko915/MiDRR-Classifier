"""Grouped cross-validation and hyper-parameter search (Phase 5).

Why this module exists
----------------------
``train.py`` originally fit the model on a *single* train/test split and
reported one accuracy number.  With this project's small N (a few hundred
sessions, two scenarios) that single number is noisy — it can look great or
terrible depending on which students happened to land in the test split.

Cross-validation fixes that: it re-runs the split ``k`` times over different
slices of the data and reports the **mean ± standard deviation** of the score.
The mean is your honest estimate; the std tells you how much you'd trust it.
(Web-dev analogy: instead of load-testing your API once, you run the suite
five times and report the average latency *and* the spread.)

Everything here is **group-aware**, mirroring
:func:`~midrr_classifier.data_ingestion.split_train_test`.  A single student
(``player_id``) may have both a fire and an earthquake session; if those land
in different folds, the model would train on that student's behavioural
signature and then be scored on it — leaking information and inflating the
metric.  :class:`sklearn.model_selection.StratifiedGroupKFold` keeps every
``player_id`` inside a single fold while still balancing the
HIGH/MODERATE/LOW class proportions across folds.

Two public entry points:

* :func:`cross_validate_grouped` — score the *current* config's forest across
  ``k`` folds; this is the "baseline floor" / mean±std report.
* :func:`tune_hyperparameters` — grid-search a small set of forest knobs using
  the same grouped CV, and return the best combination.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate

from midrr_classifier.config import MiDRRConfig
from midrr_classifier.model_definition import build_estimator
from midrr_classifier.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Scores we report for every fold.  ``f1_macro`` (unweighted mean of the
# per-class F1s) is the primary one because it refuses to let the model "win"
# by nailing the majority MODERATE class while ignoring the minority LOW class
# — exactly the failure mode class imbalance creates.  Accuracy is kept as the
# familiar headline number.
SCORING: dict[str, Any] = {
    "accuracy": "accuracy",
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
}

# The metric the tuning search optimises / refits on.
REFIT_METRIC = "f1_macro"

# Default hyper-parameter grid.  Centred on the diagram-spec starting point
# (n_estimators=100, max_depth=8) and kept deliberately small: with small N a
# large grid just overfits the validation folds (and takes far longer).  Each
# key is a standard RandomForestClassifier knob:
#   n_estimators     — number of trees (more = steadier, slower)
#   max_depth        — how deep each tree may grow (shallower = simpler)
#   min_samples_leaf — smallest group a split may leave behind (bigger = simpler)
#   max_features     — features considered per split (de-correlates trees)
#   class_weight     — how hard to push on the minority classes
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"],
}


def _make_cv(config: MiDRRConfig, n_splits: Optional[int] = None) -> StratifiedGroupKFold:
    """Build the grouped, stratified fold splitter used everywhere here."""
    k = n_splits or config.cv_folds
    # StratifiedGroupKFold is deterministic and does not take a random_state;
    # reproducibility comes from the fixed feature order + fixed estimator seed.
    return StratifiedGroupKFold(n_splits=k)


def cross_validate_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Any],
    config: MiDRRConfig,
    n_splits: Optional[int] = None,
) -> dict[str, Any]:
    """Group-aware, stratified k-fold CV of the config's Random Forest.

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        y: Label array ``(n_samples,)`` (``HIGH``/``MODERATE``/``LOW``).
        groups: Group id per row (``player_id``).  No group spans two folds.
        config: Supplies the forest hyper-parameters and ``cv_folds``.
        n_splits: Override ``config.cv_folds`` for this call.

    Returns:
        A JSON-serialisable dict::

            {
              "n_splits": 5,
              "metrics": {
                "accuracy":  {"mean": .., "std": .., "per_fold": [..]},
                "f1_macro":  {"mean": .., "std": .., "per_fold": [..]},
                ...
              }
            }

    Raises:
        ValueError: If a class or group layout makes a stratified grouped
            split impossible (propagated from scikit-learn) — most often
            "too few groups per class for this many folds".
    """
    estimator = build_estimator(config)
    cv = _make_cv(config, n_splits)

    logger.info(
        "Grouped CV: %d folds over %d samples / %d players.",
        cv.get_n_splits(),
        len(y),
        len(set(groups)),
    )

    raw = cross_validate(
        estimator,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=SCORING,
        return_train_score=False,
    )

    metrics: dict[str, Any] = {}
    for name in SCORING:
        scores = raw[f"test_{name}"]
        metrics[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "per_fold": [float(s) for s in scores],
        }

    result = {"n_splits": int(cv.get_n_splits()), "metrics": metrics}
    logger.info(
        "Grouped CV result -> accuracy %.4f +/- %.4f | f1_macro %.4f +/- %.4f",
        metrics["accuracy"]["mean"],
        metrics["accuracy"]["std"],
        metrics["f1_macro"]["mean"],
        metrics["f1_macro"]["std"],
    )
    return result


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Any],
    config: MiDRRConfig,
    param_grid: Optional[dict[str, list[Any]]] = None,
    n_splits: Optional[int] = None,
) -> dict[str, Any]:
    """Grid-search Random Forest hyper-parameters with grouped stratified CV.

    Optimises :data:`REFIT_METRIC` (macro-F1).  The search reuses the same
    grouped splitter as :func:`cross_validate_grouped`, so the winning params
    are chosen under the same no-player-leakage rule the final evaluation uses.

    Args:
        X, y, groups: As in :func:`cross_validate_grouped`.
        config: Base config (its ``random_state`` seeds every candidate forest;
            grid values override the rest).
        param_grid: Search space.  Defaults to :data:`DEFAULT_PARAM_GRID`.
        n_splits: Override ``config.cv_folds``.

    Returns:
        A JSON-serialisable dict::

            {
              "best_params": {...},
              "best_score": 0.87,          # mean macro-F1 of the winner
              "refit_metric": "f1_macro",
              "n_candidates": 144,
              "top": [ {params, mean_f1_macro, std_f1_macro}, ... ]  # best 5
            }

        ``best_params`` is ready to splat into :func:`build_estimator` or to
        lock into ``config.py``.
    """
    grid = param_grid or DEFAULT_PARAM_GRID
    cv = _make_cv(config, n_splits)

    # Base estimator carries only the seed; every other param comes from the
    # grid so the search fully controls the compared knobs.
    base = build_estimator(
        config,
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
    )

    search = GridSearchCV(
        estimator=base,
        param_grid=grid,
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    n_candidates = int(np.prod([len(v) for v in grid.values()]))
    logger.info(
        "Hyper-parameter search: %d candidates x %d folds = %d fits.",
        n_candidates,
        cv.get_n_splits(),
        n_candidates * cv.get_n_splits(),
    )

    search.fit(X, y, groups=groups)

    results = search.cv_results_
    order = np.argsort(results[f"mean_test_{REFIT_METRIC}"])[::-1]
    top = [
        {
            "params": results["params"][i],
            f"mean_{REFIT_METRIC}": float(results[f"mean_test_{REFIT_METRIC}"][i]),
            f"std_{REFIT_METRIC}": float(results[f"std_test_{REFIT_METRIC}"][i]),
        }
        for i in order[:5]
    ]

    result = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "refit_metric": REFIT_METRIC,
        "n_candidates": n_candidates,
        "top": top,
    }
    logger.info(
        "Best params (%s = %.4f): %s",
        REFIT_METRIC,
        result["best_score"],
        result["best_params"],
    )
    return result
