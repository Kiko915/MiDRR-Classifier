"""Tests for Phase 6 evaluation: full metrics, global importances, wiring.

All synthetic; no real CSV or pre-trained model needed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from midrr_classifier.config import MiDRRConfig
from midrr_classifier.data_schema import LABEL_CLASSES
from midrr_classifier.evaluate import evaluate_model
from midrr_classifier.explainability import compute_global_importances
from midrr_classifier.model_definition import MiDRRClassifier
from midrr_classifier.utils.metrics import (
    compute_full_metrics,
    print_classification_report,
)

FEATURES = MiDRRConfig().feature_cols


def _make_feature_frame(n_per_class: int = 15, seed: int = 0) -> pd.DataFrame:
    """Separable 9-feature table (one scenario, expert-labeled)."""
    rng = np.random.default_rng(seed)
    centers = {"HIGH": 0.8, "MODERATE": 0.5, "LOW": 0.2}
    rows = []
    for cls, center in centers.items():
        for i in range(n_per_class):
            row = {f: float(rng.normal(center, 0.05)) for f in FEATURES}
            row["player_id"] = f"{cls[:1]}{i:02d}"
            row["scenario_type"] = "fire"
            row["preparedness_level"] = cls
            row["label_source"] = "expert"
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_full_metrics
# ---------------------------------------------------------------------------


def test_full_metrics_structure() -> None:
    y_true = np.array(["HIGH", "MODERATE", "LOW", "HIGH"])
    y_pred = np.array(["HIGH", "MODERATE", "LOW", "MODERATE"])
    m = compute_full_metrics(y_true, y_pred, labels=LABEL_CLASSES)

    assert set(m) == {
        "accuracy", "macro", "weighted", "per_class",
        "confusion_matrix", "n_samples", "labels",
    }
    for avg in ("macro", "weighted"):
        assert set(m[avg]) == {"precision", "recall", "f1"}
    assert set(m["per_class"]) == set(LABEL_CLASSES)
    for stats in m["per_class"].values():
        assert set(stats) == {"precision", "recall", "f1", "support"}
    assert m["n_samples"] == 4


def test_full_metrics_confusion_matrix_shape() -> None:
    y_true = np.array(["HIGH", "MODERATE", "LOW"])
    y_pred = np.array(["HIGH", "MODERATE", "LOW"])
    cm = compute_full_metrics(y_true, y_pred, labels=LABEL_CLASSES)["confusion_matrix"]
    assert cm["labels"] == LABEL_CLASSES
    assert np.array(cm["matrix"]).shape == (3, 3)


def test_full_metrics_perfect_prediction() -> None:
    y = np.array(["HIGH", "MODERATE", "LOW", "HIGH", "LOW"])
    m = compute_full_metrics(y, y, labels=LABEL_CLASSES)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro"]["f1"] == pytest.approx(1.0)
    # Diagonal confusion matrix.
    mat = np.array(m["confusion_matrix"]["matrix"])
    assert np.trace(mat) == mat.sum()


def test_full_metrics_support_sums_to_n() -> None:
    y_true = np.array(["HIGH"] * 3 + ["MODERATE"] * 2 + ["LOW"])
    y_pred = y_true.copy()
    m = compute_full_metrics(y_true, y_pred, labels=LABEL_CLASSES)
    assert sum(c["support"] for c in m["per_class"].values()) == 6


def test_full_metrics_json_serializable() -> None:
    y = np.array(["HIGH", "MODERATE", "LOW"])
    json.dumps(compute_full_metrics(y, y, labels=LABEL_CLASSES))  # must not raise


def test_printed_report_labels_align_with_support(capsys) -> None:
    # Regression: LABEL_CLASSES is not alphabetical, so the printed report must
    # pass labels= (not only target_names=) or MODERATE/LOW rows get swapped.
    y = np.array(["HIGH"] + ["MODERATE"] * 3 + ["LOW"] * 2)  # supports 1/3/2
    print_classification_report(y, y, labels=LABEL_CLASSES)
    out = capsys.readouterr().out

    def _support(line_label: str) -> int:
        line = next(ln for ln in out.splitlines() if ln.strip().startswith(line_label))
        return int(line.split()[-1])

    assert _support("MODERATE") == 3
    assert _support("LOW") == 2


# ---------------------------------------------------------------------------
# compute_global_importances
# ---------------------------------------------------------------------------


@pytest.fixture()
def fitted_model_and_data():
    df = _make_feature_frame()
    X = df[FEATURES].to_numpy()
    y = df["preparedness_level"].to_numpy()
    clf = MiDRRClassifier(MiDRRConfig(n_estimators=25, random_state=0))
    clf.build_model()
    clf.fit(X, y)
    return clf.model, X, y


def test_importances_keys_cover_all_features(fitted_model_and_data) -> None:
    model, X, y = fitted_model_and_data
    imp = compute_global_importances(model, X, y, FEATURES, n_repeats=5)
    assert set(imp["gini"]) == set(FEATURES)
    assert set(imp["permutation"]) == set(FEATURES)
    for stats in imp["permutation"].values():
        assert set(stats) == {"mean", "std"}


def test_importances_rankings_sorted(fitted_model_and_data) -> None:
    model, X, y = fitted_model_and_data
    imp = compute_global_importances(model, X, y, FEATURES, n_repeats=5)

    gini_vals = [imp["gini"][f] for f in imp["ranking_gini"]]
    perm_vals = [imp["permutation"][f]["mean"] for f in imp["ranking_permutation"]]
    assert gini_vals == sorted(gini_vals, reverse=True)
    assert perm_vals == sorted(perm_vals, reverse=True)
    assert set(imp["ranking_gini"]) == set(FEATURES)


def test_importances_json_serializable(fitted_model_and_data) -> None:
    model, X, y = fitted_model_and_data
    json.dumps(compute_global_importances(model, X, y, FEATURES, n_repeats=5))


# ---------------------------------------------------------------------------
# evaluate_model end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture()
def trained_setup(tmp_path):
    """Write a model .pkl + test.csv under tmp dirs; return a YAML config path."""
    import yaml

    proc = tmp_path / "processed"
    models = tmp_path / "models"
    proc.mkdir()
    models.mkdir()

    df = _make_feature_frame()
    df.to_csv(proc / "test.csv", index=False)

    cfg = MiDRRConfig(
        processed_data_dir=str(proc),
        models_dir=str(models),
        n_estimators=25,
        random_state=0,
    )
    clf = MiDRRClassifier(cfg)
    clf.build_model()
    clf.fit(df[FEATURES].to_numpy(), df["preparedness_level"].to_numpy())
    clf.save(cfg.model_path)

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(
            {"processed_data_dir": str(proc), "models_dir": str(models),
             "n_estimators": 25, "random_state": 0},
            fh,
        )
    return cfg, str(cfg_path)


def test_evaluate_model_writes_metrics_and_png(trained_setup) -> None:
    cfg, cfg_path = trained_setup
    test_csv = os.path.join(cfg.processed_data_dir, "test.csv")

    result = evaluate_model(cfg.model_path, test_csv, config_path=cfg_path)

    assert set(result) == {"test", "importances"}
    assert os.path.exists(cfg.confusion_matrix_path)
    assert os.path.exists(cfg.metrics_path)

    with open(cfg.metrics_path) as fh:
        saved = json.load(fh)
    assert "test" in saved and "importances" in saved
    assert saved["test"]["accuracy"] == pytest.approx(1.0)  # separable data


def test_evaluate_model_merges_not_clobbers(trained_setup) -> None:
    # A prior phase's key (e.g. train.py's "cv") must survive evaluation.
    cfg, cfg_path = trained_setup
    from midrr_classifier.utils.metrics import update_metrics_json

    update_metrics_json(cfg.metrics_path, {"cv": {"placeholder": True}})
    evaluate_model(
        cfg.model_path,
        os.path.join(cfg.processed_data_dir, "test.csv"),
        config_path=cfg_path,
    )
    with open(cfg.metrics_path) as fh:
        saved = json.load(fh)
    assert set(saved) >= {"cv", "test", "importances"}


def test_evaluate_model_persist_false_writes_nothing(trained_setup) -> None:
    cfg, cfg_path = trained_setup
    evaluate_model(
        cfg.model_path,
        os.path.join(cfg.processed_data_dir, "test.csv"),
        config_path=cfg_path,
        persist=False,
    )
    assert not os.path.exists(cfg.metrics_path)
    assert not os.path.exists(cfg.confusion_matrix_path)
