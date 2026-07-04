"""Tests for Phase 9 manuscript-figure generation from metrics.json.

All synthetic; every fixture reuses the real Phase 5/6 functions so the
metrics dict shape can never drift from what train.py/evaluate.py actually
produce.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from midrr_classifier import reporting
from midrr_classifier.config import MiDRRConfig
from midrr_classifier.data_schema import LABEL_CLASSES
from midrr_classifier.explainability import compute_global_importances
from midrr_classifier.model_definition import MiDRRClassifier
from midrr_classifier.model_selection import cross_validate_grouped
from midrr_classifier.utils.metrics import compute_full_metrics

FEATURES = MiDRRConfig().feature_cols


def _make_dataset(n_per_class: int = 15, seed: int = 0) -> pd.DataFrame:
    """Separable 9-feature table with distinct player_ids (for group-aware CV)."""
    rng = np.random.default_rng(seed)
    centers = {"HIGH": 0.8, "MODERATE": 0.5, "LOW": 0.2}
    rows = []
    for cls, center in centers.items():
        for i in range(n_per_class):
            row = {f: float(rng.normal(center, 0.05)) for f in FEATURES}
            row["player_id"] = f"{cls[:1]}{i:02d}"
            row["preparedness_level"] = cls
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def full_metrics() -> dict:
    """A complete metrics dict: cv + test + importances, from real functions."""
    df = _make_dataset()
    X = df[FEATURES].to_numpy()
    y = df["preparedness_level"].to_numpy()
    groups = df["player_id"].to_numpy()

    cfg = MiDRRConfig(n_estimators=25, random_state=0, cv_folds=3)
    cv = cross_validate_grouped(X, y, groups, cfg)

    clf = MiDRRClassifier(cfg)
    clf.build_model()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    test_metrics = compute_full_metrics(y, y_pred, labels=LABEL_CLASSES)
    importances = compute_global_importances(clf.model, X, y, FEATURES, n_repeats=5)

    return {"cv": cv, "test": test_metrics, "importances": importances}


# ---------------------------------------------------------------------------
# load_metrics
# ---------------------------------------------------------------------------


def test_load_metrics_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        reporting.load_metrics(str(tmp_path / "nope.json"))


def test_load_metrics_round_trips(tmp_path, full_metrics) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(full_metrics))
    loaded = reporting.load_metrics(str(path))
    assert set(loaded) == {"cv", "test", "importances"}


# ---------------------------------------------------------------------------
# Confusion matrix figure
# ---------------------------------------------------------------------------


def test_confusion_matrix_figure_written(tmp_path, full_metrics) -> None:
    out = tmp_path / "cm.png"
    result = reporting.plot_confusion_matrix_figure(full_metrics, str(out))
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_confusion_matrix_figure_requires_test_key() -> None:
    with pytest.raises(KeyError):
        reporting.plot_confusion_matrix_figure({}, "unused.png")


# ---------------------------------------------------------------------------
# Feature importance figure
# ---------------------------------------------------------------------------


def test_feature_importance_figure_written(tmp_path, full_metrics) -> None:
    out = tmp_path / "imp.png"
    result = reporting.plot_feature_importance_figure(full_metrics, str(out))
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_feature_importance_figure_requires_importances_key() -> None:
    with pytest.raises(KeyError):
        reporting.plot_feature_importance_figure({}, "unused.png")


# ---------------------------------------------------------------------------
# CV variance figure
# ---------------------------------------------------------------------------


def test_cv_variance_figure_written(tmp_path, full_metrics) -> None:
    out = tmp_path / "cv.png"
    result = reporting.plot_cv_variance_figure(full_metrics, str(out))
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_cv_variance_figure_requires_cv_key() -> None:
    with pytest.raises(KeyError):
        reporting.plot_cv_variance_figure({}, "unused.png")


# ---------------------------------------------------------------------------
# Per-class table
# ---------------------------------------------------------------------------


def test_per_class_table_rows_and_columns(full_metrics) -> None:
    df = reporting.render_per_class_table(full_metrics)
    assert list(df["class"]) == list(LABEL_CLASSES) + ["macro avg", "weighted avg"]
    assert list(df.columns) == ["class", "precision", "recall", "f1", "support"]


def test_per_class_table_support_sums_match_n(full_metrics) -> None:
    df = reporting.render_per_class_table(full_metrics)
    per_class_support = df[df["class"].isin(LABEL_CLASSES)]["support"].sum()
    assert per_class_support == full_metrics["test"]["n_samples"]


def test_per_class_table_writes_csv_and_markdown(tmp_path, full_metrics) -> None:
    csv_path = tmp_path / "table.csv"
    md_path = tmp_path / "table.md"
    reporting.render_per_class_table(
        full_metrics, save_path_csv=str(csv_path), save_path_md=str(md_path)
    )

    reloaded = pd.read_csv(csv_path)
    assert len(reloaded) == 5  # 3 classes + macro + weighted

    md_text = md_path.read_text(encoding="utf-8")
    assert "| class |" in md_text
    assert "HIGH" in md_text
    assert "macro avg" in md_text


def test_per_class_table_requires_test_key() -> None:
    with pytest.raises(KeyError):
        reporting.render_per_class_table({})


def test_per_class_table_row_order_survives_metrics_json_round_trip(
    tmp_path, full_metrics
) -> None:
    # Regression: update_metrics_json persists with `sort_keys=True`, which
    # alphabetizes per_class's dict keys on disk (HIGH, LOW, MODERATE) even
    # though LABEL_CLASSES is HIGH/MODERATE/LOW. Row order must come from the
    # "labels" list (untouched by sort_keys), not from dict iteration order.
    from midrr_classifier.utils.metrics import update_metrics_json

    metrics_path = tmp_path / "metrics.json"
    update_metrics_json(str(metrics_path), full_metrics)

    reloaded = reporting.load_metrics(str(metrics_path))
    assert list(reloaded["test"]["per_class"].keys()) == ["HIGH", "LOW", "MODERATE"]

    df = reporting.render_per_class_table(reloaded)
    assert list(df["class"]) == list(LABEL_CLASSES) + ["macro avg", "weighted avg"]


# ---------------------------------------------------------------------------
# generate_all_figures orchestration
# ---------------------------------------------------------------------------


def test_generate_all_figures_full_metrics(tmp_path, full_metrics) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(full_metrics))
    out_dir = tmp_path / "figures"

    saved = reporting.generate_all_figures(
        metrics_path=str(metrics_path), output_dir=str(out_dir)
    )

    assert set(saved) == {
        "confusion_matrix",
        "feature_importance",
        "cv_variance",
        "per_class_table_csv",
        "per_class_table_md",
    }
    for path in saved.values():
        assert os.path.exists(path)


def test_generate_all_figures_partial_metrics_skips_missing(tmp_path) -> None:
    # Only "cv" present — e.g. train.py ran but evaluate.py hasn't yet.
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "cv": {
                    "n_splits": 3,
                    "metrics": {
                        "accuracy": {
                            "mean": 0.9,
                            "std": 0.05,
                            "per_fold": [0.85, 0.9, 0.95],
                        },
                    },
                }
            }
        )
    )
    out_dir = tmp_path / "figures"

    saved = reporting.generate_all_figures(
        metrics_path=str(metrics_path), output_dir=str(out_dir)
    )

    assert set(saved) == {"cv_variance"}
    assert os.path.exists(saved["cv_variance"])


def test_generate_all_figures_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        reporting.generate_all_figures(
            metrics_path=str(tmp_path / "nope.json"), output_dir=str(tmp_path / "out")
        )
