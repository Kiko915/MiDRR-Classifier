"""Contract tests for label casing.

BERONG_SMP_WEB expects uppercase labels ("HIGH", "MODERATE", "LOW").
These tests guard against silent casing drift that would break the
dashboard integration without raising any runtime error.
"""

from __future__ import annotations

from midrr_classifier.data_schema import LABEL_CLASSES


def test_label_classes_are_uppercase() -> None:
    for label in LABEL_CLASSES:
        assert label == label.upper(), (
            f"Label {label!r} must be uppercase to match dashboard contract. "
            "See docs/MiDRR_ML_Development_Plan.md §2."
        )


def test_label_classes_exact_values() -> None:
    assert LABEL_CLASSES == ["HIGH", "MODERATE", "LOW"], (
        "LABEL_CLASSES must be exactly ['HIGH', 'MODERATE', 'LOW'] "
        "in this order (matches dashboard PrepLevel enum and confusion matrix axis)."
    )


def test_label_classes_count() -> None:
    assert len(LABEL_CLASSES) == 3
