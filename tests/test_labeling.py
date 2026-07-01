"""Tests for midrr_classifier.labeling (Phase 2.5 step 3).

Covers the rule-based score->label mapping, the phase-based cross-check,
expert-vs-rule label precedence (including NaN-safety), and the
rule/expert agreement (kappa) helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from midrr_classifier.labeling import (
    attach_labels,
    phase_outcome_label,
    resolve_label,
    rule_based_label,
    rule_expert_agreement,
)


# ---------------------------------------------------------------------------
# rule_based_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (100.0, "HIGH"),
        (75.0, "HIGH"),  # boundary — inclusive
        (74.9, "MODERATE"),
        (40.0, "MODERATE"),  # boundary — inclusive
        (39.9, "LOW"),
        (0.0, "LOW"),
    ],
)
def test_rule_based_label_thresholds(score: float, expected: str) -> None:
    assert rule_based_label(score) == expected


# ---------------------------------------------------------------------------
# phase_outcome_label
# ---------------------------------------------------------------------------


def test_phase_outcome_prevention_is_high() -> None:
    assert phase_outcome_label("prevention", evacuated_safely=True) == "HIGH"
    assert phase_outcome_label("prevention", evacuated_safely=False) == "HIGH"


def test_phase_outcome_intervention_is_moderate() -> None:
    assert phase_outcome_label("intervention", evacuated_safely=True) == "MODERATE"


def test_phase_outcome_evacuation_depends_on_safety() -> None:
    assert phase_outcome_label("evacuation", evacuated_safely=True) == "MODERATE"
    assert phase_outcome_label("evacuation", evacuated_safely=False) == "LOW"


def test_phase_outcome_unrecognized_phase_is_low() -> None:
    assert phase_outcome_label("unknown", evacuated_safely=True) == "LOW"


# ---------------------------------------------------------------------------
# resolve_label — precedence + NaN-safety
# ---------------------------------------------------------------------------


def test_resolve_label_expert_wins_over_everything() -> None:
    assert resolve_label("HIGH", rule_score=10.0, rule_label="LOW") == ("HIGH", "expert")


def test_resolve_label_rule_label_wins_over_rule_score() -> None:
    assert resolve_label(None, rule_score=10.0, rule_label="MODERATE") == ("MODERATE", "rule")


def test_resolve_label_falls_back_to_rule_score() -> None:
    assert resolve_label(None, rule_score=80.0, rule_label=None) == ("HIGH", "rule")


def test_resolve_label_nothing_available_returns_none_pair() -> None:
    assert resolve_label(None, None, None) == (None, None)


@pytest.mark.parametrize("missing_expert", [None, np.nan, float("nan"), ""])
def test_resolve_label_treats_missing_expert_as_absent(missing_expert) -> None:
    # Regression test: NaN is truthy in Python — `if expert_label:` alone would
    # wrongly treat a NaN read back from a CSV round-trip as a real override.
    assert resolve_label(missing_expert, rule_score=80.0, rule_label=None) == ("HIGH", "rule")


def test_resolve_label_treats_whitespace_only_expert_as_absent() -> None:
    assert resolve_label("   ", rule_score=80.0, rule_label=None) == ("HIGH", "rule")


# ---------------------------------------------------------------------------
# attach_labels — vectorized DataFrame version
# ---------------------------------------------------------------------------


def test_attach_labels_resolves_per_row() -> None:
    df = pd.DataFrame(
        {
            "simulation_score": [80, 30],
            "prep_level": [None, "LOW"],
            "expert_label": ["MODERATE", None],
        }
    )
    out = attach_labels(df)
    assert list(out["preparedness_level"]) == ["MODERATE", "LOW"]
    assert list(out["label_source"]) == ["expert", "rule"]


def test_attach_labels_missing_columns_default_to_none() -> None:
    df = pd.DataFrame({"session_id": ["s1", "s2"]})
    out = attach_labels(df)
    assert out["preparedness_level"].isna().all()
    assert out["label_source"].isna().all()


def test_attach_labels_does_not_mutate_input() -> None:
    df = pd.DataFrame({"simulation_score": [80]})
    attach_labels(df)
    assert "preparedness_level" not in df.columns


# ---------------------------------------------------------------------------
# rule_expert_agreement
# ---------------------------------------------------------------------------


def test_rule_expert_agreement_perfect_agreement() -> None:
    df = pd.DataFrame({"rule_label": ["HIGH", "LOW", "MODERATE"], "expert_label": ["HIGH", "LOW", "MODERATE"]})
    result = rule_expert_agreement(df)
    assert result["kappa"] == pytest.approx(1.0)
    assert result["agreement_rate"] == pytest.approx(1.0)
    assert result["n"] == 3


def test_rule_expert_agreement_partial_agreement() -> None:
    df = pd.DataFrame({"rule_label": ["HIGH", "LOW", "MODERATE"], "expert_label": ["HIGH", "LOW", "LOW"]})
    result = rule_expert_agreement(df)
    assert result["agreement_rate"] == pytest.approx(2 / 3)
    assert result["n"] == 3


def test_rule_expert_agreement_drops_unpaired_rows() -> None:
    df = pd.DataFrame(
        {
            "rule_label": ["HIGH", "LOW", None],
            "expert_label": ["HIGH", None, "LOW"],
        }
    )
    result = rule_expert_agreement(df)
    assert result["n"] == 1  # only the first row has both labels


def test_rule_expert_agreement_raises_on_no_overlap() -> None:
    df = pd.DataFrame({"rule_label": ["HIGH", None], "expert_label": [None, "LOW"]})
    with pytest.raises(ValueError, match="overlapping"):
        rule_expert_agreement(df)
