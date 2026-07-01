"""Tests for midrr_classifier.synth (Phase 2.5 step 5 — 9-feature synth data).

Covers the locked 250-session / 35-45-20 class distribution, the legacy
uniform n_per_class= mode, the new event vocabulary, and that the output
feeds build_feature_table() cleanly with no NaN across all 9 features.
"""

from __future__ import annotations

import pandas as pd
import pytest

from midrr_classifier.data_schema import (
    DROP_COVER_HOLD_EVENT,
    EXT_SPRAY_EVENT,
    HAZARD_NEUTRALIZE_EVENT,
    PHASE_TRANSITION_EVENT,
    PIN_PULL_EVENT,
)
from midrr_classifier.feature_engineering import build_feature_table
from midrr_classifier.synth import _allocate_class_counts, generate_logs


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------


def test_allocate_class_counts_sums_exactly_to_total() -> None:
    counts = _allocate_class_counts(250, {"HIGH": 0.35, "MODERATE": 0.45, "LOW": 0.20})
    assert sum(counts.values()) == 250


def test_allocate_class_counts_locked_split() -> None:
    counts = _allocate_class_counts(250, {"HIGH": 0.35, "MODERATE": 0.45, "LOW": 0.20})
    assert counts == {"HIGH": 88, "MODERATE": 112, "LOW": 50}


def test_generate_logs_default_class_distribution() -> None:
    df = generate_logs(seed=1)
    sessions = df.drop_duplicates("session_id")
    assert len(sessions) == 250
    counts = sessions["preparedness_level"].value_counts().to_dict()
    assert counts == {"MODERATE": 112, "HIGH": 88, "LOW": 50}


def test_generate_logs_default_splits_evenly_across_scenarios() -> None:
    df = generate_logs(seed=1)
    sessions = df.drop_duplicates("session_id")
    counts = sessions["scenario_type"].value_counts().to_dict()
    assert counts == {"fire": 125, "earthquake": 125}


def test_generate_logs_legacy_n_per_class_is_uniform() -> None:
    df = generate_logs(n_per_class=4, seed=42)
    sessions = df.drop_duplicates("session_id")
    counts = sessions.groupby(["scenario_type", "preparedness_level"]).size()
    assert (counts == 4).all()
    assert len(sessions) == 4 * 3 * 2  # 4 per class * 3 classes * 2 scenarios


# ---------------------------------------------------------------------------
# Event vocabulary (v1.2)
# ---------------------------------------------------------------------------


def test_fire_sessions_emit_pass_technique_events() -> None:
    df = generate_logs(n_per_class=15, seed=3, scenario_types=("fire",))
    event_types = set(df["event_type"])
    assert PIN_PULL_EVENT in event_types
    assert EXT_SPRAY_EVENT in event_types
    assert HAZARD_NEUTRALIZE_EVENT in event_types
    assert PHASE_TRANSITION_EVENT in event_types


def test_earthquake_sessions_emit_drop_cover_hold() -> None:
    df = generate_logs(n_per_class=15, seed=3, scenario_types=("earthquake",))
    assert DROP_COVER_HOLD_EVENT in set(df["event_type"])


def test_data_is_tagged_as_synthetic() -> None:
    df = generate_logs(n_per_class=2, seed=0)
    assert (df["data_source"] == "synthetic").all()


# ---------------------------------------------------------------------------
# Feeds build_feature_table cleanly
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_feature_table() -> pd.DataFrame:
    raw = generate_logs(seed=7)
    return build_feature_table(raw)


def test_synthetic_pipeline_no_nan(synthetic_feature_table: pd.DataFrame) -> None:
    numeric_cols = [
        "decision_latency", "spray_accuracy", "path_efficiency_ratio",
        "hazard_avoidance_ratio", "evacuation_time", "interaction_frequency",
        "resource_utilization", "panic_proxy", "situational_awareness",
    ]
    assert not synthetic_feature_table[numeric_cols].isnull().any().any()


def test_synthetic_pipeline_row_count_matches_sessions(synthetic_feature_table: pd.DataFrame) -> None:
    assert len(synthetic_feature_table) == 250


def test_synthetic_panic_proxy_separates_skill_levels(synthetic_feature_table: pd.DataFrame) -> None:
    # Regression guard for the speed-jitter fix: HIGH must show LESS panic
    # than LOW (the redefined speed^2-std-dev feature had been inverted).
    means = synthetic_feature_table.groupby("preparedness_level")["panic_proxy"].mean()
    assert means["HIGH"] < means["MODERATE"] < means["LOW"]


def test_synthetic_spray_accuracy_separates_skill_levels(synthetic_feature_table: pd.DataFrame) -> None:
    fire_rows = synthetic_feature_table[synthetic_feature_table["scenario_type"] == "fire"]
    means = fire_rows.groupby("preparedness_level")["spray_accuracy"].mean()
    assert means["HIGH"] > means["MODERATE"] > means["LOW"]
