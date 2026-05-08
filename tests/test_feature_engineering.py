"""Tests for midrr_classifier.feature_engineering.

These tests use a small synthetic DataFrame that satisfies the raw log
schema, so they run without any real gameplay data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from midrr_classifier.data_schema import FEATURE_SCHEMA
from midrr_classifier.feature_engineering import (
    build_feature_table,
    compute_decision_delay,
    compute_evacuation_time,
    compute_hazard_avoidance_ratio,
    compute_interaction_frequency,
    compute_panic_proxy,
    compute_path_efficiency,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def dummy_events() -> pd.DataFrame:
    """Ten synthetic event rows for a single player × fire run."""
    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame(
        {
            "player_id": ["p001"] * n,
            "scenario_type": ["fire"] * n,
            "timestamp": np.linspace(0.0, 60.0, n),
            "x": rng.uniform(0, 100, n),
            "y": [64.0] * n,
            "z": rng.uniform(0, 100, n),
            "event_type": [
                "move", "move", "hazard_proximity", "move",
                "door_open", "move", "move", "extinguisher_use",
                "move", "emergency_exit",
            ],
            "hazard_distance": [10.0, 8.0, 3.0, 4.0, 6.0, 7.0, 9.0, 2.0, 11.0, 15.0],
            "preparedness_level": ["High"] * n,
        }
    )


@pytest.fixture()
def multi_player_events(dummy_events: pd.DataFrame) -> pd.DataFrame:
    """Two players (p001 fire, p002 earthquake) for group-by tests."""
    p2 = dummy_events.copy()
    p2["player_id"] = "p002"
    p2["scenario_type"] = "earthquake"
    p2["preparedness_level"] = "Low"
    return pd.concat([dummy_events, p2], ignore_index=True)


# ---------------------------------------------------------------------------
# Per-feature function tests
# ---------------------------------------------------------------------------


def test_compute_evacuation_time_positive(dummy_events: pd.DataFrame) -> None:
    result = compute_evacuation_time(dummy_events)
    assert result >= 0.0, "Evacuation time must be non-negative."


def test_compute_evacuation_time_empty() -> None:
    assert compute_evacuation_time(pd.DataFrame()) == 0.0


def test_compute_decision_delay_non_negative(dummy_events: pd.DataFrame) -> None:
    result = compute_decision_delay(dummy_events)
    assert result >= 0.0


def test_compute_path_efficiency_in_range(dummy_events: pd.DataFrame) -> None:
    result = compute_path_efficiency(dummy_events)
    assert 0.0 < result <= 1.0, f"Path efficiency {result} out of (0, 1]."


def test_compute_hazard_avoidance_in_range(dummy_events: pd.DataFrame) -> None:
    result = compute_hazard_avoidance_ratio(dummy_events)
    assert 0.0 <= result <= 1.0


def test_compute_interaction_frequency_non_negative(dummy_events: pd.DataFrame) -> None:
    result = compute_interaction_frequency(dummy_events)
    assert result >= 0.0


def test_compute_panic_proxy_non_negative(dummy_events: pd.DataFrame) -> None:
    result = compute_panic_proxy(dummy_events)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# build_feature_table tests
# ---------------------------------------------------------------------------


def test_build_feature_table_row_count(multi_player_events: pd.DataFrame) -> None:
    feature_df = build_feature_table(multi_player_events)
    # Two distinct (player_id, scenario_type) pairs → 2 rows
    assert len(feature_df) == 2


def test_build_feature_table_columns(multi_player_events: pd.DataFrame) -> None:
    feature_df = build_feature_table(multi_player_events)
    expected_cols = set(FEATURE_SCHEMA.keys())
    missing = expected_cols - set(feature_df.columns)
    assert not missing, f"Feature table is missing columns: {missing}"


def test_build_feature_table_labels_preserved(multi_player_events: pd.DataFrame) -> None:
    feature_df = build_feature_table(multi_player_events)
    labels = set(feature_df["preparedness_level"].unique())
    assert labels == {"High", "Low"}


def test_build_feature_table_no_nulls(multi_player_events: pd.DataFrame) -> None:
    feature_df = build_feature_table(multi_player_events)
    numeric_cols = [
        "evacuation_time", "decision_delay", "path_efficiency_ratio",
        "hazard_avoidance_ratio", "interaction_frequency", "panic_proxy",
    ]
    assert not feature_df[numeric_cols].isnull().any().any(), (
        "Feature table must not contain NaN values."
    )
