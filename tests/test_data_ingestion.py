"""Tests for midrr_classifier.data_ingestion.split_train_test.

Focus: the group-aware guarantee — no player_id appears in both train
and test, regardless of how many sessions that player has.
"""

from __future__ import annotations

import pandas as pd
import pytest

from midrr_classifier.data_ingestion import split_train_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_feature_table(
    n_players: int = 30,
    scenarios: tuple[str, ...] = ("fire", "earthquake"),
    labels: tuple[str, ...] = ("HIGH", "MODERATE", "LOW"),
) -> pd.DataFrame:
    """Build a synthetic feature table with one row per player × scenario."""
    rows = []
    for i in range(n_players):
        label = labels[i % len(labels)]
        for scenario in scenarios:
            rows.append(
                {
                    "player_id": f"p{i:03d}",
                    "scenario_type": scenario,
                    "evacuation_time": float(10 + i),
                    "decision_delay": float(1 + i * 0.1),
                    "path_efficiency_ratio": 0.8,
                    "hazard_avoidance_ratio": 0.9,
                    "interaction_frequency": 0.1,
                    "panic_proxy": 20.0,
                    "preparedness_level": label,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def feature_table() -> pd.DataFrame:
    return _make_feature_table(n_players=30)


@pytest.fixture()
def single_session_table() -> pd.DataFrame:
    """Each player has only one session (fire only)."""
    return _make_feature_table(n_players=30, scenarios=("fire",))


# ---------------------------------------------------------------------------
# Core guarantee: no player_id leaks across splits
# ---------------------------------------------------------------------------


def test_no_player_id_leakage(feature_table: pd.DataFrame) -> None:
    train, test = split_train_test(feature_table, test_size=0.3, random_state=42)
    train_ids = set(train["player_id"])
    test_ids = set(test["player_id"])
    assert train_ids.isdisjoint(test_ids), (
        f"player_id leakage detected — {train_ids & test_ids} appear in both splits."
    )


def test_no_leakage_multi_session(feature_table: pd.DataFrame) -> None:
    """Same guarantee when players have multiple sessions (fire + earthquake)."""
    train, test = split_train_test(feature_table, test_size=0.3, random_state=0)
    assert set(train["player_id"]).isdisjoint(set(test["player_id"]))


def test_all_rows_accounted_for(feature_table: pd.DataFrame) -> None:
    train, test = split_train_test(feature_table, test_size=0.3, random_state=42)
    assert len(train) + len(test) == len(feature_table)


# ---------------------------------------------------------------------------
# Size and stratification
# ---------------------------------------------------------------------------


def test_split_size_approximate(feature_table: pd.DataFrame) -> None:
    """Test split ≈ 70/30 at the player level (±1 player rounding)."""
    train, test = split_train_test(feature_table, test_size=0.3, random_state=42)
    n_players = feature_table["player_id"].nunique()
    n_test_players = test["player_id"].nunique()
    expected = round(n_players * 0.3)
    assert abs(n_test_players - expected) <= 1


def test_class_balance_preserved(feature_table: pd.DataFrame) -> None:
    """Each class should appear in both train and test."""
    train, test = split_train_test(feature_table, test_size=0.3, random_state=42)
    for label in ("HIGH", "MODERATE", "LOW"):
        assert label in train["preparedness_level"].values, f"{label} missing from train"
        assert label in test["preparedness_level"].values, f"{label} missing from test"


def test_single_session_per_player(single_session_table: pd.DataFrame) -> None:
    """Group-aware split degrades gracefully to row-level split when each player has one session."""
    train, test = split_train_test(single_session_table, test_size=0.3, random_state=42)
    assert set(train["player_id"]).isdisjoint(set(test["player_id"]))
    assert len(train) + len(test) == len(single_session_table)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_stratify_col_raises(feature_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="preparedness_level"):
        split_train_test(feature_table.drop(columns=["preparedness_level"]))


def test_missing_group_col_raises(feature_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="player_id"):
        split_train_test(feature_table.drop(columns=["player_id"]))


def test_custom_group_col(feature_table: pd.DataFrame) -> None:
    """group_col can be any column, not just player_id."""
    df = feature_table.rename(columns={"player_id": "student_uuid"})
    train, test = split_train_test(
        df, test_size=0.3, group_col="student_uuid", random_state=42
    )
    assert set(train["student_uuid"]).isdisjoint(set(test["student_uuid"]))
