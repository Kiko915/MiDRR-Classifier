"""Tests for midrr_classifier.data_ingestion.

Covers: the group-aware split guarantee (no player_id in both train and
test), the v1.2 expert-only test-split circularity guard, and the
Turso/CSV session ingestion adapter (Phase 2.5 step 4).
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

from midrr_classifier.data_ingestion import (
    load_sessions,
    load_sessions_from_turso,
    split_train_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_feature_table(
    n_players: int = 30,
    scenarios: tuple[str, ...] = ("fire", "earthquake"),
    labels: tuple[str, ...] = ("HIGH", "MODERATE", "LOW"),
    label_source: str | None = None,
) -> pd.DataFrame:
    """Build a synthetic feature table with one row per player × scenario."""
    rows = []
    for i in range(n_players):
        label = labels[i % len(labels)]
        for scenario in scenarios:
            row = {
                "player_id": f"p{i:03d}",
                "scenario_type": scenario,
                "decision_latency": float(1 + i * 0.1),
                "spray_accuracy": 0.7,
                "path_efficiency_ratio": 0.8,
                "hazard_avoidance_ratio": 0.9,
                "evacuation_time": float(10 + i),
                "interaction_frequency": 0.1,
                "resource_utilization": 0.8,
                "panic_proxy": 20.0,
                "situational_awareness": 0.75,
                "preparedness_level": label,
            }
            if label_source is not None:
                row["label_source"] = label_source
            rows.append(row)
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


# ---------------------------------------------------------------------------
# Expert-only test-split circularity guard (Phase 2.5 step 4)
# ---------------------------------------------------------------------------


def test_expert_only_test_split_drops_rule_rows() -> None:
    """Every row in the test split must be label_source='expert' when present."""
    # Mix of expert- and rule-labeled players so some rule rows are guaranteed
    # to land in whichever split sklearn's stratified sampling picks as test.
    expert_half = _make_feature_table(n_players=15, label_source="expert")
    rule_half = _make_feature_table(n_players=15, label_source="rule")
    rule_half["player_id"] = rule_half["player_id"].apply(lambda p: f"r{p}")
    df = pd.concat([expert_half, rule_half], ignore_index=True)

    train, test = split_train_test(df, test_size=0.3, random_state=42)

    assert (test["label_source"] == "expert").all()
    # No player leaked across splits despite the post-hoc row filtering.
    assert set(train["player_id"]).isdisjoint(set(test["player_id"]))


def test_expert_only_enforcement_is_noop_without_label_source(feature_table: pd.DataFrame) -> None:
    """Backward compatibility: no label_source column -> no rows dropped."""
    train, test = split_train_test(feature_table, test_size=0.3, random_state=42)
    assert len(train) + len(test) == len(feature_table)


def test_expert_only_enforcement_is_noop_when_all_none() -> None:
    """label_source present but entirely null (e.g. synthetic data) -> no-op."""
    df = _make_feature_table(n_players=30)
    df["label_source"] = None
    train, test = split_train_test(df, test_size=0.3, random_state=42)
    assert len(train) + len(test) == len(df)


def test_expert_only_enforcement_can_be_disabled() -> None:
    df = _make_feature_table(n_players=30, label_source="rule")
    train, test = split_train_test(
        df, test_size=0.3, random_state=42, enforce_expert_only_test=False
    )
    assert len(train) + len(test) == len(df)


# ---------------------------------------------------------------------------
# Turso / CSV session ingestion adapter (Phase 2.5 step 4)
# ---------------------------------------------------------------------------


class _FakeResultSet:
    def __init__(self, rows: list[tuple], columns: list[str]) -> None:
        self.rows = rows
        self.columns = columns


class _FakeTursoClient:
    def __init__(self, rows: list[tuple], columns: list[str]) -> None:
        self._rows = rows
        self._columns = columns
        self.closed = False

    def execute(self, query: str) -> _FakeResultSet:
        return _FakeResultSet(self._rows, self._columns)

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_libsql_client(monkeypatch: pytest.MonkeyPatch):
    """Injects a fake ``libsql_client`` module into sys.modules.

    Lets ``load_sessions_from_turso``'s deferred ``import libsql_client``
    resolve without the optional dependency installed or a real network call.
    """
    columns = [
        "session_id", "student_name", "simulation_type", "event_log",
        "move_log_csv", "simulation_score", "passed", "prep_level",
        "confidence", "expert_label",
    ]
    event_log = json.dumps([
        {"timestamp": 0.0, "event_type": "session_start", "x": 0.0, "y": 64.0, "z": 0.0, "hazard_distance": 20.0},
        {"timestamp": 2.0, "event_type": "fire_alarm_activate", "x": 1.0, "y": 64.0, "z": 1.0, "hazard_distance": 15.0},
        {"timestamp": 10.0, "event_type": "assembly_area_reached", "x": 53.0, "y": 64.0, "z": 73.0, "hazard_distance": 30.0},
    ])
    move_log_csv = "timestamp,x,y,z,hazard_distance\n0.0,0.0,64.0,0.0,20.0\n1.0,5.0,64.0,5.0,18.0\n"
    rows = [("sess_1", "stu_001", "fire", event_log, move_log_csv, 82.0, True, "HIGH", 0.9, None)]

    fake_client = _FakeTursoClient(rows, columns)
    fake_module = types.SimpleNamespace(create_client_sync=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "libsql_client", fake_module)
    return fake_module, fake_client


class TestTursoIngestion:
    def test_returns_validated_raw_log(self, fake_libsql_client) -> None:
        fake_module, fake_client = fake_libsql_client
        raw_df = load_sessions_from_turso("libsql://fake-db", "fake-token")

        assert not raw_df.empty
        assert set(raw_df["player_id"]) == {"stu_001"}
        assert set(raw_df["scenario_type"]) == {"fire"}
        assert (raw_df["preparedness_level"] == "HIGH").all()

    def test_falls_back_to_rule_label_without_expert_override(self, fake_libsql_client) -> None:
        raw_df = load_sessions_from_turso("libsql://fake-db", "fake-token")
        assert (raw_df["label_source"] == "rule").all()

    def test_connects_with_given_credentials_and_closes_client(self, fake_libsql_client) -> None:
        fake_module, fake_client = fake_libsql_client
        load_sessions_from_turso("libsql://fake-db", "fake-token")
        fake_module.create_client_sync.assert_called_once_with(
            url="libsql://fake-db", auth_token="fake-token"
        )
        assert fake_client.closed

    def test_load_sessions_dispatches_to_turso(self, fake_libsql_client) -> None:
        raw_df = load_sessions("turso", database_url="libsql://fake-db", auth_token="tok")
        assert not raw_df.empty

    def test_load_sessions_turso_requires_database_url(self) -> None:
        with pytest.raises(ValueError, match="database_url"):
            load_sessions("turso")

    def test_load_sessions_unknown_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown source"):
            load_sessions("xml")

    def test_load_sessions_csv_requires_csv_path(self) -> None:
        with pytest.raises(ValueError, match="csv_path"):
            load_sessions("csv")

    def test_missing_libsql_client_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force `import libsql_client` to fail even if the optional extra
        # happens to be installed in this environment.
        monkeypatch.setitem(sys.modules, "libsql_client", None)
        with pytest.raises(ImportError, match="libsql-client"):
            load_sessions_from_turso("libsql://fake-db")
