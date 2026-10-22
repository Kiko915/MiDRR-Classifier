"""Tests for midrr_classifier.feature_engineering (v1.2 — 9-feature contract).

These tests use small synthetic DataFrames that satisfy the raw log
schema, so they run without any real gameplay data. Each of the 9
features is exercised for both the fire computation and its earthquake
analog (see FEATURE_DEFINITIONS in data_schema.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from midrr_classifier.data_schema import FEATURE_SCHEMA
from midrr_classifier.feature_engineering import (
    build_feature_table,
    compute_decision_latency,
    compute_evacuation_time,
    compute_hazard_avoidance_ratio,
    compute_interaction_frequency,
    compute_panic_proxy,
    compute_path_efficiency,
    compute_resource_utilization,
    compute_situational_awareness,
    compute_spray_accuracy,
)


def _row(
    player_id: str,
    scenario_type: str,
    t: float,
    event_type: str,
    x: float,
    z: float,
    hd: float,
    **extra,
) -> dict:
    row = {
        "player_id": player_id,
        "scenario_type": scenario_type,
        "timestamp": t,
        "event_type": event_type,
        "x": x,
        "y": 64.0,
        "z": z,
        "hazard_distance": hd,
        "preparedness_level": "HIGH",
    }
    row.update(extra)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fire_events() -> pd.DataFrame:
    """A fire session exercising all 9 features, including the PASS sequence."""
    rows = [
        _row("p001", "fire", 0.0, "session_start", 0.0, 0.0, 20.0),
        _row("p001", "fire", 1.0, "move", 1.0, 0.0, 15.0),
        _row("p001", "fire", 2.0, "move", 2.0, 0.0, 3.0),
        _row("p001", "fire", 3.0, "hazard_proximity", 2.0, 0.0, 3.0),
        _row("p001", "fire", 4.0, "fire_alarm_activate", 2.0, 0.0, 3.0),
        _row("p001", "fire", 4.5, "phase_transition", 2.0, 0.0, 3.0, phase="intervention"),
        _row("p001", "fire", 5.0, "pin_pull", 2.0, 0.0, 3.0, extinguisher_class="CO2", nearby_player_count=1),
        _row("p001", "fire", 5.5, "ext_spray", 2.0, 0.0, 3.0, hit_fire=1, extinguisher_class="CO2", nearby_player_count=1),
        _row("p001", "fire", 6.0, "ext_spray", 2.0, 0.0, 3.0, hit_fire=0, extinguisher_class="CO2", nearby_player_count=1),
        _row("p001", "fire", 6.5, "hazard_neutralize", 2.0, 0.0, 3.0),
        _row("p001", "fire", 7.0, "move", 3.0, 0.0, 8.0),
        _row("p001", "fire", 8.0, "emergency_exit", 4.0, 0.0, 12.0),
        _row("p001", "fire", 9.0, "move", 4.5, 0.0, 16.0),
        _row("p001", "fire", 10.0, "assembly_area_reached", 5.0, 0.0, 20.0),
        _row("p001", "fire", 10.0, "session_end", 5.0, 0.0, 20.0),
    ]
    return pd.concat(rows, ignore_index=True)


@pytest.fixture()
def earthquake_events() -> pd.DataFrame:
    """An earthquake session including a re-cover (aftershock) DCH event."""
    rows = [
        _row("p002", "earthquake", 0.0, "session_start", 0.0, 0.0, 20.0),
        _row("p002", "earthquake", 1.0, "move", 1.0, 0.0, 15.0),
        _row("p002", "earthquake", 3.0, "drop_cover_hold", 2.0, 0.0, 6.0),
        _row("p002", "earthquake", 20.0, "drop_cover_hold", 2.0, 0.0, 3.0),
        _row("p002", "earthquake", 21.0, "move", 3.0, 0.0, 10.0),
        _row("p002", "earthquake", 22.0, "emergency_exit", 4.0, 0.0, 12.0),
        _row("p002", "earthquake", 25.0, "assembly_area_reached", 5.0, 0.0, 20.0),
        _row("p002", "earthquake", 25.0, "session_end", 5.0, 0.0, 20.0),
    ]
    return pd.concat(rows, ignore_index=True)


@pytest.fixture()
def multi_player_events(fire_events: pd.DataFrame, earthquake_events: pd.DataFrame) -> pd.DataFrame:
    fire = fire_events.copy()
    quake = earthquake_events.copy()
    quake["preparedness_level"] = "LOW"
    return pd.concat([fire, quake], ignore_index=True)


# ---------------------------------------------------------------------------
# decision_latency
# ---------------------------------------------------------------------------


def test_decision_latency_fire_anchors_from_sim_start(fire_events: pd.DataFrame) -> None:
    # session_start at t=0, first valid action (fire_alarm_activate) at t=4.
    assert compute_decision_latency(fire_events) == pytest.approx(4.0)


def test_decision_latency_earthquake_uses_drop_cover_hold(earthquake_events: pd.DataFrame) -> None:
    # session_start at t=0, first drop_cover_hold at t=3.
    assert compute_decision_latency(earthquake_events) == pytest.approx(3.0)


def test_decision_latency_falls_back_to_evacuation_time_if_no_action() -> None:
    rows = pd.concat(
        [
            _row("p003", "fire", 0.0, "session_start", 0.0, 0.0, 20.0),
            _row("p003", "fire", 5.0, "session_end", 1.0, 0.0, 20.0),
        ],
        ignore_index=True,
    )
    assert compute_decision_latency(rows) == pytest.approx(compute_evacuation_time(rows))


# ---------------------------------------------------------------------------
# spray_accuracy
# ---------------------------------------------------------------------------


def test_spray_accuracy_fire_ratio(fire_events: pd.DataFrame) -> None:
    # 2 ext_spray events, 1 hit_fire=1 -> 0.5.
    assert compute_spray_accuracy(fire_events) == pytest.approx(0.5)


def test_spray_accuracy_fire_zero_when_no_sprays() -> None:
    rows = pd.concat(
        [_row("p004", "fire", 0.0, "session_start", 0.0, 0.0, 20.0)], ignore_index=True
    )
    assert compute_spray_accuracy(rows) == 0.0


def test_spray_accuracy_earthquake_uses_hazard_distance(earthquake_events: pd.DataFrame) -> None:
    # 2 drop_cover_hold events: hd=6.0 (safe) and hd=3.0 (unsafe) -> 1/2 = 0.5.
    assert compute_spray_accuracy(earthquake_events) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# resource_utilization
# ---------------------------------------------------------------------------


def test_resource_utilization_fire_correct_sequencing(fire_events: pd.DataFrame) -> None:
    # pin_pull at t=5.0 precedes first ext_spray at t=5.5 -> correct PASS technique.
    assert compute_resource_utilization(fire_events) == 1.0


def test_resource_utilization_fire_no_spray_is_vacuously_correct() -> None:
    rows = pd.concat(
        [_row("p005", "fire", 0.0, "session_start", 0.0, 0.0, 20.0)], ignore_index=True
    )
    assert compute_resource_utilization(rows) == 1.0


def test_resource_utilization_fire_sprayed_without_pin_pull_is_zero() -> None:
    rows = pd.concat(
        [
            _row("p006", "fire", 0.0, "session_start", 0.0, 0.0, 20.0),
            _row("p006", "fire", 5.0, "ext_spray", 2.0, 0.0, 3.0, hit_fire=0),
        ],
        ignore_index=True,
    )
    assert compute_resource_utilization(rows) == 0.0


def test_resource_utilization_earthquake_recover_is_full_credit(earthquake_events: pd.DataFrame) -> None:
    # 2 drop_cover_hold events -> re-covered on the later shake -> 1.0.
    assert compute_resource_utilization(earthquake_events) == 1.0


def test_resource_utilization_earthquake_single_cover_is_partial_credit() -> None:
    rows = pd.concat(
        [
            _row("p007", "earthquake", 0.0, "session_start", 0.0, 0.0, 20.0),
            _row("p007", "earthquake", 3.0, "drop_cover_hold", 2.0, 0.0, 6.0),
        ],
        ignore_index=True,
    )
    assert compute_resource_utilization(rows) == 0.5


def test_resource_utilization_earthquake_never_took_cover_is_zero() -> None:
    rows = pd.concat(
        [_row("p008", "earthquake", 0.0, "session_start", 0.0, 0.0, 20.0)], ignore_index=True
    )
    assert compute_resource_utilization(rows) == 0.0


# ---------------------------------------------------------------------------
# path_efficiency_ratio / hazard_avoidance_ratio / evacuation_time (unchanged)
# ---------------------------------------------------------------------------


def test_compute_path_efficiency_in_range(fire_events: pd.DataFrame) -> None:
    result = compute_path_efficiency(fire_events)
    assert 0.0 < result <= 1.0


def test_compute_hazard_avoidance_in_range(fire_events: pd.DataFrame) -> None:
    result = compute_hazard_avoidance_ratio(fire_events)
    assert 0.0 <= result <= 1.0


def test_compute_evacuation_time_positive(fire_events: pd.DataFrame) -> None:
    assert compute_evacuation_time(fire_events) >= 0.0


def test_compute_evacuation_time_empty() -> None:
    assert compute_evacuation_time(pd.DataFrame()) == 0.0


# ---------------------------------------------------------------------------
# interaction_frequency
# ---------------------------------------------------------------------------


def test_interaction_frequency_fire_non_negative(fire_events: pd.DataFrame) -> None:
    assert compute_interaction_frequency(fire_events) >= 0.0


def test_interaction_frequency_excludes_solo_ext_spray() -> None:
    solo = pd.concat(
        [
            _row("p009", "fire", 0.0, "session_start", 0.0, 0.0, 20.0),
            _row("p009", "fire", 2.0, "ext_spray", 0.0, 0.0, 3.0, hit_fire=1, nearby_player_count=0),
            _row("p009", "fire", 4.0, "session_end", 0.0, 0.0, 20.0),
        ],
        ignore_index=True,
    )
    accompanied = solo.copy()
    accompanied.loc[accompanied["event_type"] == "ext_spray", "nearby_player_count"] = 1

    assert compute_interaction_frequency(solo) == 0.0
    assert compute_interaction_frequency(accompanied) > 0.0


def test_interaction_frequency_earthquake_counts_drop_cover_hold(earthquake_events: pd.DataFrame) -> None:
    assert compute_interaction_frequency(earthquake_events) > 0.0


# ---------------------------------------------------------------------------
# panic_proxy (redefined v1.2: std-dev of movement speed^2)
# ---------------------------------------------------------------------------


def test_panic_proxy_exact_value_for_known_speeds() -> None:
    # Moves at t=0,1,2 with steps of 1 and 2 blocks -> speeds [1, 2] -> speed^2 [1, 4].
    # np.std([1, 4], ddof=0) == 1.5
    rows = pd.concat(
        [
            _row("p010", "fire", 0.0, "move", 0.0, 0.0, 20.0),
            _row("p010", "fire", 1.0, "move", 1.0, 0.0, 20.0),
            _row("p010", "fire", 2.0, "move", 3.0, 0.0, 20.0),
        ],
        ignore_index=True,
    )
    assert compute_panic_proxy(rows) == pytest.approx(1.5)


def test_panic_proxy_zero_with_fewer_than_three_moves() -> None:
    rows = pd.concat(
        [_row("p011", "fire", 0.0, "move", 0.0, 0.0, 20.0)], ignore_index=True
    )
    assert compute_panic_proxy(rows) == 0.0


# ---------------------------------------------------------------------------
# situational_awareness (composite)
# ---------------------------------------------------------------------------


def test_situational_awareness_fire_in_range(fire_events: pd.DataFrame) -> None:
    result = compute_situational_awareness(fire_events)
    assert 0.0 <= result <= 1.0


def test_situational_awareness_earthquake_in_range(earthquake_events: pd.DataFrame) -> None:
    result = compute_situational_awareness(earthquake_events)
    assert 0.0 <= result <= 1.0


def test_situational_awareness_empty_is_zero() -> None:
    assert compute_situational_awareness(pd.DataFrame()) == 0.0


# ---------------------------------------------------------------------------
# build_feature_table tests
# ---------------------------------------------------------------------------


def test_build_feature_table_row_count(multi_player_events: pd.DataFrame) -> None:
    feature_df = build_feature_table(multi_player_events)
    assert len(feature_df) == 2


def test_build_feature_table_columns(multi_player_events: pd.DataFrame) -> None:
    feature_df = build_feature_table(multi_player_events)
    expected_cols = set(FEATURE_SCHEMA.keys())
    missing = expected_cols - set(feature_df.columns)
    assert not missing, f"Feature table is missing columns: {missing}"


def test_build_feature_table_labels_preserved(multi_player_events: pd.DataFrame) -> None:
    feature_df = build_feature_table(multi_player_events)
    labels = set(feature_df["preparedness_level"].unique())
    assert labels == {"HIGH", "LOW"}


def test_build_feature_table_no_nulls_in_numeric_features(multi_player_events: pd.DataFrame) -> None:
    feature_df = build_feature_table(multi_player_events)
    numeric_cols = [
        "decision_latency", "spray_accuracy", "path_efficiency_ratio",
        "hazard_avoidance_ratio", "evacuation_time", "interaction_frequency",
        "resource_utilization", "panic_proxy", "situational_awareness",
    ]
    assert not feature_df[numeric_cols].isnull().any().any(), (
        "Feature table must not contain NaN values in the 9 engineered features."
    )


def test_build_feature_table_label_source_defaults_to_none(multi_player_events: pd.DataFrame) -> None:
    # No label_source column in the raw log -> resolved to None, not a crash/NaN error.
    feature_df = build_feature_table(multi_player_events)
    assert feature_df["label_source"].isna().all()


def test_build_feature_table_fire_and_earthquake_dispatch_differently(
    multi_player_events: pd.DataFrame,
) -> None:
    feature_df = build_feature_table(multi_player_events)
    fire_row = feature_df[feature_df["scenario_type"] == "fire"].iloc[0]
    quake_row = feature_df[feature_df["scenario_type"] == "earthquake"].iloc[0]
    # Both scenarios produce a value for every feature (no NaN / dispatch crash).
    for col in ("spray_accuracy", "resource_utilization"):
        assert np.isfinite(fire_row[col])
        assert np.isfinite(quake_row[col])


# ---------------------------------------------------------------------------
# Testing-instrument regressions (D-001 IN-02, D-004 DF-09, D-005 DF-01)
# ---------------------------------------------------------------------------


def _ticks(player_id: str, scenario_type: str, distances: list[float], start: float = 1.0) -> list[pd.DataFrame]:
    return [_row(player_id, scenario_type, start + i, "move", float(i), 0.0, d) for i, d in enumerate(distances)]


def test_hazard_avoidance_counts_only_move_ticks() -> None:
    # 7 safe + 3 unsafe ticks; the session_start/assembly event rows are not timesteps.
    rows = [_row("p1", "fire", 0.0, "session_start", 0.0, 0.0, 10.0)]
    rows += _ticks("p1", "fire", [10.0] * 7 + [2.0] * 3)
    rows += [_row("p1", "fire", 11.0, "assembly_area_reached", 10.0, 0.0, 10.0)]
    assert compute_hazard_avoidance_ratio(pd.concat(rows, ignore_index=True)) == pytest.approx(0.7)


def test_hazard_avoidance_ignores_post_assembly_ticks() -> None:
    rows = [_row("p1", "fire", 0.0, "session_start", 0.0, 0.0, 10.0)]
    rows += _ticks("p1", "fire", [10.0] * 7 + [2.0] * 3)
    rows += [_row("p1", "fire", 11.0, "assembly_area_reached", 10.0, 0.0, 10.0)]
    rows += _ticks("p1", "fire", [1.0] * 10, start=12.0)  # wandering back toward the fire
    assert compute_hazard_avoidance_ratio(pd.concat(rows, ignore_index=True)) == pytest.approx(0.7)


def test_hazard_avoidance_excludes_blank_distances() -> None:
    rows = [_row("p1", "fire", 0.0, "session_start", 0.0, 0.0, 10.0)]
    rows += _ticks("p1", "fire", [10.0] * 7 + [2.0] * 3 + [np.nan] * 4)
    df = pd.concat(rows, ignore_index=True)
    assert compute_hazard_avoidance_ratio(df) == pytest.approx(0.7)


def test_build_feature_table_one_row_per_session() -> None:
    # Same student, same scenario, two attempts -> two rows, not one merged row.
    first = [_row("p1", "fire", 0.0, "session_start", 0.0, 0.0, 10.0, session_id="S-A"),
             _row("p1", "fire", 1.0, "door_open", 0.0, 0.0, 10.0, session_id="S-A"),
             _row("p1", "fire", 20.0, "assembly_area_reached", 5.0, 0.0, 10.0, session_id="S-A")]
    retry = [_row("p1", "fire", 0.0, "session_start", 0.0, 0.0, 10.0, session_id="S-B"),
             _row("p1", "fire", 9.0, "door_open", 0.0, 0.0, 10.0, session_id="S-B"),
             _row("p1", "fire", 90.0, "assembly_area_reached", 5.0, 0.0, 10.0, session_id="S-B")]
    feature_df = build_feature_table(pd.concat(first + retry, ignore_index=True))
    assert len(feature_df) == 2
    by_session = feature_df.set_index("session_id")
    assert by_session.loc["S-A", "evacuation_time"] == 20.0
    assert by_session.loc["S-B", "evacuation_time"] == 90.0
    assert by_session.loc["S-B", "decision_latency"] == 9.0


def test_build_feature_table_without_session_id_keeps_legacy_grouping(multi_player_events: pd.DataFrame) -> None:
    feature_df = build_feature_table(multi_player_events)
    assert feature_df["session_id"].isna().all()


def test_build_feature_table_skips_session_with_blank_required_value() -> None:
    good = [_row("p1", "fire", 0.0, "session_start", 0.0, 0.0, 10.0, session_id="ok"),
            _row("p1", "fire", 5.0, "assembly_area_reached", 5.0, 0.0, 10.0, session_id="ok")]
    bad = [_row("p2", "fire", 0.0, "session_start", 0.0, 0.0, 10.0, session_id="broken"),
           _row("p2", "fire", np.nan, "assembly_area_reached", 5.0, 0.0, 10.0, session_id="broken")]
    feature_df = build_feature_table(pd.concat(good + bad, ignore_index=True))
    assert list(feature_df["session_id"]) == ["ok"]
    assert feature_df.attrs["incomplete_sessions"] == ["broken"]


# ---------------------------------------------------------------------------
# Testing-instrument Minor regressions (D-006 DF-11, D-007 DF-10, D-009 DF-03)
# ---------------------------------------------------------------------------


def test_earthquake_interaction_frequency_counts_emergency_exit() -> None:
    quake = pd.concat([
        _row("p", "earthquake", 0.0, "session_start", 0.0, 0.0, 10.0),
        _row("p", "earthquake", 2.0, "drop_cover_hold", 0.0, 0.0, 10.0),
        _row("p", "earthquake", 8.0, "emergency_exit", 0.0, 0.0, 10.0),
        _row("p", "earthquake", 10.0, "assembly_area_reached", 0.0, 0.0, 10.0),
    ], ignore_index=True)
    # drop_cover_hold + emergency_exit + assembly = 3 events over 10 s.
    assert compute_interaction_frequency(quake) == pytest.approx(0.3)


def test_earthquake_interaction_frequency_ignores_fire_only_events() -> None:
    quake = pd.concat([
        _row("p", "earthquake", 0.0, "session_start", 0.0, 0.0, 10.0),
        _row("p", "earthquake", 1.0, "door_open", 0.0, 0.0, 10.0),
        _row("p", "earthquake", 2.0, "fire_alarm_activate", 0.0, 0.0, 10.0),
        _row("p", "earthquake", 10.0, "assembly_area_reached", 0.0, 0.0, 10.0),
    ], ignore_index=True)
    assert compute_interaction_frequency(quake) == pytest.approx(0.1)  # only the assembly event


def test_fire_interaction_frequency_unchanged_by_earthquake_set() -> None:
    fire = pd.concat([
        _row("p", "fire", 0.0, "session_start", 0.0, 0.0, 10.0),
        _row("p", "fire", 1.0, "emergency_exit", 0.0, 0.0, 10.0),  # not a fire interaction
        _row("p", "fire", 2.0, "door_open", 0.0, 0.0, 10.0),
        _row("p", "fire", 10.0, "assembly_area_reached", 0.0, 0.0, 10.0),
    ], ignore_index=True)
    assert compute_interaction_frequency(fire) == pytest.approx(0.2)


def test_evacuation_time_subtracts_sim_start() -> None:
    late_clock = pd.concat([
        _row("p", "fire", 5.0, "session_start", 0.0, 0.0, 10.0),
        _row("p", "fire", 45.0, "assembly_area_reached", 0.0, 0.0, 10.0),
    ], ignore_index=True)
    assert compute_evacuation_time(late_clock) == pytest.approx(40.0)


def test_evacuation_time_uses_earliest_event_when_no_session_start() -> None:
    rows = pd.concat([
        _row("p", "fire", 2.0, "move", 0.0, 0.0, 10.0),
        _row("p", "fire", 12.0, "assembly_area_reached", 0.0, 0.0, 10.0),
    ], ignore_index=True)
    assert compute_evacuation_time(rows) == pytest.approx(10.0)


def test_build_feature_table_skips_session_with_negative_duration() -> None:
    good = [_row("p1", "fire", 0.0, "session_start", 0.0, 0.0, 10.0, session_id="ok"),
            _row("p1", "fire", 5.0, "assembly_area_reached", 5.0, 0.0, 10.0, session_id="ok")]
    # Assembly logged BEFORE the session started -> negative evacuation_time.
    bad = [_row("p2", "fire", 0.0, "session_start", 0.0, 0.0, 10.0, session_id="neg"),
           _row("p2", "fire", -30.0, "assembly_area_reached", 5.0, 0.0, 10.0, session_id="neg")]
    feature_df = build_feature_table(pd.concat(good + bad, ignore_index=True))
    assert list(feature_df["session_id"]) == ["ok"]
    assert feature_df.attrs["invalid_sessions"] == ["neg"]
    assert (feature_df["evacuation_time"] >= 0).all()
