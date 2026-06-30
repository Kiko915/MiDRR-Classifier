"""Tests for the mid-session streaming prediction module."""

from __future__ import annotations

import pytest

from midrr_classifier.streaming import SessionBuffer, SnapshotResult, StreamingPredictor

# ---------------------------------------------------------------------------
# Minimal event helpers
# ---------------------------------------------------------------------------

def _move(t: float, x: float, z: float, hd: float = 10.0) -> dict:
    return {
        "player_id": "stu_01",
        "session_id": "sess_test",
        "scenario_type": "fire",
        "timestamp": t,
        "event_type": "move",
        "x": x, "y": -32.0, "z": z,
        "hazard_distance": hd,
        "interaction_target": None,
        "nearby_player_count": None,
    }


def _event(t: float, etype: str, hd: float = 10.0, nearby: int | None = None) -> dict:
    return {
        "player_id": "stu_01",
        "session_id": "sess_test",
        "scenario_type": "fire",
        "timestamp": t,
        "event_type": etype,
        "x": 100.0, "y": -32.0, "z": 100.0,
        "hazard_distance": hd,
        "interaction_target": None,
        "nearby_player_count": nearby,
    }


# ---------------------------------------------------------------------------
# SessionBuffer
# ---------------------------------------------------------------------------

class TestSessionBuffer:
    def test_ingest_normalizes_move_tick(self):
        buf = SessionBuffer("s1", "p1", "fire")
        ev = _move(1.0, 0.0, 0.0)
        ev["event_type"] = "move_tick"
        df = buf.ingest([ev])
        assert "move" in df["event_type"].values

    def test_ingest_preserves_ccs_fire(self):
        # ccs_fire is a distinct canonical scenario type — must NOT be collapsed to fire.
        buf = SessionBuffer("s1", "p1", "ccs_fire")
        ev = _move(1.0, 0.0, 0.0)
        ev["scenario_type"] = "ccs_fire"
        df = buf.ingest([ev])
        assert "ccs_fire" in df["scenario_type"].values

    def test_event_count_accumulates(self):
        buf = SessionBuffer("s1", "p1", "fire")
        buf.ingest([_move(1.0, 0.0, 0.0), _move(2.0, 1.0, 0.0)])
        buf.ingest([_move(3.0, 2.0, 0.0)])
        assert buf.event_count == 3

    def test_is_complete_false_without_assembly(self):
        buf = SessionBuffer("s1", "p1", "fire")
        df = buf.ingest([_move(1.0, 0.0, 0.0)])
        assert not buf.is_complete(df)

    def test_is_complete_true_after_assembly(self):
        buf = SessionBuffer("s1", "p1", "fire")
        df = buf.ingest([_move(1.0, 0.0, 0.0), _event(5.0, "assembly_area_reached")])
        assert buf.is_complete(df)

    def test_empty_ingest_returns_empty_df(self):
        buf = SessionBuffer("s1", "p1", "fire")
        df = buf.ingest([])
        assert df.empty


# ---------------------------------------------------------------------------
# StreamingPredictor (no model — features only)
# ---------------------------------------------------------------------------

class TestStreamingPredictorNoModel:
    def setup_method(self):
        self.predictor = StreamingPredictor()

    def _basic_events(self):
        return [
            _event(0.0, "session_start"),
            _move(0.1, 100.0, 100.0, hd=15.0),
            _move(0.2, 100.5, 100.0, hd=14.0),
            _move(0.3, 101.0, 100.0, hd=13.0),
            _move(0.4, 101.5, 100.0, hd=4.0),  # inside hazard zone
            _event(0.5, "fire_alarm_activate", hd=4.0),
            _move(0.6, 102.0, 101.0, hd=6.0),
            _move(0.7, 103.0, 102.0, hd=7.0),
        ]

    def test_returns_snapshot_result(self):
        snap = self.predictor.update("s1", "p1", "fire", self._basic_events())
        assert isinstance(snap, SnapshotResult)

    def test_prediction_is_none_without_model(self):
        snap = self.predictor.update("s1", "p1", "fire", self._basic_events())
        assert snap.prediction is None
        assert snap.prep_score is None

    def test_feature_ranges_valid(self):
        snap = self.predictor.update("s1", "p1", "fire", self._basic_events())
        f = snap.features
        assert f["evacuation_time"] >= 0
        assert f["decision_delay"] >= 0
        assert 0 < f["path_efficiency_ratio"] <= 1.0
        assert 0 <= f["hazard_avoidance_ratio"] <= 1.0
        assert f["interaction_frequency"] >= 0
        assert f["panic_proxy"] >= 0

    def test_elapsed_time_matches_max_timestamp(self):
        snap = self.predictor.update("s1", "p1", "fire", self._basic_events())
        assert snap.elapsed_time == pytest.approx(0.7)

    def test_is_complete_false_mid_session(self):
        snap = self.predictor.update("s1", "p1", "fire", self._basic_events())
        assert not snap.is_complete

    def test_is_complete_true_after_assembly(self):
        events = self._basic_events() + [_event(2.0, "assembly_area_reached")]
        snap = self.predictor.update("s2", "p1", "fire", events)
        assert snap.is_complete

    def test_incremental_updates_accumulate(self):
        batch1 = self._basic_events()
        batch2 = [_move(0.8, 104.0, 103.0, hd=8.0), _move(0.9, 105.0, 104.0, hd=9.0)]

        self.predictor.update("s3", "p1", "fire", batch1)
        snap = self.predictor.update("s3", "p1", "fire", batch2)

        assert snap.event_count == len(batch1) + len(batch2)

    def test_move_tick_normalised_ccs_fire_preserved(self):
        # move_tick → move; ccs_fire stays as ccs_fire (distinct building).
        events = [_move(0.1, 100.0, 100.0)]
        events[0]["event_type"] = "move_tick"
        events[0]["scenario_type"] = "ccs_fire"
        snap = self.predictor.update("s4", "p1", "ccs_fire", events)
        assert snap.elapsed_time > 0  # pipeline didn't crash

    def test_close_session_removes_buffer(self):
        self.predictor.update("s5", "p1", "fire", self._basic_events())
        assert self.predictor.active_session_count == 1
        self.predictor.close_session("s5")
        assert self.predictor.active_session_count == 0

    def test_empty_event_list_returns_zero_features(self):
        snap = self.predictor.update("s6", "p1", "fire", [])
        assert snap.elapsed_time == 0.0
        assert all(v == 0.0 for v in snap.features.values())
