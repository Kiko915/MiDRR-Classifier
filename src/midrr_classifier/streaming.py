"""Mid-session (real-time) prediction support.

The Minecraft mod POSTs accumulated events to the API every few seconds
at 20 Hz. :class:`StreamingPredictor` maintains a per-session event buffer,
recomputes all six features on each update, and returns a
:class:`SnapshotResult` the dashboard can display live.

Architecture
------------
Mod (20 Hz) --> POST /session/{id}/events (every ~5 s)
                         |
                  StreamingPredictor.update()
                         |
                  SessionBuffer.ingest()   <- appends & normalizes
                         |
                  compute_*() on full buffer so far
                         |
                  SnapshotResult  <-- prediction + features + elapsed time

The classifier is optional: if no trained model has been loaded yet,
``SnapshotResult.prediction`` is ``None`` and only features are returned.
This lets the pipeline be exercised before the model is trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from midrr_classifier.data_ingestion import normalize_raw_log
from midrr_classifier.feature_engineering import (
    compute_decision_delay,
    compute_evacuation_time,
    compute_hazard_avoidance_ratio,
    compute_interaction_frequency,
    compute_panic_proxy,
    compute_path_efficiency,
)
from midrr_classifier.utils.logging_utils import get_logger

logger = get_logger(__name__)

_FEATURE_COLS = [
    "evacuation_time",
    "decision_delay",
    "path_efficiency_ratio",
    "hazard_avoidance_ratio",
    "interaction_frequency",
    "panic_proxy",
]


@dataclass
class SnapshotResult:
    """Feature + prediction snapshot returned after each event batch.

    Returned by :meth:`StreamingPredictor.update` on every POST from the mod.
    The FastAPI layer serialises this directly to JSON for the dashboard.
    """

    session_id: str
    player_id: str
    scenario_type: str
    elapsed_time: float
    event_count: int
    is_complete: bool
    features: dict[str, float]
    prediction: str | None = None  # "HIGH" / "MODERATE" / "LOW", or None if no model loaded
    prep_score: float | None = None  # winning class probability scaled to 0-100


class SessionBuffer:
    """Accumulates and normalises events for a single in-progress session.

    Intentionally thin — all feature logic stays in ``feature_engineering.py``.
    """

    def __init__(self, session_id: str, player_id: str, scenario_type: str) -> None:
        self.session_id = session_id
        self.player_id = player_id
        self.scenario_type = scenario_type
        self._rows: list[dict[str, Any]] = []

    def ingest(self, events: list[dict[str, Any]]) -> pd.DataFrame:
        """Append *events* to the buffer and return the full normalised DataFrame."""
        self._rows.extend(events)
        if not self._rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._rows)
        return normalize_raw_log(df)

    @property
    def event_count(self) -> int:
        return len(self._rows)

    def is_complete(self, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        return "assembly_area_reached" in df["event_type"].values


class StreamingPredictor:
    """Manages in-progress sessions and returns live prediction snapshots.

    Instantiate once per process (e.g. at FastAPI startup). Thread-safety
    note: for a multi-worker deployment, move session state to Redis or
    Postgres; in-memory is fine for a single-process thesis demo.

    Example::

        predictor = StreamingPredictor()
        predictor.load_model("models/midrr_rf.pkl")

        snapshot = predictor.update(
            session_id="sess_001",
            player_id="stu_42",
            scenario_type="fire",
            events=[{"timestamp": 1.0, "event_type": "move", ...}, ...],
        )
        print(snapshot.prediction)   # "HIGH" / "MODERATE" / "LOW" / None
        print(snapshot.elapsed_time) # seconds elapsed so far
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionBuffer] = {}
        self._clf = None
        self._cfg = None

    def load_model(self, model_path: str, config_path: str | None = None) -> None:
        """Load a trained model so :meth:`update` can return live predictions."""
        from midrr_classifier.config import load_config
        from midrr_classifier.model_definition import MiDRRClassifier

        self._cfg = load_config(config_path)
        clf = MiDRRClassifier(self._cfg)
        clf.load(model_path)
        self._clf = clf
        logger.info("StreamingPredictor: model loaded from %s", model_path)

    def update(
        self,
        session_id: str,
        player_id: str,
        scenario_type: str,
        events: list[dict[str, Any]],
    ) -> SnapshotResult:
        """Ingest new events and return a refreshed feature + prediction snapshot.

        Args:
            session_id: Unique run identifier (from mod).
            player_id: Stable student UUID.
            scenario_type: ``"fire"`` or ``"earthquake"`` (canonical or mod alias).
            events: New events since the last POST. May be empty (heartbeat).

        Returns:
            :class:`SnapshotResult` with current features and, if a model is
            loaded, a live ``prediction`` and ``prep_score``.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionBuffer(session_id, player_id, scenario_type)

        buf = self._sessions[session_id]
        df = buf.ingest(events)

        if df.empty:
            features = {col: 0.0 for col in _FEATURE_COLS}
            elapsed = 0.0
            complete = False
        else:
            features = {
                "evacuation_time": compute_evacuation_time(df),
                "decision_delay": compute_decision_delay(df),
                "path_efficiency_ratio": compute_path_efficiency(df),
                "hazard_avoidance_ratio": compute_hazard_avoidance_ratio(df),
                "interaction_frequency": compute_interaction_frequency(df),
                "panic_proxy": compute_panic_proxy(df),
            }
            elapsed = float(df["timestamp"].max())
            complete = buf.is_complete(df)

        prediction = None
        prep_score = None

        if self._clf is not None and self._cfg is not None:
            vec = np.array([[features[c] for c in self._cfg.feature_cols]])
            prediction = self._clf.predict(vec)[0]
            proba = self._clf.predict_proba(vec)[0]
            pred_idx = list(self._clf.model.classes_).index(prediction)
            prep_score = round(float(proba[pred_idx]) * 100, 1)

        logger.debug(
            "StreamingPredictor update: session=%s elapsed=%.1fs events=%d prediction=%s",
            session_id, elapsed, buf.event_count, prediction,
        )

        return SnapshotResult(
            session_id=session_id,
            player_id=player_id,
            scenario_type=scenario_type,
            elapsed_time=elapsed,
            event_count=buf.event_count,
            is_complete=complete,
            features=features,
            prediction=prediction,
            prep_score=prep_score,
        )

    def close_session(self, session_id: str) -> None:
        """Drop a session's buffer once ``session_end`` is received."""
        self._sessions.pop(session_id, None)
        logger.info("StreamingPredictor: session closed (%s)", session_id)

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)
