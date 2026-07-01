"""POST /session/{session_id}/events — mid-session streaming prediction.

Real-time counterpart to the batch ``/predict`` endpoint
(``docs/telemetry_contract.md`` §6). The mod POSTs accumulated events every
~5 seconds; this route feeds them into :class:`StreamingPredictor`, which
maintains a per-session event buffer, recomputes the 9 features on the
events seen so far, and returns a live snapshot the dashboard can render.

This was previously implemented (``streaming.py``) but never mounted on the
FastAPI app — closing that telemetry_contract.md §6 gap.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from midrr_classifier.streaming import StreamingPredictor

router = APIRouter()

# Singleton predictor — holds all in-progress session buffers for this
# process. Single-process assumption is documented in streaming.py (fine for
# the thesis demo; move to Redis/Postgres for a multi-worker deployment).
_predictor = StreamingPredictor()


class SessionEvent(BaseModel):
    """One raw-log row (telemetry_contract.md §3a/§4).

    Different ``event_type`` values carry different optional fields
    (``hit_fire``, ``extinguisher_class``, ``phase``, ``nearby_player_count``,
    ...) — accepted via ``extra="allow"`` rather than declared individually,
    since the streaming payload is intentionally sparse per event.
    """

    model_config = ConfigDict(extra="allow")

    timestamp: float
    event_type: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    hazard_distance: float | None = None


class SessionEventsRequest(BaseModel):
    """Body for POST /session/{session_id}/events."""

    contract_version: str = Field(default="1.2")
    session_id: str
    player_id: str
    scenario_type: str
    events: list[SessionEvent] = Field(default_factory=list)


class SessionSnapshotResponse(BaseModel):
    """Response body — mirrors ``streaming.SnapshotResult``."""

    session_id: str
    player_id: str
    scenario_type: str
    elapsed_time: float
    event_count: int
    is_complete: bool
    features: dict[str, float]
    prediction: str | None = None
    prep_score: float | None = None


def _try_load_model() -> None:
    """Best-effort model load at import time.

    A missing model is not an error here — ``StreamingPredictor`` serves
    features with ``prediction=None`` until a model is trained, matching
    telemetry_contract.md §6c ("prediction is null until a trained model is
    loaded").
    """
    from midrr_classifier.config import load_config

    cfg = load_config()
    try:
        _predictor.load_model(cfg.model_path)
    except FileNotFoundError:
        pass


_try_load_model()


@router.post(
    "/session/{session_id}/events",
    response_model=SessionSnapshotResponse,
    status_code=status.HTTP_200_OK,
)
def post_session_events(session_id: str, body: SessionEventsRequest) -> SessionSnapshotResponse:
    """Ingest new events for *session_id* and return a live feature/prediction snapshot."""
    if body.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path session_id '{session_id}' does not match body session_id '{body.session_id}'.",
        )

    # StreamingPredictor/normalize_raw_log expect player_id + scenario_type on
    # every event row (they're session-level in this request's JSON shape,
    # not repeated per event) — broadcast them here before ingestion.
    events: list[dict[str, Any]] = [
        {**e.model_dump(exclude_none=True), "player_id": body.player_id, "scenario_type": body.scenario_type}
        for e in body.events
    ]
    snapshot = _predictor.update(
        session_id=session_id,
        player_id=body.player_id,
        scenario_type=body.scenario_type,
        events=events,
    )

    # Contract §6d: the API closes the buffer after receiving session_end
    # (not merely once is_complete/assembly_area_reached is seen — a batch
    # may still have trailing events after the assembly area is reached).
    if any(e.event_type == "session_end" for e in body.events):
        _predictor.close_session(session_id)

    return SessionSnapshotResponse(
        session_id=snapshot.session_id,
        player_id=snapshot.player_id,
        scenario_type=snapshot.scenario_type,
        elapsed_time=snapshot.elapsed_time,
        event_count=snapshot.event_count,
        is_complete=snapshot.is_complete,
        features=snapshot.features,
        prediction=snapshot.prediction,
        prep_score=snapshot.prep_score,
    )


@router.delete("/session/{session_id}", status_code=status.HTTP_200_OK)
def delete_session(session_id: str) -> dict:
    """Explicit session cleanup (optional — the predictor auto-closes on completion)."""
    _predictor.close_session(session_id)
    return {"status": "closed", "session_id": session_id}
