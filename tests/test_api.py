"""Tests for the MiDRR API's 9-feature /predict and /session contracts.

Route handlers are called directly (not through fastapi.testclient.TestClient)
because the installed httpx/starlette versions in this environment are
mutually incompatible with TestClient — a pre-existing dependency mismatch
unrelated to this test suite. Calling the route functions directly still
exercises the exact request/response mapping logic FastAPI would dispatch to,
with predict_preparedness_full() stubbed out so these tests don't depend on
a trained model file existing on disk.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.routes import predict as predict_route
from api.routes import session as session_route
from api.schemas import FeaturesRequest, PredictResponse
from midrr_classifier.streaming import StreamingPredictor

_NINE_FEATURES = {
    "decision_latency": 3.0,
    "spray_accuracy": 0.8,
    "path_efficiency_ratio": 0.75,
    "hazard_avoidance_ratio": 0.9,
    "evacuation_time": 25.0,
    "interaction_frequency": 0.2,
    "resource_utilization": 1.0,
    "panic_proxy": 2.0,
    "situational_awareness": 0.8,
}


def _make_body() -> FeaturesRequest:
    return FeaturesRequest(player_id="stu_1", scenario_type="fire", **_NINE_FEATURES)


@pytest.fixture()
def stub_prediction(monkeypatch: pytest.MonkeyPatch):
    """Stub predict_preparedness_full so /predict tests don't need a real model."""

    def _fake(features_row, model_path=None, config_path=None):
        return {
            "label": "HIGH",
            "probabilities": {"HIGH": 0.8, "MODERATE": 0.15, "LOW": 0.05},
            "shap_values": {feat: 0.1 * (i + 1) for i, feat in enumerate(_NINE_FEATURES)},
            "feature_importances": {feat: 1 / 9 for feat in _NINE_FEATURES},
        }

    monkeypatch.setattr(predict_route, "predict_preparedness_full", _fake)
    return _fake


# ---------------------------------------------------------------------------
# FeaturesRequest schema
# ---------------------------------------------------------------------------


def test_features_request_accepts_all_nine_fields() -> None:
    body = _make_body()
    for feat, value in _NINE_FEATURES.items():
        assert getattr(body, feat) == value


def test_features_request_rejects_out_of_range_ratio() -> None:
    with pytest.raises(ValueError):
        FeaturesRequest(
            player_id="stu_1",
            scenario_type="fire",
            **{**_NINE_FEATURES, "spray_accuracy": 1.5},
        )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


def test_predict_returns_full_contract(stub_prediction) -> None:
    response = predict_route.predict(_make_body())
    assert isinstance(response, PredictResponse)
    assert response.prepLevel == "HIGH"
    assert response.prepScore == 80
    assert len(response.featureImportance) == 9
    assert {fw.feature for fw in response.featureImportance} == set(_NINE_FEATURES)
    assert response.resultText


def test_predict_feature_importance_sorted_by_absolute_shap(stub_prediction) -> None:
    response = predict_route.predict(_make_body())
    abs_weights = [abs(fw.weight) for fw in response.featureImportance]
    assert abs_weights == sorted(abs_weights, reverse=True)


def test_predict_missing_model_raises_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args, **kwargs):
        raise FileNotFoundError("model not trained yet")

    monkeypatch.setattr(predict_route, "predict_preparedness_full", _raise)
    with pytest.raises(HTTPException) as exc_info:
        predict_route.predict(_make_body())
    assert exc_info.value.status_code == 503


def test_predict_missing_feature_raises_422(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args, **kwargs):
        raise KeyError("decision_latency")

    monkeypatch.setattr(predict_route, "predict_preparedness_full", _raise)
    with pytest.raises(HTTPException) as exc_info:
        predict_route.predict(_make_body())
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# POST /session/{id}/events
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_predictor(monkeypatch: pytest.MonkeyPatch) -> StreamingPredictor:
    """Isolate each test from whatever model (if any) is on disk in this env."""
    predictor = StreamingPredictor()
    monkeypatch.setattr(session_route, "_predictor", predictor)
    return predictor


def _session_request(session_id: str, events: list[dict]) -> session_route.SessionEventsRequest:
    return session_route.SessionEventsRequest(
        contract_version="1.2",
        session_id=session_id,
        player_id="stu_001",
        scenario_type="fire",
        events=events,
    )


def test_session_events_returns_nine_feature_snapshot(fresh_predictor: StreamingPredictor) -> None:
    body = _session_request(
        "sess_1",
        [
            {"timestamp": 0.0, "event_type": "session_start", "x": 0.0, "y": 64.0, "z": 0.0, "hazard_distance": 20.0},
            {"timestamp": 1.0, "event_type": "move", "x": 1.0, "y": 64.0, "z": 1.0, "hazard_distance": 18.0},
            {"timestamp": 2.0, "event_type": "fire_alarm_activate", "x": 1.0, "y": 64.0, "z": 1.0, "hazard_distance": 18.0},
        ],
    )
    response = session_route.post_session_events("sess_1", body)
    assert set(response.features) == set(_NINE_FEATURES)
    assert response.is_complete is False
    assert response.prediction is None  # no model loaded on the fresh predictor


def test_session_events_path_body_mismatch_raises_400(fresh_predictor: StreamingPredictor) -> None:
    body = _session_request("sess_1", [])
    with pytest.raises(HTTPException) as exc_info:
        session_route.post_session_events("different_session_id", body)
    assert exc_info.value.status_code == 400


def test_session_events_closes_buffer_on_session_end(fresh_predictor: StreamingPredictor) -> None:
    start_body = _session_request(
        "sess_2",
        [{"timestamp": 0.0, "event_type": "session_start", "x": 0.0, "y": 64.0, "z": 0.0, "hazard_distance": 20.0}],
    )
    session_route.post_session_events("sess_2", start_body)
    assert fresh_predictor.active_session_count == 1

    end_body = _session_request(
        "sess_2",
        [
            {"timestamp": 10.0, "event_type": "assembly_area_reached", "x": 5.0, "y": 64.0, "z": 5.0, "hazard_distance": 20.0},
            {"timestamp": 10.1, "event_type": "session_end", "x": 5.0, "y": 64.0, "z": 5.0, "hazard_distance": 20.0},
        ],
    )
    response = session_route.post_session_events("sess_2", end_body)

    assert response.is_complete is True
    assert fresh_predictor.active_session_count == 0


def test_delete_session_closes_buffer(fresh_predictor: StreamingPredictor) -> None:
    body = _session_request(
        "sess_3",
        [{"timestamp": 0.0, "event_type": "session_start", "x": 0.0, "y": 64.0, "z": 0.0, "hazard_distance": 20.0}],
    )
    session_route.post_session_events("sess_3", body)
    assert fresh_predictor.active_session_count == 1

    result = session_route.delete_session("sess_3")
    assert result == {"status": "closed", "session_id": "sess_3"}
    assert fresh_predictor.active_session_count == 0


# ---------------------------------------------------------------------------
# Testing-instrument regressions (D-002 RF-05, D-003 MR-09)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
def test_features_request_rejects_non_finite_numbers(literal: str) -> None:
    # Python's JSON parser accepts these literals; they used to pass ge=0 and
    # crash scikit-learn with an HTTP 500.
    payload = '{"player_id": "stu_1", "scenario_type": "fire", ' + ", ".join(
        f'"{k}": {literal if k == "evacuation_time" else v}' for k, v in _NINE_FEATURES.items()
    ) + "}"
    with pytest.raises(ValueError):
        FeaturesRequest.model_validate_json(payload)


def test_session_event_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        session_route.SessionEvent.model_validate_json('{"timestamp": 1.0, "event_type": "move", "x": NaN}')


def test_predict_reports_model_data_provenance(stub_prediction, monkeypatch: pytest.MonkeyPatch) -> None:
    def _synthetic(features_row, model_path=None, config_path=None):
        return {**stub_prediction(features_row), "data_provenance": "synthetic"}

    monkeypatch.setattr(predict_route, "predict_preparedness_full", _synthetic)
    assert predict_route.predict(_make_body()).dataProvenance == "synthetic"


def test_predict_provenance_defaults_to_unknown(stub_prediction) -> None:
    assert predict_route.predict(_make_body()).dataProvenance == "unknown"


def test_validation_error_body_is_json_safe_for_non_finite_input() -> None:
    # FastAPI's default 422 handler echoes the rejected input; for inf/nan that
    # made the error response itself unencodable (HTTP 500).
    import asyncio
    import json

    from fastapi.exceptions import RequestValidationError

    from api.main import validation_error_handler

    exc = RequestValidationError(
        [{"type": "finite_number", "loc": ("body", "evacuation_time"), "msg": "Input should be a finite number", "input": float("inf")}]
    )
    response = asyncio.run(validation_error_handler(None, exc))
    assert response.status_code == 422
    assert json.loads(response.body)["detail"][0]["input"] == "inf"


# ---------------------------------------------------------------------------
# Testing-instrument Minor regressions (D-008 RF-03) + range-contract sync (D-010)
# ---------------------------------------------------------------------------


def test_features_request_rejects_unknown_extra_field() -> None:
    with pytest.raises(ValueError):
        FeaturesRequest(player_id="stu_1", scenario_type="fire", heart_rate=88.0, **_NINE_FEATURES)


def test_api_bounds_match_feature_value_ranges() -> None:
    # api/schemas.py (ge/le) and data_schema.FEATURE_VALUE_RANGES must not drift:
    # the API rejects the same values the training loader rejects.
    from midrr_classifier.data_schema import FEATURE_VALUE_RANGES

    for feat, (lo, hi) in FEATURE_VALUE_RANGES.items():
        meta = {type(m).__name__: m for m in FeaturesRequest.model_fields[feat].metadata}
        assert (meta["Ge"].ge if "Ge" in meta else None) == lo, feat
        assert (meta["Le"].le if "Le" in meta else None) == hi, feat
