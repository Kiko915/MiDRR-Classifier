"""Pydantic request/response models for the MiDRR API.

These are the "TypeScript interfaces" of the Python world — they define
the exact JSON shape the API accepts and returns.  Pydantic validates
incoming data automatically, so the route handler only sees clean values.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class FeaturesRequest(BaseModel):
    """Body for POST /predict when the caller sends pre-computed features.

    The nine fields match the engineered features produced by
    ``feature_engineering.build_feature_table()`` (v1.2, post-BFP-consultation
    9-feature contract — fire computation; see FEATURE_DEFINITIONS in
    ``data_schema.py`` for the earthquake analog of each).

    ``allow_inf_nan=False``: the JSON literals ``Infinity``/``NaN`` would
    otherwise pass the ``ge``/``le`` checks and crash scikit-learn with a 500.
    ``extra="forbid"``: an unknown field (a renamed or 10th feature) is a
    contract mismatch and gets a 422 instead of being silently dropped.
    The ge/le bounds mirror ``data_schema.FEATURE_VALUE_RANGES`` (kept in
    sync by a test).
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    player_id: str = Field(..., description="Unique player / student identifier.")
    scenario_type: str = Field(
        ...,
        description="Simulation scenario: 'fire', 'earthquake', 'ccs_fire', or 'ccs_earthquake'.",
    )

    # The nine engineered features (v1.2)
    decision_latency: float = Field(..., ge=0, description="Seconds from SIM_START to first valid safety action.")
    spray_accuracy: float = Field(..., ge=0, le=1, description="Fraction of extinguisher sprays that hit the fire.")
    path_efficiency_ratio: float = Field(..., ge=0, le=1, description="Straight-line / total path length (0–1].")
    hazard_avoidance_ratio: float = Field(..., ge=0, le=1, description="Fraction of timesteps at safe distance [0–1].")
    evacuation_time: float = Field(..., ge=0, description="Seconds from SIM_START to assembly area.")
    interaction_frequency: float = Field(..., ge=0, description="Qualifying safety interactions per second.")
    resource_utilization: float = Field(..., ge=0, le=1, description="PASS-technique sequencing correctness.")
    panic_proxy: float = Field(..., ge=0, description="Std-dev of movement speed² (higher = more erratic).")
    situational_awareness: float = Field(..., ge=0, le=1, description="Composite 'read the situation correctly' score.")


class FeatureWeight(BaseModel):
    """One entry in the featureImportance array."""

    feature: str
    weight: float


class PredictResponse(BaseModel):
    """Response body for POST /predict.

    This is the exact contract the BERONG_SMP_WEB dashboard expects.
    Do not rename fields without updating the web repo.
    """

    prepLevel: str = Field(..., description="Predicted level: 'HIGH', 'MODERATE', or 'LOW'.")
    prepScore: int = Field(..., ge=0, le=100, description="Confidence in the prediction, scaled 0–100.")
    featureImportance: List[FeatureWeight] = Field(
        ..., description="Feature importance weights, sorted highest first."
    )
    resultText: str = Field(..., description="Human-readable adaptive feedback for the student.")
    dataProvenance: str = Field(
        "unknown",
        description=(
            "Source of the data the serving model was trained on: 'synthetic', 'real', "
            "'mixed', or 'unknown'. 'synthetic' predictions are placeholders, not real-learner results."
        ),
    )
