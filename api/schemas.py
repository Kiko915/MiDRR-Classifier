"""Pydantic request/response models for the MiDRR API.

These are the "TypeScript interfaces" of the Python world — they define
the exact JSON shape the API accepts and returns.  Pydantic validates
incoming data automatically, so the route handler only sees clean values.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class FeaturesRequest(BaseModel):
    """Body for POST /predict when the caller sends pre-computed features.

    The six fields match the engineered features produced by
    ``feature_engineering.build_feature_table()``.
    """

    player_id: str = Field(..., description="Unique player / student identifier.")
    scenario_type: str = Field(
        ...,
        description="Simulation scenario: 'fire', 'earthquake', 'ccs_fire', or 'ccs_earthquake'.",
    )

    # The six engineered features
    evacuation_time: float = Field(..., ge=0, description="Seconds from scenario start to assembly area.")
    decision_delay: float = Field(..., ge=0, description="Seconds from hazard detection to first safety action.")
    path_efficiency_ratio: float = Field(..., ge=0, le=1, description="Straight-line / total path length (0–1].")
    hazard_avoidance_ratio: float = Field(..., ge=0, le=1, description="Fraction of timesteps at safe distance [0–1].")
    interaction_frequency: float = Field(..., ge=0, description="Qualifying safety interactions per second.")
    panic_proxy: float = Field(..., ge=0, description="Std-dev of bearing changes (higher = more erratic).")


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
