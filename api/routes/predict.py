"""POST /predict route — the core endpoint of the MiDRR API.

Flow (same as any REST handler you've written before):
  1. FastAPI deserialises the JSON body into FeaturesRequest (Pydantic validates it)
  2. We build a pandas Series from those 6 feature values (the format inference.py expects)
  3. We call predict_preparedness_full() which returns label + probabilities + SHAP values
  4. We sort features by |SHAP| (highest absolute impact first) and build the response

featureImportance weights are now per-student SHAP values (can be negative):
  positive weight → that feature pushed the prediction toward prepLevel
  negative weight → that feature actually helped the student (a relative strength)

Error handling:
  - Model not trained yet (FileNotFoundError) → 503 Service Unavailable
  - Missing/wrong feature names (KeyError)    → 422 Unprocessable Entity
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from api.feedback import generate_result_text
from api.schemas import FeatureWeight, FeaturesRequest, PredictResponse
from midrr_classifier.inference import predict_preparedness_full

router = APIRouter()

_FEATURE_COLS = [
    "evacuation_time",
    "decision_delay",
    "path_efficiency_ratio",
    "hazard_avoidance_ratio",
    "interaction_frequency",
    "panic_proxy",
]


@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(body: FeaturesRequest) -> PredictResponse:
    """Predict a student's disaster preparedness level from their session features.

    Accepts the six engineered features for one player in one simulation run
    and returns the predicted level, a 0–100 confidence score, per-feature
    importance weights, and a personalised feedback message.
    """
    # Build a pandas Series from the request — this is what inference.py expects.
    # Think of it like passing a row object from a database query into a function.
    features = pd.Series(
        {col: getattr(body, col) for col in _FEATURE_COLS},
        dtype=float,
    )

    try:
        result = predict_preparedness_full(features)
    except FileNotFoundError as exc:
        # Model hasn't been trained yet — this is expected during early development.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Train a model first with `python -m midrr_classifier.train`.",
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required feature: {exc}",
        ) from exc

    label: str = result["label"]
    probabilities: dict[str, float] = result["probabilities"]
    shap_vals: dict[str, float] = result["shap_values"]

    # prepScore = probability of the predicted class, scaled to 0–100.
    # Example: model is 87% confident the student is LOW → prepScore = 87.
    prep_score = round(probabilities.get(label, 0.0) * 100)

    # Sort by ABSOLUTE SHAP value — highest impact first regardless of direction.
    # The sign (positive/negative) is preserved in the weight field so the
    # dashboard can display direction if it chooses to.
    sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    feature_importance_list = [
        FeatureWeight(feature=feat, weight=round(val, 4))
        for feat, val in sorted_shap
    ]

    # The top feature is the one with highest |SHAP| — most influential for this student.
    top_feature = sorted_shap[0][0] if sorted_shap else "evacuation_time"
    top_shap_value = sorted_shap[0][1] if sorted_shap else 0.0
    result_text = generate_result_text(label, top_feature, top_shap_value, shap_vals)

    return PredictResponse(
        prepLevel=label,
        prepScore=prep_score,
        featureImportance=feature_importance_list,
        resultText=result_text,
    )
