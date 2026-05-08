"""Schema definitions for raw gameplay logs and the engineered feature table.

These constants serve as the single reference point for:
- Expected columns and types when loading raw CSV files.
- Expected columns and types for the feature table consumed by the model.
- The valid class labels for the target variable.

Validation helpers raise :class:`ValueError` early so data problems are
caught at ingestion time rather than deep inside model training.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Raw event-log schema
# One row = one in-game event for one player in one simulation run.
# ---------------------------------------------------------------------------

RAW_LOG_SCHEMA: dict[str, type] = {
    "player_id": str,       # Unique player / student identifier
    "scenario_type": str,   # "fire" or "earthquake"
    "timestamp": float,     # Seconds since scenario start
    "x": float,             # Player X coordinate in Minecraft world
    "y": float,             # Player Y coordinate (height)
    "z": float,             # Player Z coordinate
    "event_type": str,      # e.g. "move", "door_open", "extinguisher_use",
                            #       "emergency_exit", "hazard_proximity"
    "hazard_distance": float,  # Euclidean distance to nearest hazard (blocks)
    "preparedness_level": str, # "High" / "Moderate" / "Low"
                               # (repeated per row from the run-level label)
}

# ---------------------------------------------------------------------------
# Engineered feature schema
# One row = one player × one simulation run, after feature engineering.
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: dict[str, type] = {
    "player_id": str,
    "scenario_type": str,
    "evacuation_time": float,         # seconds from scenario start to exit
    "decision_delay": float,          # seconds from first hazard to first action
    "path_efficiency_ratio": float,   # straight-line dist / total path length  ∈ (0, 1]
    "hazard_avoidance_ratio": float,  # fraction of timesteps with safe distance ∈ [0, 1]
    "interaction_frequency": float,   # safety interactions per second
    "panic_proxy": float,             # std-dev of bearing changes (higher = more erratic)
    "preparedness_level": str,        # target label
}

# ---------------------------------------------------------------------------
# Valid target classes (ordered High → Low for consistent confusion matrices)
# ---------------------------------------------------------------------------

LABEL_CLASSES: list[str] = ["High", "Moderate", "Low"]

# Minimum safe distance (in Minecraft blocks) used by hazard_avoidance_ratio.
# TODO: Calibrate this threshold with domain experts / chapter 3 definitions.
SAFE_HAZARD_DISTANCE: float = 5.0

# Event types that count as "safety interactions" for interaction_frequency.
INTERACTION_EVENT_TYPES: set[str] = {
    "door_open",
    "extinguisher_use",
    "emergency_exit",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Assert that *df* contains all expected raw-log columns.

    Args:
        df: DataFrame loaded from a raw gameplay CSV.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = set(RAW_LOG_SCHEMA.keys()) - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw log is missing required columns: {sorted(missing)}"
        )


def validate_feature_schema(df: pd.DataFrame) -> None:
    """Assert that *df* contains all expected feature-table columns.

    Args:
        df: DataFrame loaded from a processed feature CSV.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = set(FEATURE_SCHEMA.keys()) - set(df.columns)
    if missing:
        raise ValueError(
            f"Feature table is missing required columns: {sorted(missing)}"
        )
