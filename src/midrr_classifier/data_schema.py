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
    "preparedness_level": str, # "HIGH" / "MODERATE" / "LOW"
                               # (repeated per row from the run-level label)
}

# ---------------------------------------------------------------------------
# Engineered feature schema
# One row = one player × one simulation run, after feature engineering.
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: dict[str, type] = {
    "player_id": str,
    "scenario_type": str,
    "evacuation_time": float,         # seconds from scenario start to assembly_area_reached (NOT emergency_exit)
    "decision_delay": float,          # seconds from first hazard zone entry to first safety interaction
    "path_efficiency_ratio": float,   # straight-line dist / total path length  ∈ (0, 1]
    "hazard_avoidance_ratio": float,  # fraction of timesteps with hazard_distance >= SAFE_HAZARD_DISTANCE ∈ [0, 1]
    "interaction_frequency": float,   # qualifying safety interactions per second; extinguisher_use excluded when alone
    "panic_proxy": float,             # std-dev of bearing changes (higher = more erratic)
    "preparedness_level": str,        # target label
}

# ---------------------------------------------------------------------------
# Valid target classes (ordered High → Low for consistent confusion matrices)
# ---------------------------------------------------------------------------

LABEL_CLASSES: list[str] = ["HIGH", "MODERATE", "LOW"]

# Minimum safe distance (in Minecraft blocks) used by hazard_avoidance_ratio.
# BFP plan specifies 2–3 m clearance; 5.0 blocks is a conservative starting proxy.
# Calibrate with domain experts / BFP officers during Phase 4.
SAFE_HAZARD_DISTANCE: float = 5.0

# Event types that count as qualifying safety interactions for interaction_frequency.
# CRITICAL — extinguisher_use is included here but compute_interaction_frequency() MUST
# filter it: only count extinguisher_use when nearby_player_count > 0 at event time.
# Fighting fire while alone is a BFP violation ("DO NOT FIGHT FIRE IF ALONE"), not a
# positive safety action. Solo extinguisher use must not increase interaction_frequency.
INTERACTION_EVENT_TYPES: set[str] = {
    "door_open",
    "fire_alarm_activate",      # COMMUNICATE step in BFP ISOLATE→COMMUNICATE→EVACUATE→RECORD
    "assembly_area_reached",    # true evacuation success (replaces emergency_exit as the endpoint)
    "extinguisher_use",         # conditional — see note above; filter by nearby_player_count > 0
}

# Event types that count as a valid "first safety action" for decision_delay.
# This is a DIFFERENT set from INTERACTION_EVENT_TYPES:
#   - emergency_exit IS included  (passing a designated exit = acting toward evacuation)
#   - assembly_area_reached is NOT included (that is the evacuation endpoint, not an action)
# Source: telemetry_contract.md §4, FEATURE_DEFINITIONS["decision_delay"].
DECISION_DELAY_ACTION_TYPES: set[str] = {
    "fire_alarm_activate",
    "door_open",
    "extinguisher_use",
    "emergency_exit",
}

# ---------------------------------------------------------------------------
# Feature operational definitions (locked to Chapter 3)
# Single source of truth for all compute_* functions in feature_engineering.py.
# ---------------------------------------------------------------------------

FEATURE_DEFINITIONS: dict[str, str] = {
    "evacuation_time": (
        "Elapsed seconds from scenario start (t=0) to the `assembly_area_reached` event. "
        "End-point is assembly_area_reached, NOT emergency_exit — exiting a building is a "
        "waypoint, not evacuation success (BFP: 'proceed to the closest assembly area'). "
        "If the player never reaches the assembly area, cap at the scenario time-limit."
    ),
    "decision_delay": (
        "Elapsed seconds from first entry into the hazard danger zone "
        "(hazard_distance < SAFE_HAZARD_DISTANCE) to the player's first qualifying safety "
        "interaction (door_open, fire_alarm_activate, or movement toward a designated exit). "
        "Measures initial reaction latency. End-point is first safety action, not "
        "assembly_area_reached."
    ),
    "path_efficiency_ratio": (
        "Straight-line Euclidean distance from the player's position at scenario start to the "
        "assembly area, divided by the cumulative path length (sum of per-tick step distances). "
        "Range (0, 1]. Value of 1.0 = perfectly direct path. Assembly-area coordinates are "
        "taken from map_metadata.json (the LSPU floor plan), not the nearest in-game exit."
    ),
    "hazard_avoidance_ratio": (
        "Fraction of per-tick timesteps where hazard_distance >= SAFE_HAZARD_DISTANCE. "
        "Range [0, 1]. 1.0 = always at safe distance. Computed over the full session window "
        "(scenario start to assembly_area_reached or scenario end, whichever comes first). "
        "SAFE_HAZARD_DISTANCE = 5.0 blocks (pending domain-expert calibration in Phase 4)."
    ),
    "interaction_frequency": (
        "Count of qualifying safety interactions divided by total session duration in seconds. "
        "Qualifying events: door_open, fire_alarm_activate, assembly_area_reached. "
        "extinguisher_use counts ONLY when nearby_player_count > 0 at event time — "
        "solo extinguisher use is a BFP violation and must be excluded from the count."
    ),
    "panic_proxy": (
        "Standard deviation of bearing changes (turn angles in degrees) across consecutive "
        "per-tick position vectors (x, z plane). Higher values indicate more erratic, "
        "unpredictable movement consistent with panic behavior. Computed over the full "
        "session trajectory from scenario start to assembly_area_reached or scenario end."
    ),
}

# ---------------------------------------------------------------------------
# Chapter 3 attribute-to-feature mapping
#
# Chapter 3, Table 1 (Granular Logging Framework) lists 8 raw logged attributes.
# This classifier produces 6 engineered features. The mapping below resolves the
# mismatch. Examiners should find a 1:1 traceable link from Table 1 to the model.
#
# Raw attribute (Ch3 Table 1)   Role        → Engineered feature(s)
# ─────────────────────────────────────────────────────────────────────────────
# Player position (x, y, z)    raw input   → path_efficiency_ratio, panic_proxy
# Timestamp / event time        raw input   → evacuation_time, decision_delay,
#                                              interaction_frequency (denominator)
# Event type                    raw input   → decision_delay (trigger),
#                                              interaction_frequency (numerator)
# Hazard distance               raw input   → hazard_avoidance_ratio
# Task Completion Time          raw input   → evacuation_time
#   Not a separate feature. It is the conceptual label for evacuation_time,
#   operationalized as seconds from scenario start to assembly_area_reached.
# Decision Sequence             raw input   → decision_delay
#   Not a separate feature. The ordering and timing of protective decisions is
#   captured by decision_delay (latency to first qualifying safety interaction).
# Safety Compliance             raw input   → interaction_frequency,
#                                              hazard_avoidance_ratio
#   Not a separate feature. Compliance is decomposed into two measurable proxies:
#   (a) rate of correct safety interactions, and (b) fraction of time at safe
#   distance from the hazard.
# Scenario type                 stratum var → (not a model feature)
#   Used for per-scenario grouping and stratified splitting only. Models are
#   trained separately per scenario type or with scenario as a group variable.
#
# CONCLUSION: Task Completion Time, Decision Sequence, and Safety Compliance are
# conceptual raw attributes from Table 1 that serve as the theoretical grounding
# for evacuation_time, decision_delay, and the compliance-pair features respectively.
# They are NOT additional features beyond the six. All 8 raw attributes are
# accounted for across the 6 engineered features + 1 stratification variable.
# ---------------------------------------------------------------------------
CH3_ATTRIBUTE_MAPPING: dict[str, dict] = {
    "player_position_xyz": {
        "role": "raw_input",
        "features": ["path_efficiency_ratio", "panic_proxy"],
    },
    "timestamp_event_time": {
        "role": "raw_input",
        "features": ["evacuation_time", "decision_delay", "interaction_frequency"],
    },
    "event_type": {
        "role": "raw_input",
        "features": ["decision_delay", "interaction_frequency"],
    },
    "hazard_distance": {
        "role": "raw_input",
        "features": ["hazard_avoidance_ratio"],
    },
    "task_completion_time": {
        "role": "raw_input",
        "features": ["evacuation_time"],
        "note": "Conceptual basis for evacuation_time; not a separate feature.",
    },
    "decision_sequence": {
        "role": "raw_input",
        "features": ["decision_delay"],
        "note": "Ordering/timing of decisions captured by decision_delay; not a separate feature.",
    },
    "safety_compliance": {
        "role": "raw_input",
        "features": ["interaction_frequency", "hazard_avoidance_ratio"],
        "note": "Decomposed into two measurable proxies; not a separate feature.",
    },
    "scenario_type": {
        "role": "stratification_variable",
        "features": [],
        "note": "Used for grouping and stratified splitting only; not a model feature.",
    },
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
