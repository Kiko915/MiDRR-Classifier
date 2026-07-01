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
    "scenario_type": str,   # "fire" or "earthquake" (canonical, post-normalization)
    "timestamp": float,     # Seconds since scenario start
    "x": float,             # Player X coordinate in Minecraft world
    "y": float,             # Player Y coordinate (height)
    "z": float,             # Player Z coordinate
    "event_type": str,      # see EVENT_TYPES below for the full vocabulary
    "hazard_distance": float,  # Euclidean distance to nearest hazard (blocks)
    "nearby_player_count": float,  # other players near the event (0 = alone).
                            # Required column post-v1.2 (was silently optional —
                            # code already reads it via `if "nearby_player_count"
                            # in events_df.columns`). Only meaningfully populated
                            # on extinguisher_use / ext_spray / pin_pull rows;
                            # blank/NaN elsewhere. Drives the BFP "DO NOT FIGHT
                            # FIRE IF ALONE" rule (spray_accuracy, resource_utilization).
    # NOTE: preparedness_level is deliberately absent — raw logs must not contain
    # labels. Labels are joined later from the expert-scoring spreadsheet keyed on
    # session_id, or attached as a rule-based weak label — see `label_source` on
    # FEATURE_SCHEMA and `labeling.py` (Phase 2.5). See telemetry_contract.md §3.
    #
    # Optional, event-specific context fields NOT enforced by validate_raw_schema()
    # (sparse — only populated for the event types that carry them), following the
    # same convention as the pre-existing `interaction_target` field:
    #   "hit_fire"           (bool/0-1) on `ext_spray` — did the spray connect with
    #                        an active hazard tile? Drives spray_accuracy.
    #   "extinguisher_class" (str) on `ext_spray`/`pin_pull` — CO2 / DRY_POWDER / ABC,
    #                        checked against EXTINGUISHER_CLASS_BY_ROOM for correctness.
    #   "phase"              (str) on `phase_transition` — one of SIM_PHASES.
}

# ---------------------------------------------------------------------------
# Engineered feature schema
# One row = one player × one simulation run, after feature engineering.
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: dict[str, type] = {
    "player_id": str,
    "scenario_type": str,
    # --- 9 locked features (v1.2, post-BFP-consultation) --------------------
    # Order matches config.MiDRRConfig.feature_cols. Fire computation is
    # primary; earthquake runs an analog computation into the same slot so
    # both scenarios produce one 9-column vector. See FEATURE_DEFINITIONS.
    "decision_latency": float,        # was decision_delay; re-anchored to SIM_START (not first hazard exposure)
    "spray_accuracy": float,          # NEW. fire: hits-on-fire / total sprays. quake: Drop-Cover-Hold correctness
    "path_efficiency_ratio": float,   # straight-line dist / total path length  ∈ (0, 1]  (unchanged)
    "hazard_avoidance_ratio": float,  # fraction of timesteps with hazard_distance >= SAFE_HAZARD_DISTANCE ∈ [0, 1]  (unchanged)
    "evacuation_time": float,         # seconds from SIM_START to assembly_area_reached (NOT emergency_exit)  (unchanged)
    "interaction_frequency": float,   # fire: alarm+extinguisher+exits per sec. quake: cover-taking+exits per sec
    "resource_utilization": float,    # NEW. fire: correct PASS sequencing (pin-pull before spray). quake: took/re-took sturdy cover
    "panic_proxy": float,             # REDEFINED: std-dev of movement speed^2 (was std-dev of bearing-change angles)
    "situational_awareness": float,   # NEW. composite: alarm/cover + safe route + low panic (+ aftershock awareness for quake)
    "preparedness_level": str,        # target label
    "label_source": str,              # "expert" (BFP-instructor override, gold) or "rule" (game rule-based weak label)
}

# ---------------------------------------------------------------------------
# Valid target classes (ordered High → Low for consistent confusion matrices)
# ---------------------------------------------------------------------------

LABEL_CLASSES: list[str] = ["HIGH", "MODERATE", "LOW"]

# ---------------------------------------------------------------------------
# Label provenance (Phase 2.5) — "expert" is the BFP-instructor-validated
# override (gold; the ONLY source allowed in the test split). "rule" is the
# game's own rule-based prep_level (weak label, training-only; see labeling.py
# and docs/labeling_rubric.md §7 for the circularity guard).
# ---------------------------------------------------------------------------

LABEL_SOURCES: set[str] = {"expert", "rule"}

# ---------------------------------------------------------------------------
# Valid scenario types  (4 distinct values — do NOT collapse ccs_fire → fire)
# ccs_* scenarios take place in the CCS Admin Building (different assembly zone).
# fire / earthquake scenarios take place in the Library building.
# Source: BERONG_SMP_WEB/apps/dashboard/src/lib/floorplans.ts
# ---------------------------------------------------------------------------

SCENARIO_TYPES: set[str] = {"fire", "earthquake", "ccs_fire", "ccs_earthquake"}

# ---------------------------------------------------------------------------
# Map metadata — assembly zone bounds per scenario type
# Used by compute_path_efficiency() as the fallback evacuation endpoint when
# assembly_area_reached was never logged (e.g. player ran out of time).
# Coordinates are Minecraft world XZ; Y is omitted (horizontal plane only).
# Source: BERONG_SMP_WEB/apps/dashboard/src/lib/floorplans.ts (verified)
# ---------------------------------------------------------------------------

# Centre of each assembly zone in world XZ coordinates.
# path_efficiency_ratio uses centre as the "ideal" endpoint when the event is absent.
ASSEMBLY_ZONE_CENTRE: dict[str, tuple[float, float]] = {
    "ccs_fire":       (106.0, 81.5),   # CCS Admin: X:76–136, Z:73–90
    "ccs_earthquake": (106.0, 81.5),
    "fire":           (53.0,  73.0),   # Library:   X:30–76,  Z:64–82
    "earthquake":     (53.0,  73.0),
}

# Full assembly zone rectangles (for validation / future use).
ASSEMBLY_ZONE_BOUNDS: dict[str, dict[str, float]] = {
    "ccs_fire":       {"xMin": 76, "xMax": 136, "zMin": 73, "zMax": 90},
    "ccs_earthquake": {"xMin": 76, "xMax": 136, "zMin": 73, "zMax": 90},
    "fire":           {"xMin": 30, "xMax": 76,  "zMin": 64, "zMax": 82},
    "earthquake":     {"xMin": 30, "xMax": 76,  "zMin": 64, "zMax": 82},
}

# Y-coordinate boundary separating ground and upper floors in the CCS building.
# ground floor: y <= CCS_FLOOR_Y_BOUNDARY  |  upper floor: y > CCS_FLOOR_Y_BOUNDARY
# Source: floorplans.ts CCS_FLOOR_Y_BOUNDARY = -26
CCS_FLOOR_Y_BOUNDARY: float = -26.0

# Minimum safe distance (in Minecraft blocks) used by hazard_avoidance_ratio.
# BFP plan specifies 2–3 m clearance; 5.0 blocks is a conservative starting proxy.
# Calibrate with domain experts / BFP officers during Phase 4.
SAFE_HAZARD_DISTANCE: float = 5.0

# ---------------------------------------------------------------------------
# Event vocabulary (v1.2 — post-BFP-consultation, 3-phase fire sim + quake DCH)
# Single source of truth for every `event_type` value the mod may emit.
# ---------------------------------------------------------------------------

# 3-phase fire state machine driving the game's own rule-based prep_level.
# See labeling.py::phase_outcome_label() (Phase 2.5 step 3).
SIM_PHASES: list[str] = ["prevention", "intervention", "evacuation"]

# New (v1.2) event_type values, named so feature_engineering.py and synth.py
# reference one spelling instead of repeating string literals.
EXT_SPRAY_EVENT: str = "ext_spray"                # PASS "Squeeze/Sweep"; carries hit_fire, extinguisher_class
PIN_PULL_EVENT: str = "pin_pull"                  # PASS "Pull"; correct sequencing precedes ext_spray
HAZARD_NEUTRALIZE_EVENT: str = "hazard_neutralize"  # a hazard tile/source was extinguished
PHASE_TRANSITION_EVENT: str = "phase_transition"    # carries `phase` ∈ SIM_PHASES
DROP_COVER_HOLD_EVENT: str = "drop_cover_hold"      # earthquake analog of taking cover

# Full event_type vocabulary (fire + earthquake). Mirrors telemetry_contract.md §4.
EVENT_TYPES: set[str] = {
    "session_start",
    "move",
    "hazard_proximity",
    "fire_alarm_activate",
    "door_open",
    "extinguisher_use",
    "emergency_exit",
    "assembly_area_reached",
    "session_end",
    EXT_SPRAY_EVENT,
    PIN_PULL_EVENT,
    HAZARD_NEUTRALIZE_EVENT,
    PHASE_TRANSITION_EVENT,
    DROP_COVER_HOLD_EVENT,
}

# Correct extinguisher class per room, per the BFP-reviewed simulation design.
# resource_utilization / spray_accuracy check the player's `extinguisher_class`
# against this map — using the wrong class for the room is incorrect PASS technique.
EXTINGUISHER_CLASS_BY_ROOM: dict[str, str] = {
    "laboratory": "CO2",
    "cafeteria": "DRY_POWDER",
    "classroom": "ABC",
}

# Event types that count as qualifying safety interactions for interaction_frequency.
# CRITICAL — extinguisher_use/ext_spray are included here but
# compute_interaction_frequency() MUST filter them: only count when
# nearby_player_count > 0 at event time. Fighting fire while alone is a BFP
# violation ("DO NOT FIGHT FIRE IF ALONE"), not a positive safety action. Solo
# extinguisher use must not increase interaction_frequency.
# Earthquake analog: drop_cover_hold + emergency_exit/assembly_area_reached.
INTERACTION_EVENT_TYPES: set[str] = {
    "door_open",
    "fire_alarm_activate",      # COMMUNICATE step in BFP ISOLATE→COMMUNICATE→EVACUATE→RECORD
    "assembly_area_reached",    # true evacuation success (replaces emergency_exit as the endpoint)
    "extinguisher_use",         # conditional — see note above; filter by nearby_player_count > 0
    EXT_SPRAY_EVENT,            # conditional — same filter as extinguisher_use
    DROP_COVER_HOLD_EVENT,      # earthquake: taking cover is the qualifying interaction
}

# Event types that count as a valid "first safety action" for decision_latency.
# This is a DIFFERENT set from INTERACTION_EVENT_TYPES:
#   - emergency_exit IS included  (passing a designated exit = acting toward evacuation)
#   - assembly_area_reached is NOT included (that is the evacuation endpoint, not an action)
# Fire anchors from SIM_START (session_start); earthquake's qualifying first
# action is DROP_COVER_HOLD_EVENT rather than any of the fire-only actions.
# Source: telemetry_contract.md §4, FEATURE_DEFINITIONS["decision_latency"].
DECISION_LATENCY_ACTION_TYPES: set[str] = {
    "fire_alarm_activate",
    "door_open",
    "extinguisher_use",
    "emergency_exit",
    EXT_SPRAY_EVENT,
    PIN_PULL_EVENT,
}

# Earthquake-only: the qualifying first action for decision_latency during shaking.
EARTHQUAKE_DECISION_LATENCY_ACTION_TYPES: set[str] = {
    DROP_COVER_HOLD_EVENT,
}

# ---------------------------------------------------------------------------
# Feature operational definitions (locked v1.2, post-BFP-consultation)
# Single source of truth for all compute_* functions in feature_engineering.py.
# Each entry documents the FIRE computation first, then the EARTHQUAKE analog —
# both write into the same schema slot so one 9-column vector serves both
# scenario families (scenario_type is the stratifier, not a 10th feature).
# ---------------------------------------------------------------------------

FEATURE_DEFINITIONS: dict[str, str] = {
    "decision_latency": (
        "FIRE: elapsed seconds from SIM_START (`session_start`) to the player's first "
        "qualifying action in DECISION_LATENCY_ACTION_TYPES (fire_alarm_activate, door_open, "
        "extinguisher_use, ext_spray, pin_pull, or emergency_exit). "
        "EARTHQUAKE: elapsed seconds from SIM_START to the first `drop_cover_hold` event. "
        "Re-anchored from first-hazard-exposure (v1.1) to SIM_START (v1.2) so latency reflects "
        "the player's total reaction time from the disaster trigger, not just proximity-triggered "
        "reaction — matches how the game's own phase timer measures Prevention-phase response."
    ),
    "spray_accuracy": (
        "FIRE: count of `ext_spray` events where hit_fire is true, divided by total `ext_spray` "
        "events. Range [0, 1]; undefined (NaN, treat as 0) if no sprays were attempted. "
        "EARTHQUAKE analog: Drop-Cover-Hold correctness score — fraction of `drop_cover_hold` "
        "events performed in a safe spot (away from windows/falling-hazard zones, per map "
        "metadata) rather than a naive count of attempts."
    ),
    "path_efficiency_ratio": (
        "Straight-line Euclidean distance from the player's position at SIM_START to the "
        "assembly area, divided by the cumulative path length (sum of per-tick step distances). "
        "Range (0, 1]. Value of 1.0 = perfectly direct path. Assembly-area coordinates are "
        "taken from map_metadata.json (the LSPU floor plan), not the nearest in-game exit. "
        "Unchanged between fire and earthquake — both measure the post-hazard evacuation leg."
    ),
    "hazard_avoidance_ratio": (
        "Fraction of per-tick timesteps where hazard_distance >= SAFE_HAZARD_DISTANCE. "
        "Range [0, 1]. 1.0 = always at safe distance. Computed over the full session window "
        "(SIM_START to assembly_area_reached or scenario end, whichever comes first). "
        "SAFE_HAZARD_DISTANCE = 5.0 blocks (pending domain-expert calibration in Phase 4). "
        "EARTHQUAKE: hazard_distance is measured to the nearest falling-hazard zone "
        "(map_metadata.json), not a fire tile — same computation, different hazard set."
    ),
    "evacuation_time": (
        "Elapsed seconds from SIM_START (`session_start`) to the `assembly_area_reached` event. "
        "End-point is assembly_area_reached, NOT emergency_exit — exiting a building is a "
        "waypoint, not evacuation success (BFP: 'proceed to the closest assembly area'). "
        "If the player never reaches the assembly area, cap at the scenario time-limit. "
        "EARTHQUAKE: same computation, but the clock only starts counting toward evacuation "
        "credit after shaking stops (post-shaking evacuation is the BFP-correct behavior)."
    ),
    "interaction_frequency": (
        "FIRE: count of qualifying safety interactions divided by total session duration in "
        "seconds. Qualifying events: door_open, fire_alarm_activate, assembly_area_reached, "
        "ext_spray/extinguisher_use. ext_spray/extinguisher_use count ONLY when "
        "nearby_player_count > 0 at event time — solo extinguisher use is a BFP violation and "
        "must be excluded from the count. "
        "EARTHQUAKE: count of drop_cover_hold + emergency_exit + assembly_area_reached events "
        "divided by session duration — cover-taking is the earthquake analog of a safety "
        "interaction."
    ),
    "resource_utilization": (
        "FIRE: correct PASS technique sequencing — 1.0 if the player's first `pin_pull` "
        "precedes their first `ext_spray` (Pull before Squeeze/Sweep) AND extinguisher_class "
        "matches EXTINGUISHER_CLASS_BY_ROOM for the room they used it in; partial credit (0.5) "
        "for one condition met; 0.0 if neither, or if they sprayed without pulling the pin. "
        "EARTHQUAKE analog: took a sturdy/appropriate cover on first drop_cover_hold, and "
        "re-covered on any subsequent aftershock (hazard_neutralize-equivalent: aftershock "
        "response) — 1.0 if both hold, 0.5 if only the initial cover was correct, else 0.0."
    ),
    "panic_proxy": (
        "REDEFINED v1.2 (was: std-dev of bearing-change turn angles). Standard deviation of "
        "per-tick movement speed-squared (|velocity|^2) across the session trajectory from "
        "SIM_START to assembly_area_reached or scenario end. Higher values indicate erratic "
        "speed bursts (sudden sprints/stops) consistent with panic; a bearing-only proxy missed "
        "panicked players who move in a straight line but at wildly inconsistent speed. Same "
        "computation for fire and earthquake."
    ),
    "situational_awareness": (
        "NEW. Composite [0, 1] score, NOT a raw log measurement — combines three already-"
        "computed signals so the model has one 'did they read the situation correctly' feature: "
        "FIRE = mean of (fire_alarm_activate was pressed ? 1 : 0), (path_efficiency_ratio), and "
        "(1 - normalized panic_proxy). "
        "EARTHQUAKE = mean of (drop_cover_hold performed ? 1 : 0), (path_efficiency_ratio), "
        "(1 - normalized panic_proxy), and (re-covered on aftershock ? 1 : 0 — aftershock "
        "awareness, only included when aftershock_count > 0). "
        "Deliberately correlated with other features (by design, as a summary signal) — do not "
        "over-interpret its SHAP contribution in isolation from the components it summarizes."
    ),
}

# ---------------------------------------------------------------------------
# Chapter 3 attribute-to-feature mapping (v1.2 — 9 features)
#
# Chapter 3, Table 1 (Granular Logging Framework) lists 8 raw logged attributes.
# This classifier produced 6 engineered features through v1.1; the BFP
# consultation added 3 more (spray_accuracy, resource_utilization,
# situational_awareness) that are NOT grounded in Ch3 Table 1 — they come from
# the revised simulation design's own PASS/DCH instrumentation. The mapping
# below traces the original 8 attributes plus the new instrumentation.
#
# Raw attribute                 Role        → Engineered feature(s)
# ─────────────────────────────────────────────────────────────────────────────
# Player position (x, y, z)    raw input   → path_efficiency_ratio, panic_proxy
# Timestamp / event time        raw input   → evacuation_time, decision_latency,
#                                              interaction_frequency (denominator)
# Event type                    raw input   → decision_latency (trigger),
#                                              interaction_frequency (numerator)
# Hazard distance               raw input   → hazard_avoidance_ratio
# Task Completion Time          raw input   → evacuation_time
#   Not a separate feature. It is the conceptual label for evacuation_time,
#   operationalized as seconds from SIM_START to assembly_area_reached.
# Decision Sequence             raw input   → decision_latency, resource_utilization
#   Not a separate feature. The ordering/timing of protective decisions is
#   captured by decision_latency (reaction time) and, for the fire PASS
#   sequence specifically, by resource_utilization (pin-pull-before-spray order).
# Safety Compliance             raw input   → interaction_frequency,
#                                              hazard_avoidance_ratio
#   Not a separate feature. Compliance is decomposed into two measurable proxies:
#   (a) rate of correct safety interactions, and (b) fraction of time at safe
#   distance from the hazard.
# Scenario type                 stratum var → (not a model feature)
#   Used for per-scenario grouping and stratified splitting only. Models are
#   trained separately per scenario type or with scenario as a group variable.
# ext_spray.hit_fire (v1.2, NEW) raw input  → spray_accuracy
#   New instrumentation from the BFP-revised design, not in Ch3 Table 1.
# pin_pull + extinguisher_class (v1.2, NEW) raw input → resource_utilization
#   New instrumentation for PASS-technique correctness.
# (derived, no new raw attribute)           → situational_awareness
#   Composite of decision_latency/interaction_frequency, path_efficiency_ratio,
#   and panic_proxy — not tied to a single raw attribute; see FEATURE_DEFINITIONS.
#
# CONCLUSION: Task Completion Time, Decision Sequence, and Safety Compliance are
# conceptual raw attributes from Ch3 Table 1 that ground evacuation_time,
# decision_latency, and the compliance-pair features respectively. All 8
# original raw attributes are accounted for across 6 of the 9 features + 1
# stratification variable; the remaining 3 features (spray_accuracy,
# resource_utilization, situational_awareness) are new BFP-driven additions
# layered on top, not part of the original Ch3 mismatch resolution.
# ---------------------------------------------------------------------------
CH3_ATTRIBUTE_MAPPING: dict[str, dict] = {
    "player_position_xyz": {
        "role": "raw_input",
        "features": ["path_efficiency_ratio", "panic_proxy"],
    },
    "timestamp_event_time": {
        "role": "raw_input",
        "features": ["evacuation_time", "decision_latency", "interaction_frequency"],
    },
    "event_type": {
        "role": "raw_input",
        "features": ["decision_latency", "interaction_frequency"],
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
        "features": ["decision_latency", "resource_utilization"],
        "note": (
            "Ordering/timing of decisions captured by decision_latency; the "
            "fire PASS pull-before-spray order specifically feeds "
            "resource_utilization. Not a separate feature."
        ),
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
    "ext_spray_hit_fire": {
        "role": "raw_input",
        "features": ["spray_accuracy"],
        "note": "NEW v1.2 instrumentation (BFP consultation); not in Ch3 Table 1.",
    },
    "pin_pull_and_extinguisher_class": {
        "role": "raw_input",
        "features": ["resource_utilization"],
        "note": "NEW v1.2 instrumentation for PASS-technique correctness.",
    },
    "situational_awareness_composite": {
        "role": "derived",
        "features": ["situational_awareness"],
        "note": "Composite of other features, not a distinct raw attribute.",
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
