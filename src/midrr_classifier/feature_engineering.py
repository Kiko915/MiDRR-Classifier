"""Feature engineering pipeline for the MiDRR-Classifier.

Transforms raw per-event gameplay logs into a one-row-per-player
feature table ready for model training.

Each ``compute_*`` function receives the event log for **a single
player in a single simulation run** and returns a scalar float. Every
function dispatches on ``scenario_type`` (via :func:`_is_earthquake_scenario`)
so the same 9-column feature vector is produced whether the run was a
fire or an earthquake scenario — see FEATURE_DEFINITIONS in
``data_schema.py`` for the fire computation and its earthquake analog.
The main entry-point is :func:`build_feature_table`, which groups
the full log by ``(player_id, scenario_type)`` and applies all
feature functions.

TODO (research team): these are the v1.2 (post-BFP-consultation) formulas.
Validate against synthetic ground truth (Phase 2.5 step 5) and calibrate
thresholds (SAFE_HAZARD_DISTANCE, panic_proxy normalization) with real data
in Phase 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from midrr_classifier.data_schema import (
    ASSEMBLY_ZONE_CENTRE,
    DECISION_LATENCY_ACTION_TYPES,
    DROP_COVER_HOLD_EVENT,
    EARTHQUAKE_DECISION_LATENCY_ACTION_TYPES,
    EARTHQUAKE_INTERACTION_EVENT_TYPES,
    EXT_SPRAY_EVENT,
    INTERACTION_EVENT_TYPES,
    PIN_PULL_EVENT,
    REQUIRED_VALUE_COLUMNS,
    SAFE_HAZARD_DISTANCE,
    find_out_of_range_rows,
)
from midrr_classifier.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scenario dispatch helper
# ---------------------------------------------------------------------------


def _is_earthquake_scenario(scenario_type: object) -> bool:
    """True for ``earthquake`` and ``ccs_earthquake``, False for fire family."""
    return isinstance(scenario_type, str) and "earthquake" in scenario_type


def _scenario_of(events_df: pd.DataFrame) -> object:
    if events_df.empty or "scenario_type" not in events_df.columns:
        return None
    return events_df["scenario_type"].iloc[0]


# ---------------------------------------------------------------------------
# Per-player feature functions
# Each function accepts a DataFrame of events for ONE player × ONE run.
# ---------------------------------------------------------------------------


def compute_evacuation_time(events_df: pd.DataFrame) -> float:
    """Seconds from SIM_START to reaching the assembly area.

    True evacuation success = ``assembly_area_reached``, not ``emergency_exit``
    (BFP: 'proceed to the closest assembly area'). If the player never reaches
    the assembly area, the scenario time-limit is used as the cap (taken from
    the ``session_end`` timestamp). Falls back to ``timestamp.max`` if
    ``session_end`` is also absent. SIM_START is the ``session_start`` event
    (or the earliest event), subtracted so the result is elapsed time even if
    the clock did not start at 0. Earthquake runs use the same computation
    — evacuation credit still ends at ``assembly_area_reached``, which by
    convention only fires after shaking stops.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Elapsed seconds (≥ 0).
    """
    if events_df.empty:
        return 0.0

    # Measured from SIM_START, same anchor as decision_latency, so a mod that
    # logs session_start at a non-zero clock value still gets elapsed seconds.
    return _evacuation_window_end(events_df) - _sim_start(events_df)


def _sim_start(events_df: pd.DataFrame) -> float:
    """SIM_START timestamp: the ``session_start`` event, else the earliest event."""
    sim_start = events_df[events_df["event_type"] == "session_start"]
    if not sim_start.empty:
        return float(sim_start["timestamp"].min())
    return float(events_df["timestamp"].min())


def compute_decision_latency(events_df: pd.DataFrame) -> float:
    """Latency from SIM_START to the first valid safety action.

    Re-anchored (v1.2) from first-hazard-exposure to ``SIM_START`` — the
    ``session_start`` event, or ``timestamp.min()`` if absent — so latency
    reflects total reaction time from the disaster trigger.

    FIRE: "first valid action" = first event in
    :data:`~midrr_classifier.data_schema.DECISION_LATENCY_ACTION_TYPES`
    (``fire_alarm_activate``, ``door_open``, ``extinguisher_use``,
    ``ext_spray``, ``pin_pull``, ``emergency_exit``).
    EARTHQUAKE: "first valid action" = first
    :data:`~midrr_classifier.data_schema.DROP_COVER_HOLD_EVENT`.
    ``assembly_area_reached`` is intentionally excluded from both — it is the
    evacuation endpoint, not an initial safety reaction.

    Args:
        events_df: Event log for a single player × run, sorted by
            ``timestamp`` (ascending).

    Returns:
        Delay in seconds. Returns the total scenario duration (worst-case
        penalty) if no valid action is ever observed.
    """
    if events_df.empty:
        return 0.0

    df = events_df.sort_values("timestamp")

    t0 = _sim_start(df)

    action_types = (
        EARTHQUAKE_DECISION_LATENCY_ACTION_TYPES
        if _is_earthquake_scenario(_scenario_of(df))
        else DECISION_LATENCY_ACTION_TYPES
    )
    action_rows = df[df["event_type"].isin(action_types) & (df["timestamp"] >= t0)]

    if action_rows.empty:
        return compute_evacuation_time(df)

    return float(action_rows["timestamp"].iloc[0] - t0)


def compute_spray_accuracy(events_df: pd.DataFrame) -> float:
    """PASS-technique accuracy (fire) or Drop-Cover-Hold correctness (quake).

    FIRE: count of ``ext_spray`` events where ``hit_fire`` is true, divided
    by total ``ext_spray`` events. Returns 0.0 (not NaN) if no sprays were
    attempted — mirrors the "did nothing" penalty used elsewhere in this module.

    EARTHQUAKE: fraction of ``drop_cover_hold`` events performed at a safe
    distance from the hazard (``hazard_distance >= SAFE_HAZARD_DISTANCE`` at
    the moment of the event) — a proxy for "took cover away from
    windows/falling-hazard zones" until per-tile map metadata is available.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Ratio in [0, 1].
    """
    if events_df.empty:
        return 0.0

    if _is_earthquake_scenario(_scenario_of(events_df)):
        dch = events_df[events_df["event_type"] == DROP_COVER_HOLD_EVENT]
        if dch.empty:
            return 0.0
        safe = pd.to_numeric(dch.get("hazard_distance"), errors="coerce") >= SAFE_HAZARD_DISTANCE
        return float(safe.fillna(False).mean())

    sprays = events_df[events_df["event_type"] == EXT_SPRAY_EVENT]
    if sprays.empty:
        return 0.0
    hit_fire = pd.to_numeric(sprays.get("hit_fire"), errors="coerce").fillna(0)
    return float((hit_fire > 0).mean())


def compute_path_efficiency(events_df: pd.DataFrame) -> float:
    """Ratio of straight-line displacement to total path length.

    Straight-line distance is measured from the player's spawn position
    (first ``move`` row) to the **evacuation endpoint**: the position
    recorded on the ``assembly_area_reached`` event, or the final ``move``
    position if the player never reached the assembly area. Only ``move``
    events up to the evacuation endpoint are included in the path (no
    post-evacuation wandering inflates the denominator). Unchanged between
    fire and earthquake — both measure the post-hazard evacuation leg.

    Formula::

        path_efficiency = straight_line(start → endpoint) / cumulative_path_length

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Path efficiency ratio in (0, 1].  Returns 1.0 if the player
        did not move (degenerate case).
    """
    # Determine the time-cut and target position for the evacuation endpoint.
    assembly = events_df[events_df["event_type"] == "assembly_area_reached"]
    if not assembly.empty:
        t_end = float(assembly["timestamp"].iloc[0])
        endpoint = assembly[["x", "y", "z"]].iloc[0].to_numpy(dtype=float)
    else:
        session_end = events_df[events_df["event_type"] == "session_end"]
        t_end = (
            float(session_end["timestamp"].iloc[0])
            if not session_end.empty
            else float(events_df["timestamp"].max())
        )
        # Use the known assembly-zone centre for the scenario type as the ideal
        # endpoint; fall back to the last recorded move position if unknown.
        scenario = _scenario_of(events_df)
        if scenario and scenario in ASSEMBLY_ZONE_CENTRE:
            cx, cz = ASSEMBLY_ZONE_CENTRE[scenario]
            y_val = float(events_df["y"].iloc[-1]) if "y" in events_df.columns else 0.0
            endpoint = np.array([cx, y_val, cz])
        else:
            endpoint = None  # fall back to last move position below

    moves = events_df[
        (events_df["event_type"] == "move") & (events_df["timestamp"] <= t_end)
    ].sort_values("timestamp")

    if len(moves) < 2:
        return 1.0

    coords = moves[["x", "y", "z"]].to_numpy(dtype=float)
    start = coords[0]
    end = endpoint if endpoint is not None else coords[-1]

    straight = float(np.linalg.norm(end - start))
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    total_path = float(steps.sum())

    if total_path == 0.0:
        return 1.0

    return min(straight / total_path, 1.0)


def _evacuation_window_end(events_df: pd.DataFrame) -> float:
    """Timestamp that closes the scored window: assembly, else session_end, else last event."""
    for event_type in ("assembly_area_reached", "session_end"):
        rows = events_df[events_df["event_type"] == event_type]
        if not rows.empty:
            return float(rows["timestamp"].iloc[0])
    return float(events_df["timestamp"].max())


def compute_hazard_avoidance_ratio(events_df: pd.DataFrame) -> float:
    """Fraction of timesteps where the player maintained a safe distance.

    A higher ratio means the player consistently stayed away from
    hazards (fire, or falling-hazard zones for earthquake — same
    computation, different hazard set per telemetry_contract.md).

    Per FEATURE_DEFINITIONS the denominator is per-tick timesteps inside the
    SIM_START → assembly_area_reached (or scenario end) window, so:

    * only ``move`` ticks are counted — discrete events (alarm, door,
      phase_transition, ...) are not timesteps. If a log carries no ``move``
      rows at all, every event row is used as a fallback.
    * rows after the window closes (post-assembly wandering) are excluded.
    * rows whose ``hazard_distance`` is blank are excluded rather than being
      counted as unsafe — a missing measurement is not evidence of danger.

    TODO: Align SAFE_HAZARD_DISTANCE threshold with Chapter 3 rubric.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Ratio in [0, 1]. 0.0 if no timestep carries a hazard distance.
    """
    if events_df.empty:
        return 0.0

    in_window = events_df[events_df["timestamp"] <= _evacuation_window_end(events_df)]
    ticks = in_window[in_window["event_type"] == "move"]
    if ticks.empty:
        ticks = in_window

    distance = pd.to_numeric(ticks["hazard_distance"], errors="coerce").dropna()
    if distance.empty:
        return 0.0
    return float((distance >= SAFE_HAZARD_DISTANCE).mean())


def compute_interaction_frequency(events_df: pd.DataFrame) -> float:
    """Rate of qualifying safety interactions per second of scenario time.

    FIRE: qualifying events are those in
    :data:`~midrr_classifier.data_schema.INTERACTION_EVENT_TYPES`, with one
    exception: ``extinguisher_use``/``ext_spray`` are excluded when
    ``nearby_player_count`` is 0 (BFP rule: *"DO NOT FIGHT FIRE IF ALONE"*).
    Solo extinguisher use is a procedure violation and must not raise the
    frequency score.
    EARTHQUAKE: qualifying events are
    :data:`~midrr_classifier.data_schema.EARTHQUAKE_INTERACTION_EVENT_TYPES`
    (``drop_cover_hold``, ``emergency_exit``, ``assembly_area_reached``).

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Interactions per second (≥ 0).
    """
    duration = compute_evacuation_time(events_df)
    if duration == 0.0:
        return 0.0

    if _is_earthquake_scenario(_scenario_of(events_df)):
        qualifying = events_df["event_type"].isin(EARTHQUAKE_INTERACTION_EVENT_TYPES)
        return float(qualifying.sum() / duration)

    qualifying = events_df["event_type"].isin(INTERACTION_EVENT_TYPES)

    # Remove solo extinguisher interactions (BFP: "DO NOT FIGHT FIRE IF ALONE").
    if "nearby_player_count" in events_df.columns:
        solo_interaction = events_df["event_type"].isin(
            {"extinguisher_use", EXT_SPRAY_EVENT}
        ) & (
            pd.to_numeric(events_df["nearby_player_count"], errors="coerce").fillna(0) == 0
        )
        qualifying = qualifying & ~solo_interaction

    return float(qualifying.sum() / duration)


def compute_resource_utilization(events_df: pd.DataFrame) -> float:
    """PASS-sequencing correctness (fire) or cover discipline (quake).

    FIRE: 1.0 if the player's first ``pin_pull`` precedes their first
    ``ext_spray``/``extinguisher_use`` (Pull before Squeeze/Sweep); 0.0 if
    they sprayed without ever pulling the pin. 1.0 (vacuously correct) if no
    spray was attempted at all — resource_utilization scores *technique*, not
    whether they should have used an extinguisher (that's a rubric/other-
    feature concern). Extinguisher-class-vs-room matching (see
    FEATURE_DEFINITIONS) is NOT yet checked here — it requires a `room_type`
    field the mod doesn't emit yet; TODO once available.

    EARTHQUAKE: 1.0 if the player re-covered on a later shake (≥2
    ``drop_cover_hold`` events, proxy for "re-took cover on aftershock"), 0.5
    if they covered exactly once, 0.0 if they never took cover.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Score in [0, 1].
    """
    if events_df.empty:
        return 0.0

    if _is_earthquake_scenario(_scenario_of(events_df)):
        dch_count = int((events_df["event_type"] == DROP_COVER_HOLD_EVENT).sum())
        if dch_count == 0:
            return 0.0
        return 1.0 if dch_count >= 2 else 0.5

    sprays = events_df[
        events_df["event_type"].isin({EXT_SPRAY_EVENT, "extinguisher_use"})
    ].sort_values("timestamp")
    if sprays.empty:
        return 1.0

    pulls = events_df[events_df["event_type"] == PIN_PULL_EVENT].sort_values("timestamp")
    if pulls.empty:
        return 0.0

    sequencing_ok = float(pulls["timestamp"].iloc[0]) <= float(sprays["timestamp"].iloc[0])
    return 1.0 if sequencing_ok else 0.0


def compute_panic_proxy(events_df: pd.DataFrame) -> float:
    """Standard deviation of movement speed² as a panic proxy.

    REDEFINED (v1.2, was: std-dev of bearing-change turn angles). Computes
    per-tick horizontal (x, z) speed between consecutive ``move`` events,
    squares it, and returns the standard deviation across the session.
    Sudden sprint/stop bursts (panic) raise this value even for players who
    move in a straight line, which the old bearing-only proxy missed.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Standard deviation of speed² (blocks²/s²). Returns 0.0 if fewer than
        3 move events are available.
    """
    moves = events_df[events_df["event_type"] == "move"].sort_values("timestamp")
    if len(moves) < 3:
        return 0.0

    coords = moves[["x", "z"]].to_numpy(dtype=float)
    times = moves["timestamp"].to_numpy(dtype=float)

    step_dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    dt = np.diff(times)
    valid = dt > 0
    if valid.sum() < 2:
        return 0.0

    speed = step_dist[valid] / dt[valid]
    return float(np.std(speed**2))


def compute_situational_awareness(events_df: pd.DataFrame) -> float:
    """Composite [0, 1] "read the situation correctly" score.

    NOT a raw log measurement — combines already-computed signals into one
    summary feature (deliberately correlated with them by design; see
    FEATURE_DEFINITIONS for the SHAP-interpretation caveat).

    FIRE = mean of (alarm pressed? 1 : 0), path_efficiency_ratio,
    (1 / (1 + panic_proxy)) as a bounded "calm" proxy.
    EARTHQUAKE = mean of (drop_cover_hold performed? 1 : 0),
    path_efficiency_ratio, calm proxy, and — only when the player re-covered
    on a later shake (≥2 drop_cover_hold events, the same aftershock-response
    proxy used by compute_resource_utilization) — a 4th "aftershock
    awareness" term of 1.0.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Score in [0, 1].
    """
    if events_df.empty:
        return 0.0

    path_eff = compute_path_efficiency(events_df)
    calm = 1.0 / (1.0 + compute_panic_proxy(events_df))

    if _is_earthquake_scenario(_scenario_of(events_df)):
        dch_count = int((events_df["event_type"] == DROP_COVER_HOLD_EVENT).sum())
        cover_taken = 1.0 if dch_count > 0 else 0.0
        components = [cover_taken, path_eff, calm]
        if dch_count >= 2:
            components.append(1.0)  # re-covered on a later shake
        return float(np.mean(components))

    alarm_pressed = 1.0 if (events_df["event_type"] == "fire_alarm_activate").any() else 0.0
    return float(np.mean([alarm_pressed, path_eff, calm]))


# ---------------------------------------------------------------------------
# Main feature-table builder
# ---------------------------------------------------------------------------


def find_incomplete_sessions(events_df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows belonging to a session with a blank required value.

    A session is incomplete if ANY of its rows is missing a value in
    :data:`~midrr_classifier.data_schema.REQUIRED_VALUE_COLUMNS` — e.g. an
    ``assembly_area_reached`` row with no timestamp. Such a session cannot
    produce trustworthy features (evacuation_time, latency, ... would silently
    become NaN or wrong), so it must be flagged rather than scored.

    Sessions are identified by ``session_id`` when present; rows whose
    identity columns are themselves blank are flagged on their own.
    """
    present = [c for c in REQUIRED_VALUE_COLUMNS if c in events_df.columns]
    bad_row = events_df[present].isna().any(axis=1)
    if "session_id" in events_df.columns and events_df["session_id"].notna().any():
        bad_sessions = set(events_df.loc[bad_row, "session_id"].dropna())
        return bad_row | events_df["session_id"].isin(bad_sessions)
    keys = ["player_id", "scenario_type"]
    bad_keys = set(map(tuple, events_df.loc[bad_row, keys].dropna().to_numpy()))
    in_bad_group = events_df[keys].apply(tuple, axis=1).isin(bad_keys)
    return bad_row | in_bad_group


def build_feature_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw event logs into one feature row per session.

    Applies all ``compute_*`` functions to each session and assembles the
    results into a tidy feature table. A session is one ``session_id`` when
    that column is present (so a student who replays the same scenario gets
    one row per attempt), otherwise one ``(player_id, scenario_type)`` pair
    for legacy logs without session ids.

    Sessions with a blank value in a required column are skipped with a
    warning instead of being scored; their ids are listed in
    ``feature_df.attrs["incomplete_sessions"]``.

    Args:
        events_df: Full raw event log (output of
            :func:`~midrr_classifier.data_ingestion.load_raw_logs`).
            Must contain all columns in
            :data:`~midrr_classifier.data_schema.RAW_LOG_SCHEMA`.

    Returns:
        A :class:`pandas.DataFrame` with one row per session and columns
        matching :data:`~midrr_classifier.data_schema.FEATURE_SCHEMA`, plus
        ``session_id`` (``None`` for legacy logs).
    """
    records: list[dict] = []

    incomplete = find_incomplete_sessions(events_df)
    incomplete_ids: list[str] = []
    if incomplete.any():
        id_col = "session_id" if "session_id" in events_df.columns else "player_id"
        incomplete_ids = sorted(map(str, events_df.loc[incomplete, id_col].dropna().unique()))
        logger.warning(
            "Skipping %d incomplete session(s) with blank required values (%s): %s",
            len(incomplete_ids), ", ".join(REQUIRED_VALUE_COLUMNS), incomplete_ids,
        )
        events_df = events_df[~incomplete]

    has_session_id = "session_id" in events_df.columns
    group_keys = ["player_id", "scenario_type"] + (["session_id"] if has_session_id else [])
    # dropna=False: a legacy row with no session_id must still be scored, not
    # silently vanish from the groupby.
    groups = events_df.groupby(group_keys, sort=False, dropna=False)
    logger.info("Building features for %d session groups.", len(groups))

    for keys, group in groups:
        player_id, scenario_type = keys[0], keys[1]
        session_id = keys[2] if has_session_id and pd.notna(keys[2]) else None
        label = (
            group["preparedness_level"].iloc[0]
            if "preparedness_level" in group.columns
            else None
        )
        label_source = (
            group["label_source"].iloc[0] if "label_source" in group.columns else None
        )

        record: dict = {
            "player_id": player_id,
            "session_id": session_id,
            "scenario_type": scenario_type,
            "decision_latency": compute_decision_latency(group),
            "spray_accuracy": compute_spray_accuracy(group),
            "path_efficiency_ratio": compute_path_efficiency(group),
            "hazard_avoidance_ratio": compute_hazard_avoidance_ratio(group),
            "evacuation_time": compute_evacuation_time(group),
            "interaction_frequency": compute_interaction_frequency(group),
            "resource_utilization": compute_resource_utilization(group),
            "panic_proxy": compute_panic_proxy(group),
            "situational_awareness": compute_situational_awareness(group),
            "preparedness_level": label,
            "label_source": label_source,
        }
        records.append(record)

    feature_df = pd.DataFrame(records)

    # Impossible values (e.g. a negative duration from bad timestamps) must not
    # reach features.csv / training — skip and report, like incomplete sessions.
    out_of_range = find_out_of_range_rows(feature_df)
    invalid_ids: list[str] = []
    if out_of_range:
        for idx, feats in out_of_range.items():
            row = feature_df.loc[idx]
            invalid_ids.append(str(row["session_id"] if row["session_id"] is not None else row["player_id"]))
            logger.warning("Skipping session %s: out-of-range feature(s) %s", invalid_ids[-1], feats)
        feature_df = feature_df.drop(index=list(out_of_range)).reset_index(drop=True)

    feature_df.attrs["incomplete_sessions"] = incomplete_ids
    feature_df.attrs["invalid_sessions"] = invalid_ids
    logger.info("Feature table built: %d rows × %d columns.", *feature_df.shape)
    return feature_df
