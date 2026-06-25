"""Synthetic gameplay log generator for the MiDRR-Classifier.

**SYNTHETIC DATA — NOT REAL STUDENT SESSIONS.**
All sessions produced here are procedurally generated for pipeline
testing and CI validation.  Labels are ground-truth by construction
(the skill level is set by the caller, so the answer is known).  Do NOT
mix rows from this module with real data without clearly tagging each
row's source column (e.g. ``data_source="synthetic"``).

Events follow telemetry contract v1.1 (``docs/telemetry_contract.md``):
- ``scenario_type`` lowercase (``fire`` / ``earthquake``)
- ``timestamp`` = seconds since disaster trigger (t = 0)
- ``move`` events at 10 Hz (every 0.1 s)
- ``preparedness_level`` is included per row (ground-truth label, set by
  caller).  Real telemetry never carries this column — it is added here
  only so the batch CSV can be fed straight into ``build_feature_table()``.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np
import pandas as pd

from midrr_classifier.data_schema import LABEL_CLASSES, SAFE_HAZARD_DISTANCE

# ---------------------------------------------------------------------------
# Fixed map layout (LSPU-like building, Minecraft X/Z plane, Y=64)
# ---------------------------------------------------------------------------

_SPAWN: tuple[float, float] = (50.0, 50.0)
_HAZARD_ORIGIN: tuple[float, float] = (62.0, 50.0)  # fire start / quake epicentre
_EXIT: tuple[float, float] = (82.0, 80.0)            # emergency exit waypoint
_ASSEMBLY: tuple[float, float] = (120.0, 100.0)      # designated assembly area
_Y: float = 64.0

_DT: float = 0.1                    # seconds per move tick (10 Hz)
_TIME_LIMIT: float = 180.0          # scenario cap for non-evacuating players
_FIRE_SPREAD_RATE: float = 0.4      # blocks per second
_FIRE_SPREAD_MAX: float = 22.0      # maximum fire radius

# Threshold below which the player is "at" a waypoint (blocks)
_REACH_EXIT_DIST: float = 4.0
_REACH_ASSEMBLY_DIST: float = 6.0

# ---------------------------------------------------------------------------
# Skill profiles
# Keep ranges non-overlapping for HIGH vs LOW to produce separable data.
# ---------------------------------------------------------------------------

_PROFILE: dict[str, dict[str, Any]] = {
    "HIGH": {
        # How long before first safety action (seconds from t=0)
        "decision_delay_range": (1.5, 5.0),
        # Speed (blocks/second) during evacuation phases
        "speed_range": (4.0, 5.0),
        # Velocity direction noise (degrees std-dev) — low = calm / direct
        "noise_std": 5.0,
        # Weight of pull toward current target (0=ignore target, 1=beeline)
        "target_pull": 0.88,
        # Probability that player drifts toward hazard during pre-action phase
        "hazard_attraction": 0.02,
        # Emit fire_alarm_activate as first action (COMMUNICATE step)
        "use_alarm": True,
        # Additional door_open events after first action
        "extra_door_opens": (1, 2),
        # Probability that extinguisher_use is solo (BFP violation)
        "solo_ext_prob": 0.05,
        # Probability player reaches assembly area (not all HIGH necessarily do)
        "reaches_assembly_prob": 0.97,
    },
    "MODERATE": {
        "decision_delay_range": (6.0, 18.0),
        "speed_range": (3.0, 4.0),
        "noise_std": 25.0,
        "target_pull": 0.60,
        "hazard_attraction": 0.18,
        "use_alarm": False,
        "extra_door_opens": (0, 1),
        "solo_ext_prob": 0.35,
        "reaches_assembly_prob": 0.75,
    },
    "LOW": {
        "decision_delay_range": (18.0, 55.0),
        "speed_range": (2.0, 3.0),
        "noise_std": 60.0,
        "target_pull": 0.28,
        "hazard_attraction": 0.45,
        "use_alarm": False,
        "extra_door_opens": (0, 0),
        "solo_ext_prob": 0.75,
        "reaches_assembly_prob": 0.35,
    },
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _hazard_dist(px: float, pz: float, t: float, scenario_type: str) -> float:
    """Euclidean distance from player to nearest hazard (simulated)."""
    d2 = math.hypot(px - _HAZARD_ORIGIN[0], pz - _HAZARD_ORIGIN[1])
    if scenario_type == "fire":
        spread = min(t * _FIRE_SPREAD_RATE, _FIRE_SPREAD_MAX)
        return max(0.0, d2 - spread)
    # Earthquake: distance to epicentre only (debris zone, no spread)
    return d2


def _rotate(dx: float, dz: float, angle_rad: float) -> tuple[float, float]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return dx * c - dz * s, dx * s + dz * c


def _event(
    player_id: str,
    session_id: str,
    scenario_type: str,
    t: float,
    event_type: str,
    px: float,
    pz: float,
    hd: float,
    skill: str,
    nearby_player_count: int | None = None,
) -> dict:
    return {
        "player_id": player_id,
        "session_id": session_id,
        "scenario_type": scenario_type,
        "timestamp": round(t, 3),
        "event_type": event_type,
        "x": round(px, 2),
        "y": _Y,
        "z": round(pz, 2),
        "hazard_distance": round(hd, 3),
        "nearby_player_count": nearby_player_count,
        "preparedness_level": skill,
    }


def _simulate_session(
    player_id: str,
    session_id: str,
    scenario_type: str,
    skill: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Simulate one session for one player.  Returns a list of event dicts."""
    prof = _PROFILE[skill]
    decision_delay = float(rng.uniform(*prof["decision_delay_range"]))
    speed = float(rng.uniform(*prof["speed_range"]))
    noise_std = prof["noise_std"]
    pull = prof["target_pull"]

    # Does this player eventually reach the assembly area?
    will_reach = rng.random() < prof["reaches_assembly_prob"]

    px, pz = _SPAWN
    vx, vz = 0.0, 0.0
    t = 0.0

    rows: list[dict] = []

    # session_start --------------------------------------------------------
    hd0 = _hazard_dist(px, pz, t, scenario_type)
    rows.append(_event(player_id, session_id, scenario_type, t, "session_start", px, pz, hd0, skill))

    # Flags ----------------------------------------------------------------
    first_action_done = False
    _ed_lo, _ed_hi = prof["extra_door_opens"]
    extra_doors_remaining = int(rng.integers(_ed_lo, _ed_hi + 1)) if _ed_hi > _ed_lo else _ed_lo
    next_door_open_t: float | None = None
    exit_reached = False
    assembly_reached = False

    # Wander target for pre-action phase
    wander_x, wander_z = float(rng.normal(_SPAWN[0], 4)), float(rng.normal(_SPAWN[1], 4))
    wander_refresh_t = 2.0  # seconds between wander target refreshes

    while t < _TIME_LIMIT and not assembly_reached:
        t = round(t + _DT, 6)

        # ---- Refresh wander target ----------------------------------------
        if t >= wander_refresh_t and not first_action_done:
            wander_refresh_t = t + 2.0
            if rng.random() < prof["hazard_attraction"]:
                # Drift toward hazard
                wander_x = float(rng.normal(_HAZARD_ORIGIN[0], 3))
                wander_z = float(rng.normal(_HAZARD_ORIGIN[1], 3))
            else:
                wander_x = float(rng.normal(_SPAWN[0], 5))
                wander_z = float(rng.normal(_SPAWN[1], 5))

        # ---- Determine target for this tick --------------------------------
        if not first_action_done:
            tx, tz = wander_x, wander_z
        elif not exit_reached:
            tx, tz = _EXIT
        else:
            tx, tz = _ASSEMBLY

        # ---- Compute desired direction with noise --------------------------
        ddx, ddz = tx - px, tz - pz
        dist_to_target = math.hypot(ddx, ddz)
        if dist_to_target > 0:
            ddx /= dist_to_target
            ddz /= dist_to_target

        noise_rad = float(rng.normal(0, math.radians(noise_std)))
        ddx, ddz = _rotate(ddx, ddz, noise_rad)

        # Blend with current velocity (momentum)
        vx = pull * ddx + (1.0 - pull) * vx
        vz = pull * ddz + (1.0 - pull) * vz

        # Normalise and apply speed
        v_mag = math.hypot(vx, vz)
        if v_mag > 1e-9:
            step = speed * _DT
            vx = vx / v_mag * step
            vz = vz / v_mag * step

        px = round(px + vx, 4)
        pz = round(pz + vz, 4)
        hd = _hazard_dist(px, pz, t, scenario_type)

        # ---- Emit move event -----------------------------------------------
        rows.append(_event(player_id, session_id, scenario_type, t, "move", px, pz, hd, skill))

        # ---- First safety action at decision_delay -------------------------
        if not first_action_done and t >= decision_delay:
            if prof["use_alarm"] and scenario_type == "fire":
                action_type = "fire_alarm_activate"
                nearby = None
            elif rng.random() < 0.5:
                action_type = "door_open"
                nearby = None
            else:
                action_type = "extinguisher_use"
                nearby = 0 if rng.random() < prof["solo_ext_prob"] else 1
            rows.append(_event(player_id, session_id, scenario_type, t, action_type, px, pz, hd, skill, nearby))
            first_action_done = True
            # Schedule first extra door_open 3–8 s after first action
            if extra_doors_remaining > 0:
                next_door_open_t = t + float(rng.uniform(3, 8))

        # ---- Extra door_open events ----------------------------------------
        if (
            first_action_done
            and extra_doors_remaining > 0
            and next_door_open_t is not None
            and t >= next_door_open_t
        ):
            rows.append(_event(player_id, session_id, scenario_type, t, "door_open", px, pz, hd, skill))
            extra_doors_remaining -= 1
            next_door_open_t = t + float(rng.uniform(4, 10)) if extra_doors_remaining > 0 else None

        # ---- Waypoint: emergency exit --------------------------------------
        if first_action_done and not exit_reached:
            if math.hypot(px - _EXIT[0], pz - _EXIT[1]) < _REACH_EXIT_DIST:
                rows.append(_event(player_id, session_id, scenario_type, t, "emergency_exit", px, pz, hd, skill))
                exit_reached = True

        # ---- Waypoint: assembly area ---------------------------------------
        if exit_reached and not assembly_reached:
            dist_to_assembly = math.hypot(px - _ASSEMBLY[0], pz - _ASSEMBLY[1])
            if dist_to_assembly < _REACH_ASSEMBLY_DIST:
                if will_reach:
                    rows.append(_event(player_id, session_id, scenario_type, t, "assembly_area_reached", px, pz, hd, skill))
                    assembly_reached = True

    # ---- session_end -------------------------------------------------------
    hd_end = _hazard_dist(px, pz, t, scenario_type)
    rows.append(_event(player_id, session_id, scenario_type, t, "session_end", px, pz, hd_end, skill))

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_logs(
    n_per_class: int = 25,
    seed: int = 42,
    scenario_types: tuple[str, ...] = ("fire", "earthquake"),
) -> pd.DataFrame:
    """Generate a synthetic raw gameplay log suitable for MiDRR training.

    Each skill level × scenario combination produces ``n_per_class`` sessions.
    Total rows ≈ ``n_per_class × 3 classes × len(scenario_types) × ~avg_ticks``.

    Args:
        n_per_class: Number of simulated students per
            ``(preparedness_level, scenario_type)`` combination.
        seed: NumPy random seed for reproducibility.
        scenario_types: Scenarios to generate. Defaults to both
            ``"fire"`` and ``"earthquake"``.

    Returns:
        A :class:`pandas.DataFrame` with columns matching
        :data:`~midrr_classifier.data_schema.RAW_LOG_SCHEMA` plus
        ``session_id`` (join key) and ``data_source="synthetic"``
        (provenance tag).

    Example::

        from midrr_classifier.synth import generate_logs
        df = generate_logs(n_per_class=20, seed=0)
        print(df["preparedness_level"].value_counts())
    """
    rng = np.random.default_rng(seed)
    all_rows: list[dict] = []

    for scenario in scenario_types:
        for skill in LABEL_CLASSES:  # HIGH, MODERATE, LOW
            for i in range(n_per_class):
                player_id = f"synth_{skill[:2]}_{scenario[:2]}_{i:03d}"
                hi = int(rng.integers(0, 2**63))
                lo = int(rng.integers(0, 2**63))
                session_id = str(uuid.UUID(int=(hi << 63) | lo))
                rows = _simulate_session(player_id, session_id, scenario, skill, rng)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df["data_source"] = "synthetic"  # provenance tag — filter this out before real training
    return df
