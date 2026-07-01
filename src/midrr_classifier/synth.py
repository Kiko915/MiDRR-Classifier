"""Synthetic gameplay log generator for the MiDRR-Classifier.

**SYNTHETIC DATA — NOT REAL STUDENT SESSIONS.**
All sessions produced here are procedurally generated for pipeline
testing and CI validation.  Labels are ground-truth by construction
(the skill level is set by the caller, so the answer is known).  Do NOT
mix rows from this module with real data without clearly tagging each
row's source column (e.g. ``data_source="synthetic"``).

Events follow telemetry contract v1.2 (``docs/telemetry_contract.md``):
- ``scenario_type`` lowercase (``fire`` / ``earthquake``)
- ``timestamp`` = seconds since disaster trigger (``SIM_START``, t = 0)
- ``move`` events at 20 Hz (every 0.05 s)
- Fire runs emit the 3-phase state machine (``phase_transition``) and the
  PASS-technique sequence (``pin_pull`` -> ``ext_spray`` -> optional
  ``hazard_neutralize``, capped at 5 per session).
- Earthquake runs emit ``drop_cover_hold`` (and a second one to simulate
  re-covering on an aftershock, skill-dependent).
- ``preparedness_level`` is included per row (ground-truth label, set by
  caller).  Real telemetry never carries this column — it is added here
  only so the batch CSV can be fed straight into ``build_feature_table()``.
  ``label_source`` is deliberately NOT set — synthetic labels are ground
  truth by construction, not an expert/rule weak label, and leaving the
  column absent keeps ``split_train_test()``'s expert-only-test-set guard
  a no-op for synthetic data (see ``data_ingestion.split_train_test``).
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np
import pandas as pd

from midrr_classifier.data_schema import (
    DROP_COVER_HOLD_EVENT,
    EXT_SPRAY_EVENT,
    HAZARD_NEUTRALIZE_EVENT,
    LABEL_CLASSES,
    PHASE_TRANSITION_EVENT,
    PIN_PULL_EVENT,
    SAFE_HAZARD_DISTANCE,
    SIM_PHASES,
)

# ---------------------------------------------------------------------------
# Fixed map layout (LSPU-like building, Minecraft X/Z plane, Y=64)
# ---------------------------------------------------------------------------

_SPAWN: tuple[float, float] = (50.0, 50.0)
_HAZARD_ORIGIN: tuple[float, float] = (62.0, 50.0)  # fire start / quake epicentre
_EXIT: tuple[float, float] = (82.0, 80.0)            # emergency exit waypoint
_ASSEMBLY: tuple[float, float] = (120.0, 100.0)      # designated assembly area
_Y: float = 64.0

_DT: float = 0.05                   # seconds per move tick (20 Hz, locked v1.2)
_TIME_LIMIT: float = 180.0          # scenario cap for non-evacuating players
_FIRE_SPREAD_RATE: float = 0.4      # blocks per second
_FIRE_SPREAD_MAX: float = 22.0      # maximum fire radius

# Threshold below which the player is "at" a waypoint (blocks)
_REACH_EXIT_DIST: float = 4.0
_REACH_ASSEMBLY_DIST: float = 6.0

_PHASE_PREVENTION, _PHASE_INTERVENTION, _PHASE_EVACUATION = SIM_PHASES

# Simulated extinguisher class for the single room modeled here. Real-room
# class-matching (EXTINGUISHER_CLASS_BY_ROOM) isn't enforced by
# feature_engineering yet (no room_type field), so any fixed value works.
_EXTINGUISHER_CLASS: str = "CO2"

# Default class mix locked to the BFP-revised simulation design (2026-07-01
# diagrams): 250 sessions total, split 35% HIGH / 45% MODERATE / 20% LOW.
_DEFAULT_N_TOTAL: int = 250
_DEFAULT_CLASS_DISTRIBUTION: dict[str, float] = {
    "HIGH": 0.35,
    "MODERATE": 0.45,
    "LOW": 0.20,
}

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
        # Probability that an extinguisher interaction is solo (BFP violation)
        "solo_ext_prob": 0.05,
        # Probability player reaches assembly area (not all HIGH necessarily do)
        "reaches_assembly_prob": 0.97,
        # --- fire: PASS-technique (pin_pull -> ext_spray) parameters --------
        "ext_engage_prob": 0.55,       # chance they attempt the extinguisher at all
        "pin_pull_correct_prob": 0.95,  # chance pin_pull precedes the first spray
        "spray_hit_prob": 0.85,         # chance each ext_spray lands on the fire
        "spray_count_range": (2, 4),    # number of spray attempts per engagement
        # --- earthquake: Drop-Cover-Hold parameters --------------------------
        "dch_prob": 0.95,        # chance they perform drop_cover_hold at all
        "dch_recover_prob": 0.65,  # chance of a second DCH (re-cover on aftershock)
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
        "ext_engage_prob": 0.55,
        "pin_pull_correct_prob": 0.55,
        "spray_hit_prob": 0.55,
        "spray_count_range": (2, 5),
        "dch_prob": 0.70,
        "dch_recover_prob": 0.30,
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
        "ext_engage_prob": 0.60,
        "pin_pull_correct_prob": 0.15,
        "spray_hit_prob": 0.25,
        "spray_count_range": (3, 7),
        "dch_prob": 0.35,
        "dch_recover_prob": 0.10,
    },
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _is_earthquake(scenario_type: str) -> bool:
    return "earthquake" in scenario_type


def _hazard_dist(px: float, pz: float, t: float, scenario_type: str) -> float:
    """Euclidean distance from player to nearest hazard (simulated)."""
    d2 = math.hypot(px - _HAZARD_ORIGIN[0], pz - _HAZARD_ORIGIN[1])
    if not _is_earthquake(scenario_type):
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
    hit_fire: int | None = None,
    extinguisher_class: str | None = None,
    phase: str | None = None,
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
        "hit_fire": hit_fire,
        "extinguisher_class": extinguisher_class,
        "phase": phase,
        "preparedness_level": skill,
    }


def _emit_pass_sequence(
    rows: list[dict],
    player_id: str,
    session_id: str,
    scenario_type: str,
    t: float,
    px: float,
    pz: float,
    hd: float,
    skill: str,
    prof: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """Emit a pin_pull -> N x ext_spray (-> hazard_neutralize on hits) burst.

    Correct PASS technique = pin_pull before the first spray. Skill controls
    whether that happens (``pin_pull_correct_prob``) and how accurate each
    spray is (``spray_hit_prob``). Player is treated as stationary for the
    duration of the burst (fighting a fire is not a moving activity).
    """
    nearby = 0 if rng.random() < prof["solo_ext_prob"] else 1
    pull_correct = rng.random() < prof["pin_pull_correct_prob"]
    n_sprays = int(rng.integers(*prof["spray_count_range"]))

    cursor_t = t
    if pull_correct:
        rows.append(
            _event(
                player_id, session_id, scenario_type, cursor_t, PIN_PULL_EVENT,
                px, pz, hd, skill, nearby, extinguisher_class=_EXTINGUISHER_CLASS,
            )
        )
        cursor_t += float(rng.uniform(0.3, 1.0))
    # else: technique violation — sprays without ever pulling the pin.

    hits = 0
    for _ in range(n_sprays):
        hit = rng.random() < prof["spray_hit_prob"]
        rows.append(
            _event(
                player_id, session_id, scenario_type, cursor_t, EXT_SPRAY_EVENT,
                px, pz, hd, skill, nearby,
                hit_fire=int(hit), extinguisher_class=_EXTINGUISHER_CLASS,
            )
        )
        if hit:
            hits += 1
        cursor_t += float(rng.uniform(0.4, 1.2))

    for _ in range(min(hits, 5)):
        rows.append(
            _event(player_id, session_id, scenario_type, cursor_t, HAZARD_NEUTRALIZE_EVENT, px, pz, hd, skill)
        )
        cursor_t += 0.2


def _simulate_session(
    player_id: str,
    session_id: str,
    scenario_type: str,
    skill: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Simulate one session for one player.  Returns a list of event dicts."""
    prof = _PROFILE[skill]
    is_quake = _is_earthquake(scenario_type)
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
    if not is_quake:
        rows.append(
            _event(player_id, session_id, scenario_type, t, PHASE_TRANSITION_EVENT, px, pz, hd0, skill, phase=_PHASE_PREVENTION)
        )

    # Flags ----------------------------------------------------------------
    first_action_done = False
    _ed_lo, _ed_hi = prof["extra_door_opens"]
    extra_doors_remaining = int(rng.integers(_ed_lo, _ed_hi + 1)) if _ed_hi > _ed_lo else _ed_lo
    next_door_open_t: float | None = None
    exit_reached = False
    assembly_reached = False
    pending_ext_engage_t: float | None = None
    ext_sequence_done = False
    pending_recover_t: float | None = None

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

        # Normalise and apply speed, with a per-tick speed jitter proportional
        # to composure (noise_std): calm (HIGH) players keep a near-constant
        # pace; panicked (LOW) players alternate bursts of sprinting and
        # near-freezing. Without this, every tick's step length collapses to
        # exactly `speed * _DT` regardless of skill, and panic_proxy (std-dev
        # of speed^2, redefined in Phase 2.5 step 2) carries no real signal.
        speed_jitter = max(0.1, float(rng.normal(1.0, noise_std / 100.0)))
        v_mag = math.hypot(vx, vz)
        if v_mag > 1e-9:
            step = speed * speed_jitter * _DT
            vx = vx / v_mag * step
            vz = vz / v_mag * step

        px = round(px + vx, 4)
        pz = round(pz + vz, 4)
        hd = _hazard_dist(px, pz, t, scenario_type)

        # ---- Emit move event -----------------------------------------------
        rows.append(_event(player_id, session_id, scenario_type, t, "move", px, pz, hd, skill))

        # ---- First safety action at decision_delay -------------------------
        if not first_action_done and t >= decision_delay:
            first_action_done = True
            if is_quake:
                if rng.random() < prof["dch_prob"]:
                    rows.append(_event(player_id, session_id, scenario_type, t, DROP_COVER_HOLD_EVENT, px, pz, hd, skill))
                    if rng.random() < prof["dch_recover_prob"]:
                        pending_recover_t = t + float(rng.uniform(15, 40))
                # else: player never takes cover — decision_latency falls back
                # to the worst-case penalty in feature_engineering, matching
                # rubric E1 ("ran/froze in danger during shaking" = 0 score).
            else:
                action_type = "fire_alarm_activate" if prof["use_alarm"] else "door_open"
                rows.append(_event(player_id, session_id, scenario_type, t, action_type, px, pz, hd, skill))
                rows.append(
                    _event(player_id, session_id, scenario_type, t, PHASE_TRANSITION_EVENT, px, pz, hd, skill, phase=_PHASE_INTERVENTION)
                )
                if rng.random() < prof["ext_engage_prob"]:
                    pending_ext_engage_t = t + float(rng.uniform(1.0, 3.0))

            # Schedule first extra door_open 3–8 s after first action
            if extra_doors_remaining > 0:
                next_door_open_t = t + float(rng.uniform(3, 8))

        # ---- Fire: PASS-technique burst (pin_pull -> ext_spray -> neutralize) --
        if (
            not is_quake
            and pending_ext_engage_t is not None
            and not ext_sequence_done
            and t >= pending_ext_engage_t
        ):
            _emit_pass_sequence(rows, player_id, session_id, scenario_type, t, px, pz, hd, skill, prof, rng)
            ext_sequence_done = True

        # ---- Earthquake: re-cover on a later shake (aftershock proxy) ----------
        if is_quake and pending_recover_t is not None and t >= pending_recover_t:
            rows.append(_event(player_id, session_id, scenario_type, t, DROP_COVER_HOLD_EVENT, px, pz, hd, skill))
            pending_recover_t = None

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
                if not is_quake:
                    rows.append(
                        _event(player_id, session_id, scenario_type, t, PHASE_TRANSITION_EVENT, px, pz, hd, skill, phase=_PHASE_EVACUATION)
                    )

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


def _allocate_class_counts(n_total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split *n_total* across ``LABEL_CLASSES`` per *weights*, summing exactly.

    Uses the largest-remainder method so rounding never loses or invents a
    session (``sum(result.values()) == n_total``).
    """
    raw = {label: n_total * weights[label] for label in LABEL_CLASSES}
    floors = {label: int(math.floor(v)) for label, v in raw.items()}
    remainder = n_total - sum(floors.values())
    by_fraction = sorted(LABEL_CLASSES, key=lambda label: raw[label] - floors[label], reverse=True)
    for label in by_fraction[:remainder]:
        floors[label] += 1
    return floors


def _split_across_scenarios(count: int, n_scenarios: int) -> list[int]:
    """Split *count* as evenly as possible across *n_scenarios* buckets."""
    base, rem = divmod(count, n_scenarios)
    return [base + (1 if i < rem else 0) for i in range(n_scenarios)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_logs(
    n_per_class: int | None = None,
    n_total: int = _DEFAULT_N_TOTAL,
    class_distribution: dict[str, float] | None = None,
    seed: int = 42,
    scenario_types: tuple[str, ...] = ("fire", "earthquake"),
) -> pd.DataFrame:
    """Generate a synthetic raw gameplay log suitable for MiDRR training.

    Default behavior (no *n_per_class*) locks in the BFP-revised simulation
    design's class mix: **250 sessions total, ~35% HIGH / 45% MODERATE / 20%
    LOW**, split evenly across *scenario_types*. Pass *n_per_class* for the
    old uniform behavior (equal count per class per scenario) — e.g. useful
    in notebooks/tests that want a small balanced sample regardless of the
    locked real-world class mix.

    Args:
        n_per_class: If given, generate exactly this many sessions per
            ``(preparedness_level, scenario_type)`` combination (uniform,
            legacy behavior). Overrides *n_total*/*class_distribution*.
        n_total: Total sessions across all classes and scenario types
            combined, when *n_per_class* is not given. Defaults to 250.
        class_distribution: Fraction of *n_total* per class. Defaults to
            ``{"HIGH": 0.35, "MODERATE": 0.45, "LOW": 0.20}``. Ignored when
            *n_per_class* is given.
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
        df = generate_logs(seed=0)  # 250 sessions @ 35/45/20
        print(df.drop_duplicates("session_id")["preparedness_level"].value_counts())
    """
    rng = np.random.default_rng(seed)
    all_rows: list[dict] = []

    if n_per_class is not None:
        per_class_per_scenario = {
            skill: {scenario: n_per_class for scenario in scenario_types} for skill in LABEL_CLASSES
        }
    else:
        weights = class_distribution or _DEFAULT_CLASS_DISTRIBUTION
        class_counts = _allocate_class_counts(n_total, weights)
        per_class_per_scenario = {
            skill: dict(zip(scenario_types, _split_across_scenarios(class_counts[skill], len(scenario_types))))
            for skill in LABEL_CLASSES
        }

    for scenario in scenario_types:
        for skill in LABEL_CLASSES:  # HIGH, MODERATE, LOW
            n = per_class_per_scenario[skill][scenario]
            for i in range(n):
                player_id = f"synth_{skill[:2]}_{scenario[:2]}_{i:03d}"
                hi = int(rng.integers(0, 2**63))
                lo = int(rng.integers(0, 2**63))
                session_id = str(uuid.UUID(int=(hi << 63) | lo))
                rows = _simulate_session(player_id, session_id, scenario, skill, rng)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df["data_source"] = "synthetic"  # provenance tag — filter this out before real training
    return df
