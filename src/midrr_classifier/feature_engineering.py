"""Feature engineering pipeline for the MiDRR-Classifier.

Transforms raw per-event gameplay logs into a one-row-per-player
feature table ready for model training.

Each ``compute_*`` function receives the event log for **a single
player in a single simulation run** and returns a scalar float.
The main entry-point is :func:`build_feature_table`, which groups
the full log by ``(player_id, scenario_type)`` and applies all
feature functions.

TODO (research team): Replace the placeholder formulas below with
the exact definitions agreed upon in Chapter 3 of the thesis once
the dataset is finalised.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from midrr_classifier.data_schema import (
    INTERACTION_EVENT_TYPES,
    SAFE_HAZARD_DISTANCE,
)
from midrr_classifier.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-player feature functions
# Each function accepts a DataFrame of events for ONE player × ONE run.
# ---------------------------------------------------------------------------


def compute_evacuation_time(events_df: pd.DataFrame) -> float:
    """Total scenario duration for the player.

    Defined as the difference between the last and first recorded
    timestamps.  A successful evacuation ends with an
    ``emergency_exit`` event; the formula is agnostic to outcome and
    simply measures elapsed time.

    TODO: Decide whether to cap at the scenario time-limit or use the
    actual exit timestamp for successful runs only.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Elapsed seconds (≥ 0).
    """
    if events_df.empty or len(events_df) < 2:
        return 0.0
    return float(events_df["timestamp"].max() - events_df["timestamp"].min())


def compute_decision_delay(events_df: pd.DataFrame) -> float:
    """Latency from the first hazard exposure to the first valid action.

    "First hazard exposure" = first event where ``hazard_distance`` is
    below :data:`~midrr_classifier.data_schema.SAFE_HAZARD_DISTANCE`.
    "First valid action" = first ``emergency_exit``, ``door_open``, or
    ``extinguisher_use`` event after that point.

    TODO: Refine the definition of "valid action" to match the
    scenario-specific rubric from Chapter 3.

    Args:
        events_df: Event log for a single player × run, sorted by
            ``timestamp`` (ascending).

    Returns:
        Delay in seconds.  Returns the total scenario duration if no
        valid action is observed after hazard exposure (worst case).
    """
    df = events_df.sort_values("timestamp")
    hazard_rows = df[df["hazard_distance"] < SAFE_HAZARD_DISTANCE]
    if hazard_rows.empty:
        return 0.0

    hazard_time = hazard_rows["timestamp"].iloc[0]
    post_hazard = df[df["timestamp"] >= hazard_time]
    action_rows = post_hazard[post_hazard["event_type"].isin(INTERACTION_EVENT_TYPES)]

    if action_rows.empty:
        # No action taken — penalise with full scenario duration
        return compute_evacuation_time(df)

    return float(action_rows["timestamp"].iloc[0] - hazard_time)


def compute_path_efficiency(events_df: pd.DataFrame) -> float:
    """Ratio of straight-line displacement to total path length.

    A value close to 1 indicates a direct route to the exit; a value
    close to 0 indicates excessive backtracking or panic-like movement.

    Formula::

        path_efficiency = straight_line_distance / cumulative_path_length

    TODO: Filter to ``move`` events only; include Y-axis in 3-D distance.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Path efficiency ratio in (0, 1].  Returns 1.0 if the player
        did not move (degenerate case).
    """
    moves = events_df[events_df["event_type"] == "move"].sort_values("timestamp")
    if len(moves) < 2:
        return 1.0

    coords = moves[["x", "y", "z"]].to_numpy()

    # Straight-line displacement (start → end)
    straight = float(np.linalg.norm(coords[-1] - coords[0]))

    # Cumulative step-by-step path length
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    total_path = float(steps.sum())

    if total_path == 0.0:
        return 1.0

    return min(straight / total_path, 1.0)  # clamp to [0, 1]


def compute_hazard_avoidance_ratio(events_df: pd.DataFrame) -> float:
    """Fraction of timesteps where the player maintained a safe distance.

    A higher ratio means the player consistently stayed away from
    hazards (fire, debris).

    TODO: Align SAFE_HAZARD_DISTANCE threshold with Chapter 3 rubric.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Ratio in [0, 1].
    """
    if events_df.empty:
        return 0.0
    safe = (events_df["hazard_distance"] >= SAFE_HAZARD_DISTANCE).sum()
    return float(safe / len(events_df))


def compute_interaction_frequency(events_df: pd.DataFrame) -> float:
    """Rate of safety-relevant interactions per second of scenario time.

    Safety interactions are: ``door_open``, ``extinguisher_use``, and
    ``emergency_exit`` (see
    :data:`~midrr_classifier.data_schema.INTERACTION_EVENT_TYPES`).

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Interactions per second (≥ 0).
    """
    duration = compute_evacuation_time(events_df)
    if duration == 0.0:
        return 0.0
    count = events_df["event_type"].isin(INTERACTION_EVENT_TYPES).sum()
    return float(count / duration)


def compute_panic_proxy(events_df: pd.DataFrame) -> float:
    """Standard deviation of consecutive bearing changes as a panic proxy.

    Frequent, large direction changes suggest erratic / panicked
    movement.  We measure the bearing (azimuth) between consecutive
    move events and return the standard deviation of turn angles.

    TODO: Consider also incorporating speed spikes as additional signal.

    Args:
        events_df: Event log for a single player × run.

    Returns:
        Standard deviation of turn angles in degrees.  Returns 0.0 if
        fewer than 3 move events are available.
    """
    moves = events_df[events_df["event_type"] == "move"].sort_values("timestamp")
    if len(moves) < 3:
        return 0.0

    coords = moves[["x", "z"]].to_numpy()  # top-down bearing uses X and Z
    directions = np.diff(coords, axis=0)

    bearings = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    turn_angles = np.abs(np.diff(bearings))
    # Normalise angles to [0, 180]
    turn_angles = np.where(turn_angles > 180, 360 - turn_angles, turn_angles)

    return float(np.std(turn_angles))


# ---------------------------------------------------------------------------
# Main feature-table builder
# ---------------------------------------------------------------------------


def build_feature_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw event logs into one feature row per player per run.

    Applies all ``compute_*`` functions to each ``(player_id,
    scenario_type)`` group and assembles the results into a tidy
    feature table.

    Args:
        events_df: Full raw event log (output of
            :func:`~midrr_classifier.data_ingestion.load_raw_logs`).
            Must contain all columns in
            :data:`~midrr_classifier.data_schema.RAW_LOG_SCHEMA`.

    Returns:
        A :class:`pandas.DataFrame` with one row per player × run and
        columns matching
        :data:`~midrr_classifier.data_schema.FEATURE_SCHEMA`.
    """
    records: list[dict] = []

    groups = events_df.groupby(["player_id", "scenario_type"], sort=False)
    logger.info("Building features for %d player×run groups.", len(groups))

    for (player_id, scenario_type), group in groups:
        label = group["preparedness_level"].iloc[0]

        record: dict = {
            "player_id": player_id,
            "scenario_type": scenario_type,
            "evacuation_time": compute_evacuation_time(group),
            "decision_delay": compute_decision_delay(group),
            "path_efficiency_ratio": compute_path_efficiency(group),
            "hazard_avoidance_ratio": compute_hazard_avoidance_ratio(group),
            "interaction_frequency": compute_interaction_frequency(group),
            "panic_proxy": compute_panic_proxy(group),
            "preparedness_level": label,
        }
        records.append(record)

    feature_df = pd.DataFrame(records)
    logger.info("Feature table built: %d rows × %d columns.", *feature_df.shape)
    return feature_df
