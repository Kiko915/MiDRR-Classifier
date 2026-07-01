"""Data loading and train/test splitting utilities.

Provides thin wrappers that enforce schema validation and keep I/O
concerns out of feature engineering and model training code.
"""

from __future__ import annotations

import io
import json
import os
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split

from midrr_classifier.data_schema import (
    RAW_LOG_SCHEMA,
    validate_feature_schema,
    validate_raw_schema,
)
from midrr_classifier.labeling import attach_labels
from midrr_classifier.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from midrr_classifier.config import MiDRRConfig

logger = get_logger(__name__)

# Mod v0 emits these event_type names; canonical names are defined in telemetry_contract.md §4.
# scenario_type has NO aliases — ccs_fire / ccs_earthquake / fire / earthquake are all
# distinct canonical values (different buildings, different assembly zones).
# Remove an entry here once the mod is updated to emit the canonical name directly.
_SCENARIO_TYPE_ALIASES: dict[str, str] = {}
_EVENT_TYPE_ALIASES: dict[str, str] = {
    "move_tick": "move",
}


def normalize_raw_log(df: pd.DataFrame) -> pd.DataFrame:
    """Map mod v0 event/scenario names to canonical contract names.

    Safe to call on already-normalized data — aliases that are not present
    are silently ignored by ``DataFrame.replace``.
    """
    df = df.copy()
    df["scenario_type"] = df["scenario_type"].replace(_SCENARIO_TYPE_ALIASES)
    df["event_type"] = df["event_type"].replace(_EVENT_TYPE_ALIASES)
    return df


def resolve_session_labels(
    df: pd.DataFrame,
    expert_col: str = "expert_label",
    rule_label_col: str = "prep_level",
    rule_score_col: str = "simulation_score",
) -> pd.DataFrame:
    """Attach ``preparedness_level``/``label_source`` before feature building.

    Thin wrapper around :func:`~midrr_classifier.labeling.attach_labels` —
    the single call site every session source (CSV batch, and the Turso
    ``sessions`` adapter landing in Phase 2.5 step 4) should route through
    so expert-vs-rule label precedence is applied consistently. See
    ``docs/labeling_rubric.md`` §7 for the expert-gold / rule-weak-label
    policy this implements.

    Args:
        df: Session-level (or raw-log-with-broadcast-session-columns)
            DataFrame. Any of *expert_col*/*rule_label_col*/*rule_score_col*
            may be absent.

    Returns:
        A copy of *df* with ``preparedness_level`` and ``label_source`` set.
    """
    return attach_labels(
        df,
        expert_col=expert_col,
        rule_label_col=rule_label_col,
        rule_score_col=rule_score_col,
    )


def load_raw_logs(path: str) -> pd.DataFrame:
    """Load raw gameplay event logs from a CSV file.

    The CSV must contain all columns defined in
    :data:`~midrr_classifier.data_schema.RAW_LOG_SCHEMA`.

    Args:
        path: Absolute or relative path to the raw CSV file.

    Returns:
        A :class:`pandas.DataFrame` with the raw event rows.

    Raises:
        FileNotFoundError: If the CSV does not exist at *path*.
        ValueError: If required columns are missing (schema validation).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw log file not found: {path}")

    logger.info("Loading raw logs from %s", path)
    df = pd.read_csv(path)
    df = normalize_raw_log(df)
    validate_raw_schema(df)
    logger.info("Loaded %d event rows from %s", len(df), path)
    return df


def load_feature_table(path: str) -> pd.DataFrame:
    """Load a pre-built feature table from a CSV file.

    The CSV must contain all columns defined in
    :data:`~midrr_classifier.data_schema.FEATURE_SCHEMA`.

    Args:
        path: Absolute or relative path to the processed feature CSV.

    Returns:
        A :class:`pandas.DataFrame` with one row per player per run.

    Raises:
        FileNotFoundError: If the CSV does not exist at *path*.
        ValueError: If required columns are missing (schema validation).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature table not found: {path}")

    logger.info("Loading feature table from %s", path)
    df = pd.read_csv(path)
    validate_feature_schema(df)
    logger.info("Loaded %d feature rows from %s", len(df), path)
    return df


def _parse_move_log_csv(move_log_csv: Any) -> pd.DataFrame:
    """Parse a Turso ``sessions.move_log_csv`` cell into per-tick move rows."""
    if not isinstance(move_log_csv, str) or not move_log_csv.strip():
        return pd.DataFrame()
    moves = pd.read_csv(io.StringIO(move_log_csv))
    if "event_type" not in moves.columns:
        moves["event_type"] = "move"
    return moves


def _parse_event_log(event_log: Any) -> pd.DataFrame:
    """Parse a Turso ``sessions.event_log`` cell (JSON array) into event rows."""
    if event_log is None or (isinstance(event_log, float) and pd.isna(event_log)):
        return pd.DataFrame()
    events = event_log if isinstance(event_log, list) else json.loads(event_log)
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events)


def _explode_session_row(row: "pd.Series[Any]") -> pd.DataFrame:
    """Turn one label-resolved Turso ``sessions`` row into long-format raw-log rows.

    Combines the per-event stream (``event_log``, JSON) with the per-tick
    move stream (``move_log_csv``, embedded CSV) and broadcasts the
    session-level identity/label columns onto every resulting row.
    """
    combined = pd.concat(
        [_parse_event_log(row.get("event_log")), _parse_move_log_csv(row.get("move_log_csv"))],
        ignore_index=True,
        sort=False,
    )
    if combined.empty:
        return combined

    combined["session_id"] = row.get("session_id")
    combined["player_id"] = row.get("student_name")
    combined["scenario_type"] = row.get("simulation_type")
    combined["preparedness_level"] = row.get("preparedness_level")
    combined["label_source"] = row.get("label_source")
    return combined


def _sessions_table_to_raw_log(sessions_df: pd.DataFrame, expert_col: str) -> pd.DataFrame:
    """Convert a Turso ``sessions`` table (§3b) into a validated raw-log DataFrame.

    Resolves labels first (one decision per session), then explodes each
    row's ``event_log``/``move_log_csv`` into the long-format rows
    :func:`~midrr_classifier.feature_engineering.build_feature_table` expects.
    """
    sessions_df = resolve_session_labels(
        sessions_df,
        expert_col=expert_col,
        rule_label_col="prep_level",
        rule_score_col="simulation_score",
    )

    exploded = [_explode_session_row(row) for _, row in sessions_df.iterrows()]
    exploded = [frame for frame in exploded if not frame.empty]
    if not exploded:
        return pd.DataFrame(columns=list(RAW_LOG_SCHEMA.keys()))

    raw_df = pd.concat(exploded, ignore_index=True, sort=False)

    # Some event types legitimately carry neither hazard_distance nor
    # nearby_player_count (e.g. phase_transition) — ensure the columns exist
    # so validate_raw_schema() checks presence, not per-row population.
    for col in RAW_LOG_SCHEMA:
        if col not in raw_df.columns:
            raw_df[col] = pd.NA

    raw_df = normalize_raw_log(raw_df)
    validate_raw_schema(raw_df)
    logger.info(
        "Turso ingestion: %d sessions exploded into %d raw-log rows.",
        len(sessions_df), len(raw_df),
    )
    return raw_df


def load_sessions_from_turso(
    database_url: str,
    auth_token: str | None = None,
    expert_col: str = "expert_label",
    query: str = "SELECT * FROM sessions",
) -> pd.DataFrame:
    """Read the Turso ``sessions`` table (libSQL) and return a raw-log DataFrame.

    Live counterpart to :func:`load_raw_logs` — see
    ``docs/telemetry_contract.md`` §3b for the `sessions` schema this reads
    (``student_name``, ``simulation_type``, ``event_log``, ``move_log_csv``,
    ``simulation_score``, ``passed``, ``prep_level``, ``confidence``, plus
    whatever column carries the BFP-instructor override, named *expert_col*).

    Requires the optional ``libsql-client`` dependency
    (``pip install libsql-client`` or ``pip install -e ".[turso]"``) — only
    imported here so CSV-only usage never needs it installed.

    Args:
        database_url: Turso/libSQL connection URL
            (``config.turso_database_url``).
        auth_token: Turso auth token (``config.turso_auth_token``).
        expert_col: Column on the `sessions` table carrying the
            BFP-instructor override label, if present.
        query: Override to filter/limit which sessions are pulled
            (e.g. restrict to a batch or date range).

    Returns:
        A raw-log DataFrame that has passed :func:`~midrr_classifier.
        data_schema.validate_raw_schema`, with ``preparedness_level`` and
        ``label_source`` already resolved per session.
    """
    try:
        import libsql_client
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "Turso ingestion requires the optional 'libsql-client' package. "
            "Install with `pip install libsql-client` or `pip install -e \".[turso]\"`."
        ) from exc

    client = libsql_client.create_client_sync(url=database_url, auth_token=auth_token)
    try:
        result = client.execute(query)
        sessions_df = pd.DataFrame(result.rows, columns=result.columns)
    finally:
        client.close()

    logger.info("Loaded %d session rows from Turso.", len(sessions_df))
    return _sessions_table_to_raw_log(sessions_df, expert_col=expert_col)


def load_sessions(
    source: str,
    *,
    csv_path: str | None = None,
    sessions_csv_path: str | None = None,
    database_url: str | None = None,
    auth_token: str | None = None,
    config: "MiDRRConfig | None" = None,
    expert_col: str = "expert_label",
) -> pd.DataFrame:
    """Single entry point for raw-log ingestion — CSV batch or live Turso.

    Both backends return the same raw-log shape (validated against
    :data:`~midrr_classifier.data_schema.RAW_LOG_SCHEMA`) with labels
    already resolved through :func:`resolve_session_labels`, so callers
    (:func:`~midrr_classifier.feature_engineering.build_feature_table`) never
    need to know which transport the data came from.

    Args:
        source: ``"csv"`` or ``"turso"``.
        csv_path: Path to the batched raw-log CSV (``source="csv"``,
            required). See ``docs/telemetry_contract.md`` §3a.
        sessions_csv_path: Optional companion ``sessions_<batch>.csv`` (§5)
            carrying per-session ``prep_level``/``simulation_score``/
            *expert_col*, joined on ``session_id`` to resolve labels. If
            omitted, any ``preparedness_level`` already present in
            *csv_path* is used as-is (fully backward compatible).
        database_url: Turso connection URL (``source="turso"``). Falls back
            to ``config.turso_database_url`` if *config* is given.
        auth_token: Turso auth token. Falls back to
            ``config.turso_auth_token``.
        config: Optional :class:`~midrr_classifier.config.MiDRRConfig` to
            source Turso credentials from instead of passing them directly.
        expert_col: Column carrying the BFP-instructor override label.

    Returns:
        A validated raw-log DataFrame.

    Raises:
        ValueError: If *source* is unrecognized, or a required path/URL is
            missing for the chosen source.
    """
    if source == "csv":
        if csv_path is None:
            raise ValueError("source='csv' requires csv_path.")
        raw_df = load_raw_logs(csv_path)
        if sessions_csv_path is not None:
            sessions_df = pd.read_csv(sessions_csv_path)
            sessions_df = resolve_session_labels(sessions_df, expert_col=expert_col)
            raw_df = raw_df.merge(
                sessions_df[["session_id", "preparedness_level", "label_source"]],
                on="session_id",
                how="left",
                suffixes=("", "_resolved"),
            )
            for col in ("preparedness_level", "label_source"):
                if f"{col}_resolved" in raw_df.columns:
                    raw_df[col] = raw_df[f"{col}_resolved"]
                    raw_df = raw_df.drop(columns=[f"{col}_resolved"])
        return raw_df

    if source == "turso":
        url = database_url or (config.turso_database_url if config else None)
        token = auth_token or (config.turso_auth_token if config else None)
        if not url:
            raise ValueError(
                "source='turso' requires database_url (or config.turso_database_url)."
            )
        return load_sessions_from_turso(url, token, expert_col=expert_col)

    raise ValueError(f"Unknown source '{source}'. Expected 'csv' or 'turso'.")


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.3,
    stratify_col: str = "preparedness_level",
    group_col: str = "player_id",
    random_state: int = 42,
    label_source_col: str = "label_source",
    enforce_expert_only_test: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware stratified split of a feature table.

    **Why group-aware?**
    A student (``player_id``) may have sessions for both FIRE and
    EARTHQUAKE.  A naive row-level split could place that student's fire
    session in train and earthquake session in test.  The model would
    then see the student's behavioural signature during training, which
    inflates test-set performance and prevents generalisation to new
    students — the actual deployment scenario.

    This function splits at the **player level**: every row belonging to
    a given ``player_id`` goes entirely into train or entirely into test.

    Stratification is performed on each player's modal
    ``preparedness_level`` so that class proportions are approximately
    preserved in both splits despite the group constraint.

    **Circularity guard (labeling_rubric.md §7):** if *label_source_col* is
    present and carries real values, any row landing in the test split that
    is NOT ``label_source="expert"`` is dropped from the test set (never
    moved to train — that would leak the player into both splits). Rows
    with no ``label_source`` info at all (legacy data, or the column
    absent) are left untouched, so this is fully backward compatible.

    Args:
        df: Feature table (output of
            :func:`~midrr_classifier.feature_engineering.build_feature_table`
            or loaded via :func:`load_feature_table`).
        test_size: Fraction of **players** (not rows) to place in the
            test set.
        stratify_col: Column used for class-balanced sampling.
            Must be present in *df*.
        group_col: Column whose values define the groups that must not
            span train and test.  Defaults to ``"player_id"``.
        random_state: Seed for reproducibility.
        label_source_col: Column identifying ``"expert"`` vs ``"rule"``
            labeled rows (see ``labeling.py``).
        enforce_expert_only_test: If ``True`` (default) and *label_source_col*
            carries real (non-null) values, drop non-expert rows from the
            test split.

    Returns:
        A ``(train_df, test_df)`` tuple.  No ``player_id`` appears in
        both splits.

    Raises:
        ValueError: If *stratify_col* or *group_col* is absent from *df*.
        ValueError: If any class has fewer groups than required for a
            stratified split (mirrors sklearn behaviour).
    """
    for col in (stratify_col, group_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # One row per player: use modal label for stratification.
    group_summary = (
        df.groupby(group_col)[stratify_col]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
        .rename(columns={stratify_col: "_strat_label"})
    )

    train_groups, test_groups = _sklearn_split(
        group_summary[group_col],
        test_size=test_size,
        stratify=group_summary["_strat_label"],
        random_state=random_state,
    )

    train_ids: set = set(train_groups)
    test_ids: set = set(test_groups)

    train_df = df[df[group_col].isin(train_ids)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_ids)].reset_index(drop=True)

    has_label_source_info = (
        label_source_col in df.columns and df[label_source_col].notna().any()
    )
    if enforce_expert_only_test and has_label_source_info:
        pre_count = len(test_df)
        test_df = test_df[test_df[label_source_col] == "expert"].reset_index(drop=True)
        dropped = pre_count - len(test_df)
        if dropped:
            logger.warning(
                "Dropped %d non-expert-labeled row(s) from the test split "
                "(circularity guard — labeling_rubric.md §7: test set must be expert-only).",
                dropped,
            )
        orphaned_players = test_ids - set(test_df[group_col])
        if orphaned_players:
            logger.warning(
                "%d test-split player(s) had no expert-labeled rows and "
                "contributed zero rows to the final test set: %s",
                len(orphaned_players), sorted(orphaned_players),
            )

    logger.info(
        "Group-aware split -> train: %d rows (%d players), test: %d rows (%d players)",
        len(train_df), len(train_ids),
        len(test_df), test_df[group_col].nunique(),
    )
    return train_df, test_df
