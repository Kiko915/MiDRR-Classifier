"""Data loading and train/test splitting utilities.

Provides thin wrappers that enforce schema validation and keep I/O
concerns out of feature engineering and model training code.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split

from midrr_classifier.data_schema import validate_feature_schema, validate_raw_schema
from midrr_classifier.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Mod v0 emits these names; canonical names are defined in telemetry_contract.md §2/§4.
# Remove an entry here once the mod is updated to emit the canonical name directly.
_SCENARIO_TYPE_ALIASES: dict[str, str] = {
    "ccs_fire": "fire",
    "ccs_earthquake": "earthquake",
}
_EVENT_TYPE_ALIASES: dict[str, str] = {
    "move_tick": "move",
}


def _normalize_raw_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scenario_type"] = df["scenario_type"].replace(_SCENARIO_TYPE_ALIASES)
    df["event_type"] = df["event_type"].replace(_EVENT_TYPE_ALIASES)
    return df


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
    df = _normalize_raw_log(df)
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


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.3,
    stratify_col: str = "preparedness_level",
    group_col: str = "player_id",
    random_state: int = 42,
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

    logger.info(
        "Group-aware split → train: %d rows (%d players), test: %d rows (%d players)",
        len(train_df), len(train_ids),
        len(test_df), len(test_ids),
    )
    return train_df, test_df
