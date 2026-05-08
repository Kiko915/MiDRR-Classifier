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
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature table into stratified train and test sets.

    Stratification ensures that the class distribution of
    ``preparedness_level`` is preserved in both splits, which is
    important given the relatively small expected sample size.

    Args:
        df: Feature table (output of
            :func:`~midrr_classifier.feature_engineering.build_feature_table`
            or loaded via :func:`load_feature_table`).
        test_size: Fraction of rows to place in the test set.
        stratify_col: Column to use for stratified sampling.
        random_state: Seed for reproducibility.

    Returns:
        A ``(train_df, test_df)`` tuple, both as
        :class:`pandas.DataFrame`.

    Raises:
        ValueError: If *stratify_col* is not present in *df*.
    """
    if stratify_col not in df.columns:
        raise ValueError(
            f"Stratify column '{stratify_col}' not found in DataFrame."
        )

    train_df, test_df = _sklearn_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state,
    )

    logger.info(
        "Split → train: %d rows, test: %d rows (stratified on '%s')",
        len(train_df),
        len(test_df),
        stratify_col,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
