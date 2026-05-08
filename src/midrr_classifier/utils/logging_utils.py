"""Logging configuration for the MiDRR-Classifier package."""

import logging
import sys

_FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger pre-configured at INFO level.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance with a stream handler attached
        if the root logger has no handlers yet.
    """
    logger = logging.getLogger(name)
    if not logging.root.handlers:
        setup_root_logging()
    return logger


def setup_root_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a human-readable console handler.

    Call once at application entry-point (e.g. ``train.py`` or
    ``evaluate.py``).  Subsequent calls are no-ops because
    ``basicConfig`` only acts when the root logger has no handlers.

    Args:
        level: Logging level for the root handler (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format=_FMT,
        datefmt=_DATE_FMT,
        stream=sys.stdout,
    )
