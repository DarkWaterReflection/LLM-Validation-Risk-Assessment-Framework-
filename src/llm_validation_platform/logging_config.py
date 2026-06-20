"""Structured, audit-friendly logging.

Every validation run emits a deterministic, timestamped log that becomes part of
the evidence package retained by the Model Risk Management Group.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("mrm")
    if logger.handlers:  # idempotent across repeated calls / Streamlit reruns
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"mrm.{name}")
