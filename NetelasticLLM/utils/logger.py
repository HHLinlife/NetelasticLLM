"""
utils/logger.py
Centralised logging configuration for NetelasticLLM.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name:       str               = "netelastic",
    log_dir:    Optional[str]     = None,
    level:      int               = logging.INFO,
    console:    bool              = True,
    file_log:   bool              = True,
) -> logging.Logger:
    """
    Configure and return a named logger.

    Args:
        name    : Logger name (also used as filename prefix).
        log_dir : Directory for log files. Defaults to 'logs/'.
        level   : Logging level (e.g. logging.DEBUG).
        console : Whether to attach a StreamHandler.
        file_log: Whether to write logs to a timestamped file.
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if file_log:
        log_dir = log_dir or "logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file  = os.path.join(log_dir, f"{name}_{timestamp}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("Logging to file: %s", log_file)

    return logger


def get_logger(name: str = "netelastic") -> logging.Logger:
    """Return (or create) the named logger without re-configuring handlers."""
    return logging.getLogger(name)
