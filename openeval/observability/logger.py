# openeval/observability/logger.py

import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Central logger factory.

    Every module gets its own logger from here:
        logger = get_logger(__name__)

    __name__ → gives the module path, e.g. "openeval.judge.judge".
    This way the log shows which file the message came from.
    """

    logger = logging.getLogger(name)

    # Are handlers already attached? (avoid duplicates if get_logger is called twice)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Terminal handler (rich) ──────────────────────────────
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,          # shows which file:line
        rich_tracebacks=True,    # renders exceptions nicely
        markup=True,
    )
    rich_handler.setLevel(logging.INFO)  # only INFO+ in the terminal
    logger.addHandler(rich_handler)

    # ── File handler ─────────────────────────────────────────
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # A separate file each day: logs/openeval_2026-05-21.log
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"openeval_{today}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # everything including DEBUG in the file
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)

    return logger