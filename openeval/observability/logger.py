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
    Merkezi logger factory.
    
    Her modül kendi logger'ını buradan alır:
        logger = get_logger(__name__)
    
    __name__ → "openeval.judge.judge" gibi modül yolunu verir.
    Bu sayede logda hangi dosyadan geldiği görünür.
    """

    logger = logging.getLogger(name)

    # Zaten handler eklenmiş mi? (get_logger iki kez çağrılırsa duplicate olmasın)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Terminal handler (rich) ──────────────────────────────
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,          # hangi dosya:satır gösterir
        rich_tracebacks=True,    # exception'ları güzel gösterir
        markup=True,
    )
    rich_handler.setLevel(logging.INFO)  # Terminal'de sadece INFO+
    logger.addHandler(rich_handler)

    # ── Dosya handler ────────────────────────────────────────
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Her gün ayrı dosya: logs/openeval_2026-05-21.log
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"openeval_{today}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Dosyada DEBUG dahil her şey
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)

    return logger