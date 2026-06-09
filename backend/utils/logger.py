"""
utils/logger.py – Application-wide logging setup.

Every module obtains a logger via:
    from backend.utils.logger import get_logger
    log = get_logger(__name__)

Output goes to TWO sinks simultaneously:
  1. Console (stderr)  – human-readable during development
  2. Rotating file     – logs/proactive_ai.log (10 MB x 5 backups)

The log directory is created automatically at the project root.
Log level is controlled by the LOG_LEVEL env var (default: INFO).
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOG_LEVEL_NAME: str = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL: int = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)

# logs/ directory sits at the project root (parent of utils/)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_LOG_DIR: Path = _PROJECT_ROOT / "logs"
_LOG_FILE: Path = _LOG_DIR / "proactive_ai.log"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_MAX_BYTES: int = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT: int = 5                # proactive_ai.log + 5 rotated copies

# ---------------------------------------------------------------------------
# One-time root logger bootstrap
# ---------------------------------------------------------------------------

_configured = False


def _build_file_handler() -> logging.Handler | None:
    """Create the file handler, falling back to temp storage if needed."""
    candidates = [_LOG_FILE, Path(tempfile.gettempdir()) / "proactive_ai.log"]
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                filename=str(candidate),
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setLevel(logging.DEBUG)
            return handler
        except OSError:
            continue
    return None


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler – respects LOG_LEVEL env var
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(_LOG_LEVEL)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)

    file_handler = _build_file_handler()
    if file_handler is not None:
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Silence overly chatty third-party libraries
    for noisy in ("uvicorn.access", "httpx", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info(
        "Logging initialised | level=%s | file=%s",
        _LOG_LEVEL_NAME,
        getattr(file_handler, "baseFilename", str(_LOG_FILE)) if file_handler else "console-only",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger. Call once at module level:

        log = get_logger(__name__)

    The first call also bootstraps the root logger (console + rotating file).
    """
    _configure_root_logger()
    return logging.getLogger(name)
