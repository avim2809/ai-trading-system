"""Structured JSON logging configuration.

Usage::

    from firm.logging_setup import setup_logging
    setup_logging()

    import logging
    log = logging.getLogger("firm.strategies.momentum")
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the ``firm`` hierarchy."""
    return logging.getLogger(name)


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "file": record.filename,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure logging for the whole process, not just ``firm.*``.

    Attaches to the root logger (rather than just the "firm" logger) so
    third-party libraries used throughout the live trading path — ib_async,
    litellm, chromadb — are captured too, not only this package's own code.
    Losing those would mean losing exactly the kind of diagnostic output
    they emit (e.g. ib_async's market-data-subscription warnings) that
    matters for debugging a live run.

    Console gets a plain human-readable line (for watching a live session
    scroll by); the optional rotating file gets structured JSON (for later
    grep/jq analysis or feeding into an LLM reflection/self-improvement
    pass). Rotation defaults (10MB x 5 backups = 50MB ceiling per log
    stream) keep disk usage bounded regardless of how long a process runs.

    Parameters
    ----------
    level:
        Minimum severity (name or int).
    log_file:
        Optional path for a rotating JSON log file.
    max_bytes:
        Max size per log file before rotation.
    backup_count:
        Number of rotated files to keep.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(funcName)s:%(lineno)d - %(message)s"
    ))
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            str(path), maxBytes=max_bytes, backupCount=backup_count
        )
        fh.setFormatter(_JSONFormatter())
        root.addHandler(fh)
