"""Live log-tail endpoint backing the frontend log monitor.

Reads the rotating JSON-lines file written by :mod:`firm.logging_setup` and
serves incremental slices via a byte-offset cursor, so the frontend can poll
cheaply instead of re-fetching the whole file on every tick.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

log = logging.getLogger(__name__)

router = APIRouter(prefix="/logs", tags=["logs"])

# Must match firm.api.app's setup_logging(log_file=...) default.
_LOG_FILE = Path(os.environ.get("FIRM_DATA_DIR", "data")) / "logs" / "api.log"
# Caps a single response (and the initial/reset read) so a client that's
# fallen far behind, or opens the page against a large file, can't force a
# huge read in one request.
_MAX_TAIL_BYTES = 512 * 1024


def _parse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except ValueError:
        # Non-JSON line (partial write, or a library writing straight to
        # stderr) — surface it as-is rather than silently dropping it.
        return {"ts": None, "level": "RAW", "logger": "", "msg": line}


@router.get("/tail")
def tail_logs(offset: int = Query(0, ge=0)) -> dict[str, Any]:
    """Return log lines appended since *offset*, plus the new cursor.

    ``offset`` is an opaque byte position from a previous call — pass 0 (or
    omit) to start from the end of the current backlog window. If the file
    was rotated/truncated since, or the client is too far behind, ``reset``
    is True and the client should discard its in-memory buffer.
    """
    if not _LOG_FILE.exists():
        return {"lines": [], "next_offset": 0, "reset": False}

    size = _LOG_FILE.stat().st_size
    reset = offset > size
    start = 0 if reset else offset

    if size - start > _MAX_TAIL_BYTES:
        start = size - _MAX_TAIL_BYTES
        reset = True

    try:
        with _LOG_FILE.open("rb") as f:
            f.seek(start)
            raw = f.read()
    except OSError:
        log.warning("Failed to read log file %s", _LOG_FILE, exc_info=True)
        return {"lines": [], "next_offset": offset, "reset": False}

    if reset and start > 0:
        # We likely seeked into the middle of a line — drop the partial
        # first fragment and keep only whole lines after it.
        nl = raw.find(b"\n")
        raw = raw[nl + 1 :] if nl != -1 else b""

    last_nl = raw.rfind(b"\n")
    if last_nl == -1:
        complete, leftover_len = b"", len(raw)
    else:
        complete, leftover_len = raw[:last_nl], len(raw) - (last_nl + 1)

    next_offset = size - leftover_len
    text = complete.decode("utf-8", errors="replace")
    lines = [p for p in (_parse_line(ln) for ln in text.splitlines()) if p is not None]
    return {"lines": lines, "next_offset": next_offset, "reset": reset}
