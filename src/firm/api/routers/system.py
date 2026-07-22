"""Server resource monitoring + service control (restart/kill) router.

Exposes CPU/RAM/disk utilization for the host the API is running on, plus
endpoints to restart or force-kill the ``ai-trading.service`` systemd unit
from the GUI. The unit has ``Restart=always``, so a kill is recovered
automatically; restart/kill are separated so an operator can distinguish a
graceful restart from "the process is stuck, force it".

Both mutating endpoints run the actual ``systemctl`` call on a short delay
via a background thread. This is required because the request handling
this call is itself served by the very process being restarted/killed —
without the delay, the process would go away before the HTTP response for
this request could be flushed to the client, and the client would see a
connection reset instead of a clean ``{"status": ...}`` response.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from typing import Any

import psutil
from fastapi import APIRouter

log = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])

_SERVICE_NAME = "ai-trading.service"
# Long enough that the response has almost certainly been flushed to the
# client before systemctl tears the process down.
_ACTION_DELAY_SECONDS = 1.0
_DISK_PATH = "/"


def _systemctl_command(*args: str) -> list[str]:
    """Build the systemctl invocation, prefixing ``sudo -n`` only if needed.

    The service unit runs with ``User=root``, so this process is already
    root and can call systemctl directly. ``sudo`` is only prepended as a
    fallback for the (non-standard) case where the API is somehow running
    as a non-root user; ``-n`` keeps it non-interactive so a missing
    passwordless-sudo rule fails fast instead of hanging on a prompt.
    """
    import os

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return ["systemctl", *args]
    return ["sudo", "-n", "systemctl", *args]


def _run_delayed(command: list[str]) -> None:
    def _invoke() -> None:
        try:
            subprocess.run(command, check=False)
        except Exception:
            # The API process may already be going down by the time this
            # fires (or shortly after) — nothing useful can be done with
            # the exception here beyond logging it.
            log.exception("Delayed command failed: %s", command)

    timer = threading.Timer(_ACTION_DELAY_SECONDS, _invoke)
    timer.daemon = True
    timer.start()


@router.get("/resources")
def get_resources() -> dict[str, Any]:
    """Current CPU/RAM/disk utilization for the host.

    ``cpu_percent(interval=None)`` returns the utilization since the last
    call (or 0.0 on the very first call ever made in the process) instead
    of blocking for a sampling window — psutil maintains the last-seen
    counters internally, so this never stalls the request.
    """
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(_DISK_PATH)

    return {
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count() or 0,
        },
        "memory": {
            "used": mem.used,
            "total": mem.total,
            "percent": mem.percent,
        },
        "disk": {
            "used": disk.used,
            "total": disk.total,
            "percent": disk.percent,
            "path": _DISK_PATH,
        },
    }


@router.post("/restart")
def restart_service() -> dict[str, Any]:
    """Restart ai-trading.service. Returns immediately; the actual restart
    happens ~1s later on a background thread so this response can flush."""
    log.warning("Service restart requested via API")
    _run_delayed(_systemctl_command("restart", _SERVICE_NAME))
    return {"status": "restarting"}


@router.post("/kill")
def kill_service() -> dict[str, Any]:
    """Force-kill ai-trading.service with SIGKILL. Returns immediately; the
    unit's Restart=always brings it back up automatically."""
    log.warning("Service force-kill requested via API")
    _run_delayed(_systemctl_command("kill", "--signal=SIGKILL", _SERVICE_NAME))
    return {"status": "killing"}
