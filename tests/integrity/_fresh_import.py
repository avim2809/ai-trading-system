"""Shared helper: run code in a fresh interpreter that cannot touch live state.

Importing ``firm.api.app`` runs ``create_app()`` and ``setup_logging(log_file=
f"{FIRM_DATA_DIR or 'data'}/logs/api.log")`` at import time, so a naive subprocess
would attach a second rotating handler to the live IBKR ``data/logs/api.log``.
Every fresh-interpreter import test in ``tests/integrity/`` must use ``run_fresh``:
temp ``FIRM_DATA_DIR`` and cwd, live-start switches popped, this checkout's ``src``
first on ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

_POPPED = ("FIRM_AUTO_START_LIVE", "FIRM_LIVE_CONFIG", "FIRM_ALLOW_TRADING")


def fresh_env(tmp_path: Path) -> dict[str, str]:
    env = {**os.environ, "FIRM_DATA_DIR": str(tmp_path), "PYTHONPATH": str(SRC)}
    for var in _POPPED:
        env.pop(var, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def repo_data_listing() -> list[str]:
    """Sorted listing of the repo's own ``data/`` tree (must not change during a test)."""
    data = REPO_ROOT / "data"
    if not data.exists():
        return []
    return sorted(str(p.relative_to(REPO_ROOT)) for p in data.rglob("*"))


def run_fresh(code: str, tmp_path: Path, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = fresh_env(tmp_path)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
