"""Run guards shared by the two core_v1 drivers (tickets P3-11 and P3-08): seal preflight, run window, memory cap.

Research-only; not imported by any live module. Every guard is unconditional: there is no override flag, and the drivers call these
before they read anything. A failed guard is a STOP (``PreflightError``), never a recorded gap and never a degraded run.

Seal preflight (P3-11 "Seal preflight"): the code guard in ``firm.research.data_access`` is fail-open without the non-root research
user, so THIS is the real check. It refuses unless the effective uid is not root, none of the sealed paths is readable by the process,
the ETF store is readable, and ``docs/GUARDRAIL_REDTEAM.md`` records a pass (every matrix row ``PASS``; ``NOT RUN`` is not a pass).

Run window: weekdays 20:00-04:00 US/Eastern (after IBKR's after-hours leg, before pre-market) or Saturday, US/Eastern, and never a
Sunday in UTC (the BTC review runs Sunday UTC). The clock is ``date -u``, not the Asia/Jerusalem host clock.

Memory cap (the host has no swap and about 5 GB free; nice/ionice do not stop the OOM killer from hitting ``firm-api``): launch the
driver under ``prlimit --as=<bytes>`` (address-space cap, 2.5e9 recommended; the S2 unit's MemoryMax=1500M is an RSS cgroup limit and
RLIMIT_AS must sit above RSS) with single-threaded BLAS. The run ABORTS (``MemoryError``) if the cap is hit; it never degrades.
``require_memory_cap`` refuses to start without a finite cap.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import resource
import subprocess
from collections.abc import Callable
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

__all__ = [
    "ETF_STORE",
    "MAX_MEMORY_CAP_BYTES",
    "REDTEAM_DOC",
    "SEALED_PATHS",
    "PreflightError",
    "assert_in_window",
    "current_utc",
    "redteam_passed",
    "require_memory_cap",
    "seal_preflight",
    "window_ok",
]

SEALED_PATHS = (
    "data/cache",
    "data_alpaca",
    "data/research/s2_forward",
    "data/forward_monitors",
    "research/monitoring_sealed",
    "docs/s2_forward_snapshot.json",
)
ETF_STORE = "data/research/eodhd/etfs_full"
LIVE_CHECKOUT = Path("/local/store/git/ai-trading-system")   # the ACL and the data live here; a run worktree holds only tracked files
REDTEAM_DOC = "docs/GUARDRAIL_REDTEAM.md"
MAX_MEMORY_CAP_BYTES = 6 * 2**30   # above this the "cap" does not protect a host with ~5 GB free
_ET = ZoneInfo("America/New_York")
_ROW = re.compile(r"^\|\s*([A-Za-z]\w{0,3})\s*\|(.*)\|\s*([^|]*?)\s*\|\s*$")


class PreflightError(RuntimeError):
    """A run guard failed: STOP (not a recorded gap)."""


def current_utc() -> dt.datetime:
    """UTC now from ``date -u`` (AGENTS rule 13: never the host's Asia/Jerusalem clock)."""
    out = subprocess.run(["date", "-u", "+%s"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    return dt.datetime.fromtimestamp(int(out), tz=dt.UTC)


def window_ok(now_utc: dt.datetime) -> tuple[bool, str]:
    """(allowed, why). Weekdays 20:00-04:00 US/Eastern, or Saturday US/Eastern, never a Sunday in UTC."""
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    utc = now_utc.astimezone(dt.UTC)
    if utc.weekday() == 6:
        return False, f"{utc:%Y-%m-%d %H:%M}Z is a Sunday in UTC (BTC review day)"
    et = utc.astimezone(_ET)
    wd, hour = et.weekday(), et.hour   # Monday == 0
    if wd == 5:
        return True, f"Saturday US/Eastern ({et:%Y-%m-%d %H:%M %Z})"
    if wd <= 4 and hour >= 20:
        return True, f"weekday evening US/Eastern ({et:%a %H:%M %Z})"
    prev = (wd - 1) % 7   # 00:00-04:00 belongs to the previous day's evening window, which must itself be a weekday
    if hour < 4 and prev <= 4:
        return True, f"early morning after a weekday evening ({et:%a %H:%M %Z})"
    return False, f"{et:%a %Y-%m-%d %H:%M %Z} is outside weekdays 20:00-04:00 US/Eastern and Saturday"


def assert_in_window(clock: Callable[[], dt.datetime] = current_utc) -> dt.datetime:
    """Raise unless now (``clock``, default ``date -u``) is inside the run window. No override exists."""
    now = clock()
    ok, why = window_ok(now)
    if not ok:
        raise PreflightError(f"outside the run window: {why}")
    log.info("run window ok: %s", why)
    return now


def redteam_passed(path: Path) -> bool:
    """True iff the red-team doc has at least one matrix row and every row's Result cell is PASS (``NOT RUN`` and ``FAIL`` are not)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    results = []
    for line in text.splitlines():
        m = _ROW.match(line)
        if not m or m.group(1) == "#" or set(m.group(1)) <= {"-", ":"}:
            continue
        cell = m.group(3).strip().upper()
        if cell in ("PASS", "FAIL", "NOT RUN") or cell.startswith(("PASS", "FAIL", "NOT RUN")):
            results.append(cell)
    return bool(results) and all(c == "PASS" for c in results)


def seal_preflight(
    repo_dir: Path,
    *,
    data_root: Path | None = None,
    euid: Callable[[], int] = os.geteuid,
    access: Callable[[str, int], bool] = os.access,
) -> dict:
    """Refuse (``PreflightError``) unless non-root, sealed paths unreadable, ETF store readable, red-team pass recorded."""
    repo = Path(repo_dir)
    root = Path(data_root) if data_root is not None else repo   # sealed paths and the ETF store are judged where the ACL and the data are
    uid = int(euid())
    if uid == 0:
        raise PreflightError("refusing to run as root (the seal rests on the research user's missing read access)")
    leaks = [p for p in SEALED_PATHS if access(str(root / p), os.R_OK)]
    if leaks:
        raise PreflightError(f"sealed path(s) readable by this process, the ACL is missing: {leaks}")
    etf_ok = bool(access(str(root / ETF_STORE), os.R_OK | os.X_OK))
    if not etf_ok:
        raise PreflightError(f"ETF store {ETF_STORE} (etfs_full) is not readable; traverse on data/ and read on data/research/eodhd are required")
    if not redteam_passed(repo / REDTEAM_DOC):
        raise PreflightError(f"{REDTEAM_DOC} does not record a pass (every red-team row must be PASS); a missing ACL is a STOP")
    log.info("seal preflight ok (euid %d)", uid)
    return {"euid": uid, "sealed_paths_unreadable": list(SEALED_PATHS), "etf_store_readable": etf_ok, "redteam_pass": True}


def require_memory_cap(limit: Callable[[], tuple[int, int]] = lambda: resource.getrlimit(resource.RLIMIT_AS)) -> int:
    """Return the effective address-space cap in bytes; refuse (``PreflightError``) if there is none or it is above the ceiling."""
    soft, hard = limit()
    inf = resource.RLIM_INFINITY
    cap = min(c for c in (soft, hard) if c != inf) if (soft != inf or hard != inf) else None
    if cap is None or cap > MAX_MEMORY_CAP_BYTES:
        raise PreflightError(
            f"no usable memory cap (RLIMIT_AS {soft}/{hard}); start under `prlimit --as=2500000000` (ceiling {MAX_MEMORY_CAP_BYTES})")
    return int(cap)
