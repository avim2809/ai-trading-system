"""SQLite-backed durable persistence for live portfolio/attribution state.

The drawdown kill switch already survives a process restart via a small JSON
file (see ``LiveTradingEngine._persist_kill_switch_state``). Two other pieces
of live-trading state did not: the NAV/equity-curve history built up in
``PortfolioState`` and the per-strategy P&L series built up in
``PerformanceAttribution``. Both are rebuilt from scratch on every process
restart (broker reconnect re-seeds cash/holdings, but *history* has no other
source of truth) — losing the equity curve on every restart/redeploy, and
resetting the ``optimal`` signal-combination method's inverse-covariance
history to empty every time.

This module stores both as whole-document JSON blobs keyed by name in a
single SQLite file, following the same ``sqlite3`` + WAL pattern as
``firm.llm.cache.ResponseCache``. A blob (not per-row upserts) is deliberate:
this state is always read back and rewritten as one unit (a full NAV history,
a full attribution snapshot), cycle cadence in live/paper trading is at most
a few times a minute, and a JSON document of a few thousand data points is
trivially cheap to serialize — far simpler and safer than maintaining
row-level schemas for nested per-strategy time series.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from firm.contracts.models import PortfolioSnapshot
from firm.time_utils import utcnow

log = logging.getLogger(__name__)

PORTFOLIO_HISTORY_KEY = "portfolio_history"
ATTRIBUTION_STATE_KEY = "attribution_state"
KILL_SWITCH_KEY = "kill_switch"
DAILY_LIMITS_KEY = "daily_limits"
TRADER_STATE_KEY = "trader_state"


class LiveStateStore:
    """Durable key/value store for live-engine state, backed by SQLite."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS state_blobs (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """

    def __init__(
        self,
        db_path: str | Path = "data/live_state.db",
        max_snapshots: int = 20_000,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Bounds portfolio-history growth for a long-running process; the
        # oldest snapshots are dropped on save once the cap is exceeded.
        # 20k snapshots (~decades at an hourly cadence) is far beyond any
        # realistic paper/live trading run.
        self._max_snapshots = max_snapshots
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(self._DDL)
        self._conn.commit()
        log.info("Live state store opened at %s", self._db_path)

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # Generic blob storage
    # ------------------------------------------------------------------

    def _save_blob(self, key: str, value: Any) -> None:
        try:
            payload = json.dumps(value)
        except (TypeError, ValueError):
            log.warning(
                "Failed to serialize live-state blob %r — not persisted",
                key, exc_info=True,
            )
            return
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO state_blobs (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    (key, payload, utcnow().isoformat()),
                )
                self._conn.commit()
            except sqlite3.Error:
                log.warning(
                    "Failed to persist live-state blob %r to %s",
                    key, self._db_path, exc_info=True,
                )

    def _load_blob(self, key: str) -> Any | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT value FROM state_blobs WHERE key = ?", (key,)
                ).fetchone()
            except sqlite3.Error:
                log.warning(
                    "Failed to read live-state blob %r from %s",
                    key, self._db_path, exc_info=True,
                )
                return None
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            log.warning(
                "Corrupt live-state blob %r in %s — ignoring persisted value",
                key, self._db_path, exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Portfolio NAV/equity-curve history
    # ------------------------------------------------------------------

    def save_portfolio_history(self, snapshots: list[PortfolioSnapshot]) -> None:
        """Persist the full portfolio snapshot history (overwrites prior save)."""
        trimmed = snapshots[-self._max_snapshots:]
        payload = [
            {**asdict(s), "asof": s.asof.isoformat()} for s in trimmed
        ]
        self._save_blob(PORTFOLIO_HISTORY_KEY, payload)

    def load_portfolio_history(self) -> list[PortfolioSnapshot]:
        """Restore portfolio snapshot history saved by a previous process.

        Returns an empty list (never raises) when nothing was persisted yet
        or the persisted blob is unreadable — restoring history is a
        continuity nicety, not something that should block engine startup.
        """
        from datetime import datetime as _dt

        raw = self._load_blob(PORTFOLIO_HISTORY_KEY)
        if not raw:
            return []
        snapshots: list[PortfolioSnapshot] = []
        for row in raw:
            try:
                row = dict(row)
                row["asof"] = _dt.fromisoformat(row["asof"])
                snapshots.append(PortfolioSnapshot(**row))
            except (TypeError, ValueError, KeyError):
                log.warning(
                    "Skipping malformed persisted portfolio snapshot: %r", row,
                )
        return snapshots

    # ------------------------------------------------------------------
    # Performance attribution
    # ------------------------------------------------------------------

    def save_attribution_state(self, state: dict[str, Any]) -> None:
        """Persist the full attribution state (see PerformanceAttribution.export_state)."""
        self._save_blob(ATTRIBUTION_STATE_KEY, state)

    def load_attribution_state(self) -> dict[str, Any] | None:
        """Restore attribution state saved by a previous process, or None."""
        return self._load_blob(ATTRIBUTION_STATE_KEY)

    # ------------------------------------------------------------------
    # Kill switch (mirror of the JSON file for a single source-of-truth DB;
    # the JSON file at ``kill_switch_state_path`` remains the mechanism
    # LiveTradingEngine actually reads on startup — see engine.py — this is
    # an additional durable copy for operators/tools that prefer querying
    # one database instead of scattered files).
    # ------------------------------------------------------------------

    def save_kill_switch(self, state: dict[str, Any]) -> None:
        self._save_blob(KILL_SWITCH_KEY, state)

    def load_kill_switch(self) -> dict[str, Any] | None:
        return self._load_blob(KILL_SWITCH_KEY)

    # ------------------------------------------------------------------
    # Daily trade-count/turnover caps (see LiveTradingEngine._check_daily_limits).
    # These are counted against a same-process in-memory total by default;
    # without persisting them here, a mid-day restart (deploy, crash, etc.)
    # silently resets the "daily" cap to zero for the rest of that day.
    # ------------------------------------------------------------------

    def save_daily_limits(self, state: dict[str, Any]) -> None:
        self._save_blob(DAILY_LIMITS_KEY, state)

    def load_daily_limits(self) -> dict[str, Any] | None:
        return self._load_blob(DAILY_LIMITS_KEY)

    # ------------------------------------------------------------------
    # TraderAgent conviction-EMA state (see TraderAgent._smooth_convictions).
    # Same rationale as daily limits: without persisting this, a mid-day
    # restart resets the smoothing memory to nothing, so the very next cycle
    # rebalances at full unsmoothed strength — exactly the volatility this
    # feature exists to damp, and restarts are frequent enough in this
    # deployment (multiple per week) that this would matter in practice.
    # ------------------------------------------------------------------

    def save_trader_state(self, state: dict[str, Any]) -> None:
        self._save_blob(TRADER_STATE_KEY, state)

    def load_trader_state(self) -> dict[str, Any] | None:
        return self._load_blob(TRADER_STATE_KEY)

    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                log.debug("Error closing live state store", exc_info=True)
