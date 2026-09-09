"""SQLite-backed persistence for triggered pattern-scan matches (Phase 3
follow-up, docs/pattern_recognition_plan.md §2b).

Follows the same ``sqlite3`` + WAL + ``threading.Lock`` + try/except-log-
and-degrade connection convention as ``firm.llm.cache.ResponseCache`` and
``firm.live.state_store.LiveStateStore`` — row-based (not the latter's
whole-document blob style) since callers filter/page individual matches and
later update a single row's ``outcome`` in place, exactly the access pattern
``firm.live.trade_history.TradeHistoryStore.update_order_status`` already
established for orders.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from firm.time_utils import utcnow

log = logging.getLogger(__name__)

_JSON_COLUMNS = ("score_breakdown", "pivots", "meta")


class PatternScanHistoryStore:
    """Durable, queryable history of every persisted pattern-scan match."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS pattern_scan_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        asof TEXT NOT NULL,
        confirm_date TEXT,
        pattern TEXT NOT NULL,
        direction TEXT NOT NULL,
        confirmed INTEGER NOT NULL,
        confirm_index INTEGER NOT NULL,
        entry REAL,
        stop REAL,
        target REAL,
        fit_quality REAL,
        geometry_tolerance_used REAL,
        volume_ratio REAL,
        duration_bars INTEGER,
        follow_through_atr REAL,
        risk_reward REAL,
        quality_score REAL,
        score_breakdown TEXT,
        pivots TEXT,
        meta TEXT,
        outcome TEXT,
        outcome_checked_at TEXT,
        source TEXT NOT NULL DEFAULT 'manual',
        created_at TEXT NOT NULL
    )
    """

    def __init__(self, db_path: str | Path = "data/pattern_scan_history.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(self._DDL)
        self._conn.commit()
        log.info("Pattern scan history store opened at %s", self._db_path)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def insert_matches(
        self,
        rows: list[dict[str, Any]],
        *,
        confirm_dates: list[str | None] | None = None,
        source: str = "manual",
    ) -> None:
        """Persist a batch of already-serialized matches — the same dict
        shape ``firm.api.routers.patterns._serialize_match`` produces.

        ``confirm_dates`` (optional, same length/order as ``rows``) carries
        each match's own confirmation-bar calendar date, distinct from
        ``asof`` (the *scan's* reference date, shared by every row in a
        batch) — outcome tracking needs the former to align a persisted row
        against freshly-fetched price data days or weeks later.
        """
        if not rows:
            return
        now = utcnow().isoformat()
        with self._lock:
            try:
                for i, row in enumerate(rows):
                    confirm_date = confirm_dates[i] if confirm_dates else None
                    self._conn.execute(
                        """INSERT INTO pattern_scan_history
                           (symbol, asof, confirm_date, pattern, direction, confirmed,
                            confirm_index, entry, stop, target, fit_quality,
                            geometry_tolerance_used, volume_ratio, duration_bars,
                            follow_through_atr, risk_reward, quality_score,
                            score_breakdown, pivots, meta, source, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            row["symbol"], row["asof"], confirm_date, row["pattern"],
                            row["direction"], 1 if row["confirmed"] else 0,
                            row["confirm_index"], row["entry"], row["stop"], row["target"],
                            row["fit_quality"], row["geometry_tolerance_used"],
                            row["volume_ratio"], row["duration_bars"],
                            row["follow_through_atr"], row["risk_reward"], row["quality_score"],
                            json.dumps(row["score_breakdown"]), json.dumps(row["pivots"]),
                            json.dumps(row["meta"]), source, now,
                        ),
                    )
                self._conn.commit()
            except sqlite3.Error:
                log.warning(
                    "Failed to persist %d pattern-scan match(es) to %s",
                    len(rows), self._db_path, exc_info=True,
                )

    def list_history(
        self,
        *,
        symbol: str | None = None,
        pattern: str | None = None,
        direction: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Paginated, filterable read. ``outcome="pending"`` matches rows
        with no outcome recorded yet; any other value matches that exact
        outcome string.
        """
        query = "SELECT * FROM pattern_scan_history WHERE 1=1"
        params: list[Any] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())
        if pattern:
            query += " AND pattern = ?"
            params.append(pattern)
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        if outcome == "pending":
            query += " AND outcome IS NULL"
        elif outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._lock:
            try:
                cur = self._conn.execute(query, params)
                cols = [d[0] for d in cur.description]
                fetched = cur.fetchall()
            except sqlite3.Error:
                log.warning(
                    "Failed to read pattern-scan history from %s", self._db_path, exc_info=True,
                )
                return []
        return [self._row_to_dict(cols, r) for r in fetched]

    def pending_outcomes(self, *, max_rows: int = 200) -> list[dict[str, Any]]:
        """Rows with no recorded outcome yet, oldest first (most likely to
        already have enough subsequent price history to resolve).
        """
        rows = self.list_history(outcome="pending", limit=max_rows)
        return list(reversed(rows))

    def update_outcome(self, row_id: int, outcome: str) -> bool:
        """Record ``row_id``'s target/stop/timeout outcome in place — the
        one exception to otherwise-append-only history, mirroring
        ``TradeHistoryStore.update_order_status``'s in-place-update pattern.
        """
        with self._lock:
            try:
                cur = self._conn.execute(
                    "UPDATE pattern_scan_history SET outcome = ?, outcome_checked_at = ? WHERE id = ?",
                    (outcome, utcnow().isoformat(), row_id),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except sqlite3.Error:
                log.warning(
                    "Failed to update outcome for pattern-scan history row %d", row_id, exc_info=True,
                )
                return False

    @staticmethod
    def _row_to_dict(cols: list[str], row: tuple) -> dict[str, Any]:
        d = dict(zip(cols, row))
        for col in _JSON_COLUMNS:
            if d.get(col):
                try:
                    d[col] = json.loads(d[col])
                except (TypeError, ValueError):
                    d[col] = None
        d["confirmed"] = bool(d.get("confirmed"))
        return d
