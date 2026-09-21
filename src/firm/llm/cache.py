"""SQLite-backed LLM response cache for deduplication and cost tracking."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResponseCache:
    """Persistent cache storing LLM responses keyed by (model, messages) hash.

    The cache reduces costs during development, backtests, and repeated
    analyses by returning previously-computed completions without hitting
    the provider.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS llm_cache (
        key TEXT PRIMARY KEY,
        response TEXT NOT NULL,
        model TEXT NOT NULL,
        tokens_in INTEGER NOT NULL DEFAULT 0,
        tokens_out INTEGER NOT NULL DEFAULT 0,
        cost REAL NOT NULL DEFAULT 0.0,
        created_at TEXT NOT NULL
    )
    """

    # Durable hit/miss counters, persisted alongside the cached responses
    # themselves rather than kept only on the Python instance.
    #
    # Every live process constructs *many* independent ``ResponseCache``
    # instances that all point at the same on-disk DB — one per
    # ``LLMAgentMixin``-based agent (fundamental_analyst, sentiment_analyst,
    # ...), one for ``LiveTradingEngine._get_llm_service`` (reflection), one
    # for ``rag.assistant.TradingAssistant`` — with no central registry
    # tying them together (unlike ``LiveTradingEngine`` itself, which is
    # published on ``app.state.live_engine`` for the API routers). An
    # in-process instance counter therefore can't answer "how many
    # hits/misses has this cache seen": whichever instance a caller happens
    # to hold (e.g. a fresh one constructed per HTTP request, as
    # ``GET /api/llm/cache/stats`` used to do) only ever sees its own calls,
    # which for a throwaway instance is always zero. Persisting the counters
    # in the same SQLite DB as the cached rows makes them durable, shared by
    # every instance/process pointed at that DB file, and survive restarts —
    # a strictly better fix than plumbing a single "live" instance through,
    # since there isn't one live instance to plumb.
    _COUNTERS_DDL = """
    CREATE TABLE IF NOT EXISTS llm_cache_counters (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        hits INTEGER NOT NULL DEFAULT 0,
        misses INTEGER NOT NULL DEFAULT 0
    )
    """

    def __init__(self, db_path: str = "data/llm_cache.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # A single connection is shared across threads (check_same_thread=False),
        # so all access is serialised by ``self._lock``; WAL mode + a busy
        # timeout let concurrent agents read/write without "database is locked"
        # errors under live trading, where all agents may query the LLM at once.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(self._DDL)
        self._conn.execute(self._COUNTERS_DDL)
        self._conn.execute(
            "INSERT OR IGNORE INTO llm_cache_counters (id, hits, misses) VALUES (1, 0, 0)"
        )
        self._conn.commit()
        # Cheap per-instance mirrors of the durable counters above (e.g. for
        # a caller that wants "hits/misses since this instance was
        # constructed" without a DB round trip). ``stats()`` reports the
        # durable, DB-wide totals, not these.
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash(
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        # Every parameter that changes the completion must be part of the key,
        # otherwise a request can receive a response generated under different
        # settings (e.g. a json_mode call served a cached free-text answer, or
        # a temperature=0 call served a temperature=0.9 result).
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
            if row:
                self._hits += 1
                self._conn.execute(
                    "UPDATE llm_cache_counters SET hits = hits + 1 WHERE id = 1"
                )
                self._conn.commit()
                return row[0]
            self._misses += 1
            self._conn.execute(
                "UPDATE llm_cache_counters SET misses = misses + 1 WHERE id = 1"
            )
            self._conn.commit()
            return None

    def put(
        self,
        key: str,
        response: str,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO llm_cache
                   (key, response, model, tokens_in, tokens_out, cost, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (key, response, model, tokens_in, tokens_out, cost,
                 datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost), 0) FROM llm_cache"
            ).fetchone()
            counters = self._conn.execute(
                "SELECT hits, misses FROM llm_cache_counters WHERE id = 1"
            ).fetchone()
        entries = row[0] if row else 0
        total_cost_saved = row[1] if row else 0.0
        hits, misses = (counters[0], counters[1]) if counters else (0, 0)
        db_size_mb = os.path.getsize(self._db_path) / (1024 * 1024) if self._db_path.exists() else 0
        return {
            # Durable totals from ``llm_cache_counters`` — shared across every
            # ``ResponseCache``/``LLMService`` instance pointed at this DB
            # (in this process or any other) and across restarts, not just
            # this Python instance's own ``get()`` calls.
            "hits": hits,
            "misses": misses,
            "total_cost_saved": total_cost_saved,
            "entries": entries,
            "db_size_mb": round(db_size_mb, 2),
        }

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.execute(
                "UPDATE llm_cache_counters SET hits = 0, misses = 0 WHERE id = 1"
            )
            self._conn.commit()
        self._hits = 0
        self._misses = 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
