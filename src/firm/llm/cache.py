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
        self._conn.commit()
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
                return row[0]
            self._misses += 1
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
        entries = row[0] if row else 0
        total_cost_saved = row[1] if row else 0.0
        db_size_mb = os.path.getsize(self._db_path) / (1024 * 1024) if self._db_path.exists() else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_cost_saved": total_cost_saved,
            "entries": entries,
            "db_size_mb": round(db_size_mb, 2),
        }

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()
        self._hits = 0
        self._misses = 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
