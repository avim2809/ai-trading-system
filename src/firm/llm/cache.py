"""SQLite-backed LLM response cache for deduplication and cost tracking."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
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
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute(self._DDL)
        self._conn.commit()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash(model: str, messages: list[dict[str, Any]]) -> str:
        payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> str | None:
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
        self._conn.execute(
            """INSERT OR REPLACE INTO llm_cache
               (key, response, model, tokens_in, tokens_out, cost, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (key, response, model, tokens_in, tokens_out, cost,
             datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def stats(self) -> dict[str, Any]:
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
        self._conn.execute("DELETE FROM llm_cache")
        self._conn.commit()
        self._hits = 0
        self._misses = 0

    def close(self) -> None:
        self._conn.close()
