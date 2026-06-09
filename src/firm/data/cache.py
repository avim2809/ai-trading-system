"""Parquet-based disk cache for API responses.

Avoids redundant network calls by persisting DataFrames as Parquet files
keyed by a caller-chosen string (e.g. ``"prices/AAPL/2018-2023"``).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("firm.data.cache")


class ParquetCache:
    """Simple key-to-Parquet file cache."""

    def __init__(self, cache_dir: str | Path = "data/cache"):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        safe = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._dir / f"{safe}.parquet"

    def has(self, key: str) -> bool:
        return self._path_for(key).exists()

    def get(self, key: str) -> pd.DataFrame | None:
        """Read cached parquet. Returns None if not found."""
        path = self._path_for(key)
        if not path.exists():
            return None
        log.debug("Cache hit: %s -> %s", key, path)
        return pd.read_parquet(path)

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Write dataframe to parquet cache."""
        path = self._path_for(key)
        df.to_parquet(path, index=False)
        log.debug("Cached %d rows: %s -> %s", len(df), key, path)

    def invalidate(self, key: str) -> None:
        path = self._path_for(key)
        if path.exists():
            path.unlink()
            log.debug("Invalidated cache key: %s", key)
