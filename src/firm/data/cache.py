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
        # Full 256-bit digest: a truncated 64-bit name risks collisions across
        # distinct datasets (one would silently overwrite/serve another).
        safe = hashlib.sha256(key.encode()).hexdigest()
        return self._dir / f"{safe}.parquet"

    @staticmethod
    def make_key(
        kind: str,
        *,
        provider: str = "",
        symbols: list[str] | None = None,
        start: str = "",
        end: str = "",
    ) -> str:
        """Build a canonical cache key for a fetched panel."""
        syms = ",".join(sorted(symbols or []))
        return f"{kind}/{provider}/{syms}/{start}_{end}"

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

    # Aliases for callers that use read/write naming.
    def write(self, key: str, df: pd.DataFrame) -> None:
        self.put(key, df)

    def read(self, key: str) -> pd.DataFrame | None:
        return self.get(key)

    def merge_combined(self, key: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge *new_df* into an existing combined panel, deduping symbol+date."""
        if new_df.empty:
            existing = self.get(key)
            return existing if existing is not None else new_df

        existing = self.get(key)
        if existing is None or existing.empty:
            merged = new_df
        else:
            merged = pd.concat([existing, new_df], ignore_index=True)
            if "date" in merged.columns and "symbol" in merged.columns:
                merged["date"] = pd.to_datetime(merged["date"])
                merged = merged.sort_values(["symbol", "date"]).drop_duplicates(
                    subset=["symbol", "date"],
                    keep="last",
                )
        self.put(key, merged)
        return merged
