"""Parquet fundamentals cache helpers (offline refresh + live merge)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_REFRESH_META = "combined/fundamentals_refresh.json"


def refresh_meta_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / _REFRESH_META


def load_refresh_meta(cache_dir: str | Path) -> dict[str, Any] | None:
    path = refresh_meta_path(cache_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        log.warning("Could not read fundamentals refresh metadata at %s", path, exc_info=True)
        return None


def save_refresh_meta(
    cache_dir: str | Path,
    *,
    symbols: list[str],
    row_count: int,
    symbol_count: int,
) -> None:
    path = refresh_meta_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "row_count": row_count,
        "symbol_count": symbol_count,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(
        "Fundamentals cache refresh recorded: %d rows, %d symbols",
        row_count, symbol_count,
    )


def hours_since_refresh(cache_dir: str | Path) -> float | None:
    meta = load_refresh_meta(cache_dir)
    if not meta or not meta.get("refreshed_at"):
        return None
    try:
        ts = datetime.fromisoformat(str(meta["refreshed_at"]))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
        return delta.total_seconds() / 3600.0
    except ValueError:
        return None


def load_cached_fundamentals_df() -> pd.DataFrame | None:
    """Return the merged fundamentals panel from ``combined/fundamentals``."""
    try:
        from firm.config import get_settings
        from firm.runtime import load_fundamentals

        cached = load_fundamentals(get_settings())
        if cached is not None and not cached.empty:
            return cached
    except Exception:
        log.debug("Cached fundamentals unavailable", exc_info=True)
    return None


def symbols_missing_fundamentals(
    universe: list[str], cached: pd.DataFrame | None,
) -> list[str]:
    if cached is None or cached.empty or "symbol" not in cached.columns:
        return list(universe)
    have = {str(s).upper() for s in cached["symbol"].unique()}
    return [sym for sym in universe if sym.upper() not in have]


def merge_with_cached_fundamentals(
    live_df: pd.DataFrame,
    cached: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if cached is None:
        cached = load_cached_fundamentals_df()
    if cached is None or cached.empty:
        return live_df

    if live_df is None or live_df.empty:
        log.debug(
            "Using cached fundamentals only: %d rows, %d symbols",
            len(cached), cached["symbol"].nunique(),
        )
        return cached

    merged = pd.concat([cached, live_df], ignore_index=True)
    if "date" in merged.columns and "symbol" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.sort_values(["symbol", "date"]).drop_duplicates(
            subset=["symbol", "date"],
            keep="last",
        )
    log.debug(
        "Merged live+cached fundamentals: %d rows, %d symbols",
        len(merged), merged["symbol"].nunique(),
    )
    return merged
