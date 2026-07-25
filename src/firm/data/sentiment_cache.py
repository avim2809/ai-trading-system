"""Parquet sentiment cache helpers (live incremental refresh + merge)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from firm.data.schemas import SENTIMENT_COLS

log = logging.getLogger(__name__)

_SENTIMENT_CACHE_KEY = "combined/sentiment"
_DEFAULT_INCREMENTAL_DAYS = 7


def sentiment_cache_key() -> str:
    return _SENTIMENT_CACHE_KEY


def load_cached_sentiment_df() -> pd.DataFrame | None:
    """Return the merged sentiment panel from ``combined/sentiment``."""
    try:
        from firm.config import get_settings
        from firm.data.cache import ParquetCache

        cache = ParquetCache(get_settings().data.cache_dir)
        cached = cache.get(_SENTIMENT_CACHE_KEY)
        if cached is not None and not cached.empty:
            return cached
    except Exception:
        log.debug("Cached sentiment unavailable", exc_info=True)
    return None


def _normalize_sentiment_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=SENTIMENT_COLS)
    out = df.copy()
    for col in SENTIMENT_COLS:
        if col not in out.columns:
            out[col] = None
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_localize(None).dt.normalize()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.upper()
    return out[SENTIMENT_COLS]


def merge_with_cached_sentiment(
    live_df: pd.DataFrame,
    cached: pd.DataFrame | None = None,
) -> pd.DataFrame:
    live_df = _normalize_sentiment_df(live_df)
    if cached is None:
        cached = load_cached_sentiment_df()
    cached = _normalize_sentiment_df(cached) if cached is not None else pd.DataFrame(columns=SENTIMENT_COLS)

    if cached.empty:
        return live_df
    if live_df.empty:
        log.debug(
            "Using cached sentiment only: %d rows, %d symbols",
            len(cached), cached["symbol"].nunique(),
        )
        return cached

    merged = pd.concat([cached, live_df], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["date", "symbol", "headline"],
        keep="last",
    )
    log.debug(
        "Merged live+cached sentiment: %d rows, %d symbols",
        len(merged), merged["symbol"].nunique(),
    )
    return merged


def partition_sentiment_fetch(
    universe: list[str],
    cached: pd.DataFrame | None,
    asof: datetime,
    lookback_days: int,
    incremental_days: int = _DEFAULT_INCREMENTAL_DAYS,
) -> dict[str, Any]:
    """Plan incremental sentiment network fetches.

    Returns a dict with:
      - ``cold_symbols``: no cache rows — full ``lookback_days`` window
      - ``warm_symbols``: cached — only last ``incremental_days`` refreshed
      - ``full_start`` / ``incremental_start`` / ``end``: YYYY-MM-DD strings
    """
    end_ts = pd.Timestamp(asof)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert("UTC").tz_localize(None)
    end_ts = end_ts.normalize()
    end = end_ts.strftime("%Y-%m-%d")
    full_start = (end_ts - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    incremental_start = (
        end_ts - pd.Timedelta(days=max(1, incremental_days))
    ).strftime("%Y-%m-%d")

    cached_norm = _normalize_sentiment_df(cached) if cached is not None else pd.DataFrame(columns=SENTIMENT_COLS)
    cold_symbols: list[str] = []
    warm_symbols: list[str] = []
    yesterday = end_ts - pd.Timedelta(days=1)

    if cached_norm.empty:
        return {
            "cold_symbols": list(universe),
            "warm_symbols": [],
            "full_start": full_start,
            "incremental_start": incremental_start,
            "end": end,
        }

    grouped = cached_norm.groupby("symbol")["date"].max()
    have = set(grouped.index)
    for sym in universe:
        sym_u = sym.upper()
        if sym_u not in have:
            cold_symbols.append(sym)
            continue
        max_date = pd.Timestamp(grouped[sym_u])
        if max_date.tzinfo is not None:
            max_date = max_date.tz_convert("UTC").tz_localize(None)
        max_date = max_date.normalize()
        if pd.isna(max_date) or max_date < yesterday:
            warm_symbols.append(sym)

    return {
        "cold_symbols": cold_symbols,
        "warm_symbols": warm_symbols,
        "full_start": full_start,
        "incremental_start": incremental_start,
        "end": end,
    }


def save_sentiment_cache(
    df: pd.DataFrame,
    *,
    lookback_days: int,
    asof: datetime | None = None,
) -> pd.DataFrame:
    """Merge *df* into disk cache and trim to ``lookback_days`` history."""
    from firm.config import get_settings
    from firm.data.cache import ParquetCache

    merged = merge_with_cached_sentiment(df)
    if merged.empty:
        return merged

    asof_ts = pd.Timestamp(asof or datetime.now(timezone.utc))
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_convert("UTC").tz_localize(None)
    asof_ts = asof_ts.normalize()
    cutoff = asof_ts - pd.Timedelta(days=lookback_days + 30)
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], utc=True).dt.tz_localize(None).dt.normalize()
        merged = merged[merged["date"] >= cutoff]

    cache = ParquetCache(get_settings().data.cache_dir)
    cache.put(_SENTIMENT_CACHE_KEY, merged.reset_index(drop=True))
    log.info(
        "Sentiment cache saved: %d rows, %d symbols",
        len(merged), merged["symbol"].nunique(),
    )
    return merged


def incremental_days_from_env() -> int:
    import os

    raw = os.getenv("FIRM_SENTIMENT_INCREMENTAL_DAYS", str(_DEFAULT_INCREMENTAL_DAYS))
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_INCREMENTAL_DAYS
