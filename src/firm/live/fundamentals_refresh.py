"""In-app fundamentals cache refresh (scheduled + on boot).

Live trading cycles read ``combined/fundamentals`` only. This module
refreshes that Parquet panel via the provider fallback chain on a daily
schedule and when the cache is stale at engine start.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd

from firm.config import Settings, get_settings
from firm.data.cache import ParquetCache
from firm.data.fundamentals_cache import hours_since_refresh, save_refresh_meta
from firm.data.providers.fallback import FallbackProvider

log = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_DAYS = 365 * 6
_DEFAULT_REFRESH_HOUR = 8
_DEFAULT_MAX_AGE_HOURS = 24.0

_refresh_lock = threading.Lock()


def _max_age_hours() -> float:
    raw = os.getenv("FIRM_FUNDAMENTALS_REFRESH_MAX_AGE_HOURS", str(_DEFAULT_MAX_AGE_HOURS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_MAX_AGE_HOURS


def cache_refresh_due(
    cache_dir: str | None = None,
    *,
    max_age_hours: float | None = None,
) -> bool:
    """True when the fundamentals cache is missing or older than *max_age_hours*."""
    cfg = get_settings()
    age = hours_since_refresh(cache_dir or cfg.data.cache_dir)
    limit = max_age_hours if max_age_hours is not None else _max_age_hours()
    if age is None:
        return True
    return age >= limit


def refresh_fundamentals_cache(
    symbols: list[str],
    *,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    settings: Settings | None = None,
) -> bool:
    """Fetch fundamentals for *symbols* and merge into ``combined/fundamentals``.

    Returns True when rows were written.  Serialized — concurrent callers block.
    """
    if not symbols:
        log.warning("Fundamentals refresh skipped — empty symbol list")
        return False

    with _refresh_lock:
        cfg = settings or get_settings()
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=lookback_days)
        start_s, end_s = start.isoformat(), end.isoformat()

        log.info(
            "Fundamentals cache refresh: %d symbols (%s → %s)",
            len(symbols), start_s, end_s,
        )
        provider = FallbackProvider(settings=cfg)
        try:
            df = provider.get_fundamentals(symbols, start_s, end_s)
        except Exception:
            log.error("Fundamentals cache refresh failed", exc_info=True)
            return False

        if df is None or df.empty:
            log.warning("Fundamentals cache refresh returned no rows")
            return False

        cache = ParquetCache(cfg.data.cache_dir)
        merged = cache.merge_combined("combined/fundamentals", df)
        cache.put("combined/fundamentals", merged)

        sym_col = merged["symbol"] if "symbol" in merged.columns else pd.Series(dtype=str)
        save_refresh_meta(
            cfg.data.cache_dir,
            symbols=list(symbols),
            row_count=len(merged),
            symbol_count=int(sym_col.nunique()),
        )
        log.info(
            "Fundamentals cache refresh done: %d rows, %d symbols",
            len(merged), sym_col.nunique(),
        )
        return True


def _refresh_in_background(
    symbols: list[str],
    *,
    reason: str,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> None:
    def _run() -> None:
        try:
            refresh_fundamentals_cache(symbols, lookback_days=lookback_days)
        except Exception:
            log.error("Background fundamentals refresh failed (%s)", reason, exc_info=True)

    thread = threading.Thread(
        target=_run,
        name=f"fundamentals-refresh-{reason}",
        daemon=True,
    )
    thread.start()
    log.info("Started background fundamentals refresh (%s)", reason)


def maybe_refresh_fundamentals_cache_on_start(
    symbols: list[str],
    *,
    max_age_hours: float | None = None,
) -> None:
    """Refresh in the background when the cache is stale at engine boot."""
    if not cache_refresh_due(max_age_hours=max_age_hours):
        age = hours_since_refresh(get_settings().data.cache_dir)
        log.info(
            "Fundamentals cache fresh (%.1fh old) — skipping boot refresh",
            age or 0.0,
        )
        return
    log.info("Fundamentals cache stale or missing — scheduling boot refresh")
    _refresh_in_background(symbols, reason="boot")


def run_scheduled_fundamentals_refresh(symbols: list[str]) -> None:
    """APScheduler entrypoint — daily fundamentals cache refresh."""
    log.info("Scheduled fundamentals cache refresh")
    _refresh_in_background(symbols, reason="scheduled")
