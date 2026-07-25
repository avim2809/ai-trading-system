"""Tests for sentiment cache helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from firm.data.cache import ParquetCache
from firm.data.sentiment_cache import (
    merge_with_cached_sentiment,
    partition_sentiment_fetch,
    save_sentiment_cache,
)
from firm.config import Settings


def _row(sym: str, day: str, score: float = 0.5) -> dict:
    return {
        "date": day,
        "symbol": sym,
        "sentiment_score": score,
        "news_volume": 1,
        "source": "massive",
        "headline": f"headline-{sym}-{day}",
    }


def test_partition_cold_vs_warm_symbols():
    cached = pd.DataFrame([
        _row("AAPL", "2026-07-23"),
        _row("MSFT", "2026-07-20"),  # stale vs asof 2026-07-24
    ])
    asof = datetime(2026, 7, 24, 15, 0, tzinfo=timezone.utc)
    plan = partition_sentiment_fetch(
        ["AAPL", "MSFT", "NVDA"],
        cached,
        asof,
        lookback_days=30,
        incremental_days=7,
    )
    assert "NVDA" in plan["cold_symbols"]
    assert "AAPL" not in plan["warm_symbols"]
    assert "MSFT" in plan["warm_symbols"]


def test_merge_dedupes_headlines():
    cached = pd.DataFrame([_row("AAPL", "2026-07-22", 0.1)])
    live = pd.DataFrame([_row("AAPL", "2026-07-22", 0.9)])
    merged = merge_with_cached_sentiment(live, cached)
    assert len(merged) == 1
    assert merged.iloc[0]["sentiment_score"] == 0.9


def test_save_sentiment_cache_writes_parquet(tmp_path, monkeypatch):
    settings = Settings()
    settings.data.cache_dir = str(tmp_path)
    monkeypatch.setattr("firm.config.get_settings", lambda: settings)

    df = pd.DataFrame([_row("AAPL", "2026-07-22")])
    saved = save_sentiment_cache(
        df,
        lookback_days=30,
        asof=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )
    assert len(saved) == 1
    cache = ParquetCache(tmp_path)
    on_disk = cache.get("combined/sentiment")
    assert on_disk is not None
    assert len(on_disk) == 1
