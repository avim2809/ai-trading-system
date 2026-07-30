"""Tests for firm.runtime.load_analyst_ratings."""

from __future__ import annotations

import pandas as pd

from firm.config import Settings
from firm.data.cache import ParquetCache
from firm.runtime import load_analyst_ratings


def _settings(cache_dir) -> Settings:
    s = Settings()
    s.data.cache_dir = str(cache_dir)
    return s


class TestLoadAnalystRatings:
    def test_reads_combined_analyst_ratings_key(self, tmp_path):
        cache = ParquetCache(tmp_path)
        df = pd.DataFrame({
            "date": ["2026-07-01"],
            "symbol": ["AAPL"],
            "strong_buy": [6],
            "buy": [23],
            "hold": [17],
            "sell": [2],
            "strong_sell": [2],
        })
        cache.put("combined/analyst_ratings", df)

        result = load_analyst_ratings(_settings(tmp_path))
        assert result is not None
        assert list(result["symbol"]) == ["AAPL"]

    def test_returns_none_when_absent(self, tmp_path):
        assert load_analyst_ratings(_settings(tmp_path)) is None
