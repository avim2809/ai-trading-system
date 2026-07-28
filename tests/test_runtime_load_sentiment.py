"""Tests for firm.runtime.load_sentiment."""

from __future__ import annotations

import pandas as pd

from firm.config import Settings
from firm.data.cache import ParquetCache
from firm.runtime import load_sentiment


def _settings(cache_dir) -> Settings:
    s = Settings()
    s.data.cache_dir = str(cache_dir)
    return s


class TestLoadSentiment:
    def test_reads_combined_sentiment_key(self, tmp_path):
        cache = ParquetCache(tmp_path)
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "symbol": ["AAPL"],
            "sentiment_score": [0.5],
            "news_volume": [10],
        })
        cache.put("combined/sentiment", df)

        result = load_sentiment(_settings(tmp_path))
        assert result is not None
        assert list(result["symbol"]) == ["AAPL"]

    def test_returns_none_when_absent(self, tmp_path):
        assert load_sentiment(_settings(tmp_path)) is None
