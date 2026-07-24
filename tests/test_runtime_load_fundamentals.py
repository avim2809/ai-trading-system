"""Tests for firm.runtime.load_fundamentals."""

from __future__ import annotations

import pandas as pd

from firm.config import Settings
from firm.data.cache import ParquetCache
from firm.runtime import load_fundamentals


def _settings(cache_dir) -> Settings:
    s = Settings()
    s.data.cache_dir = str(cache_dir)
    return s


class TestLoadFundamentals:
    def test_reads_combined_fundamentals_key(self, tmp_path):
        cache = ParquetCache(tmp_path)
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "symbol": ["AAPL"],
            "pe_ratio": [25.0],
            "roe": [0.2],
        })
        cache.put("combined/fundamentals", df)

        result = load_fundamentals(_settings(tmp_path))
        assert result is not None
        assert list(result["symbol"]) == ["AAPL"]

    def test_falls_back_to_legacy_fundamentals_key(self, tmp_path):
        cache = ParquetCache(tmp_path)
        df = pd.DataFrame({"date": ["2024-01-01"], "symbol": ["MSFT"], "pe_ratio": [30.0]})
        cache.put("fundamentals", df)

        result = load_fundamentals(_settings(tmp_path))
        assert result is not None
        assert list(result["symbol"]) == ["MSFT"]

    def test_returns_none_when_absent(self, tmp_path):
        assert load_fundamentals(_settings(tmp_path)) is None
