"""Tests for firm.runtime.load_prices.

Regression coverage: `fetch-data` writes combined price data via
ParquetCache under the key "combined/prices" (a hashed filename), but
load_prices() used to only look for a literal prices.parquet/prices.csv
file that fetch-data never created. Following the documented `fetch-data`
then `data_source="cache"` workflow never actually had any data to read.
"""

from __future__ import annotations

import pandas as pd
import pytest

from firm.config import Settings
from firm.data.cache import ParquetCache
from firm.runtime import load_prices


def _settings(cache_dir) -> Settings:
    s = Settings()
    s.data.cache_dir = str(cache_dir)
    return s


class TestLoadPrices:
    def test_reads_data_written_by_fetch_data_via_parquet_cache(self, tmp_path):
        cache = ParquetCache(tmp_path)
        df = pd.DataFrame({"date": ["2026-01-01"], "symbol": ["AAPL"], "close": [100.0]})
        cache.put("combined/prices", df)

        result = load_prices(_settings(tmp_path))
        assert list(result["symbol"]) == ["AAPL"]

    def test_falls_back_to_plain_parquet_file(self, tmp_path):
        df = pd.DataFrame({"date": ["2026-01-01"], "symbol": ["MSFT"], "close": [200.0]})
        df.to_parquet(tmp_path / "prices.parquet", index=False)

        result = load_prices(_settings(tmp_path))
        assert list(result["symbol"]) == ["MSFT"]

    def test_falls_back_to_plain_csv_file(self, tmp_path):
        df = pd.DataFrame({"date": ["2026-01-01"], "symbol": ["GOOG"], "close": [300.0]})
        df.to_csv(tmp_path / "prices.csv", index=False)

        result = load_prices(_settings(tmp_path))
        assert list(result["symbol"]) == ["GOOG"]

    def test_raises_clear_error_when_nothing_cached(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run fetch-data first"):
            load_prices(_settings(tmp_path))

    def test_prefers_parquet_cache_over_plain_file_when_both_exist(self, tmp_path):
        cache = ParquetCache(tmp_path)
        cache.put("combined/prices", pd.DataFrame({"date": ["2026-01-01"], "symbol": ["AAPL"], "close": [1.0]}))
        pd.DataFrame({"date": ["2026-01-01"], "symbol": ["STALE"], "close": [1.0]}).to_parquet(tmp_path / "prices.parquet", index=False)

        result = load_prices(_settings(tmp_path))
        assert list(result["symbol"]) == ["AAPL"]
