"""Tests for the fetch-data CLI's caching behaviour.

Regression coverage: cache keys used to be f"{kind}/{provider}/{start}_{end}"
with no symbols in them at all, so re-running fetch-data for a *different*
symbol universe against the same provider and date range silently returned
another run's cached data — including a stale partial result left behind by
the FallbackProvider partial-success bug (see test_fallback_provider.py).
Now keys are built via ParquetCache.make_key(), which includes the symbol
set, so distinct universes never collide.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from firm.data.schemas import PRICE_COLS
from firm.scripts import fetch_data


def _price_df(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-01-01"] * len(symbols),
        "symbol": symbols,
        "open": [1.0] * len(symbols),
        "high": [1.0] * len(symbols),
        "low": [1.0] * len(symbols),
        "close": [1.0] * len(symbols),
        "volume": [1.0] * len(symbols),
        "adj_close": [1.0] * len(symbols),
    })[PRICE_COLS]


def _fake_provider(symbols: list[str]) -> MagicMock:
    p = MagicMock()
    p.get_prices = MagicMock(return_value=_price_df(symbols))
    p.get_fundamentals = MagicMock(side_effect=NotImplementedError)
    p.get_news_sentiment = MagicMock(side_effect=NotImplementedError)
    return p


class TestFetchDataCacheKeys:
    def test_different_symbol_universes_do_not_collide(self, tmp_path):
        cfg = MagicMock()
        cfg.data.cache_dir = str(tmp_path)

        with patch("firm.scripts.fetch_data.get_settings", return_value=cfg), \
             patch("firm.scripts.fetch_data.setup_logging"), \
             patch("firm.scripts.fetch_data.get_provider", side_effect=lambda name, settings: _fake_provider(["AAPL", "MSFT"])):
            fetch_data.main(["--symbols", "AAPL,MSFT", "--start", "2020-01-01", "--end", "2026-01-01", "--providers", "massive"])

        with patch("firm.scripts.fetch_data.get_settings", return_value=cfg), \
             patch("firm.scripts.fetch_data.setup_logging"), \
             patch("firm.scripts.fetch_data.get_provider", side_effect=lambda name, settings: _fake_provider(["GOOG", "TSLA"])):
            fetch_data.main(["--symbols", "GOOG,TSLA", "--start", "2020-01-01", "--end", "2026-01-01", "--providers", "massive"])

        from firm.data.cache import ParquetCache
        cache = ParquetCache(str(tmp_path))
        second_run = cache.get(cache.make_key("prices", provider="massive", symbols=["GOOG", "TSLA"], start="2020-01-01", end="2026-01-01"))
        first_run = cache.get(cache.make_key("prices", provider="massive", symbols=["AAPL", "MSFT"], start="2020-01-01", end="2026-01-01"))

        assert set(second_run["symbol"]) == {"GOOG", "TSLA"}
        assert set(first_run["symbol"]) == {"AAPL", "MSFT"}

    def test_combined_prices_merges_incremental_symbol_runs(self, tmp_path):
        cfg = MagicMock()
        cfg.data.cache_dir = str(tmp_path)

        with patch("firm.scripts.fetch_data.get_settings", return_value=cfg), \
             patch("firm.scripts.fetch_data.setup_logging"), \
             patch("firm.scripts.fetch_data.get_provider", side_effect=lambda name, settings: _fake_provider(["AAPL"])):
            fetch_data.main(["--symbols", "AAPL", "--start", "2020-01-01", "--end", "2026-01-01", "--providers", "massive"])

        with patch("firm.scripts.fetch_data.get_settings", return_value=cfg), \
             patch("firm.scripts.fetch_data.setup_logging"), \
             patch("firm.scripts.fetch_data.get_provider", side_effect=lambda name, settings: _fake_provider(["AAPL", "MSFT", "GOOG"])):
            fetch_data.main(["--symbols", "AAPL,MSFT,GOOG", "--start", "2020-01-01", "--end", "2026-01-01", "--providers", "massive"])

        from firm.data.cache import ParquetCache
        cache = ParquetCache(str(tmp_path))
        combined = cache.get("combined/prices")
        assert set(combined["symbol"]) == {"AAPL", "MSFT", "GOOG"}

    def test_combined_prices_keeps_symbols_from_prior_runs(self, tmp_path):
        cfg = MagicMock()
        cfg.data.cache_dir = str(tmp_path)

        with patch("firm.scripts.fetch_data.get_settings", return_value=cfg), \
             patch("firm.scripts.fetch_data.setup_logging"), \
             patch("firm.scripts.fetch_data.get_provider", side_effect=lambda name, settings: _fake_provider(["AAPL"])):
            fetch_data.main(["--symbols", "AAPL", "--start", "2020-01-01", "--end", "2026-01-01", "--providers", "massive"])

        with patch("firm.scripts.fetch_data.get_settings", return_value=cfg), \
             patch("firm.scripts.fetch_data.setup_logging"), \
             patch("firm.scripts.fetch_data.get_provider", side_effect=lambda name, settings: _fake_provider(["GOOG"])):
            fetch_data.main(["--symbols", "GOOG", "--start", "2020-01-01", "--end", "2026-01-01", "--providers", "massive"])

        from firm.data.cache import ParquetCache
        cache = ParquetCache(str(tmp_path))
        combined = cache.get("combined/prices")
        assert set(combined["symbol"]) == {"AAPL", "GOOG"}
