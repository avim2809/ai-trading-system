"""Tests for live data feed cache-only fundamentals."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from firm.data.cache import ParquetCache
from firm.config import Settings


def test_refresh_cache_only_skips_network(tmp_path, monkeypatch):
    monkeypatch.delenv("FIRM_LIVE_FETCH_FUNDAMENTALS", raising=False)

    cache = ParquetCache(tmp_path)
    cache.put(
        "combined/fundamentals",
        pd.DataFrame({
            "date": ["2024-06-01"],
            "symbol": ["AAPL"],
            "pe_ratio": [25.0],
        }),
    )
    settings = Settings()
    settings.data.cache_dir = str(tmp_path)
    monkeypatch.setattr("firm.config.get_settings", lambda: settings)

    fund_prov = MagicMock()
    price_prov = MagicMock()
    price_prov.get_prices.return_value = pd.DataFrame()

    from firm.live.data_feed import LiveDataFeed

    feed = LiveDataFeed(
        providers={"fundamentals": fund_prov, "prices": price_prov},
        universe=["AAPL"],
    )
    view = feed.refresh()
    funds = view.fundamentals()
    assert not funds.empty
    fund_prov.get_fundamentals.assert_not_called()
