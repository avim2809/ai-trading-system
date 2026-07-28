"""Tests for live data feed cache-only fundamentals."""

from __future__ import annotations

from datetime import datetime
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


def test_refresh_sentiment_cache_skips_network_when_fresh(tmp_path, monkeypatch):
    from firm.data.cache import ParquetCache

    cache = ParquetCache(tmp_path)
    cache.put(
        "combined/sentiment",
        pd.DataFrame({
            "date": [pd.Timestamp("2026-07-23")],
            "symbol": ["AAPL"],
            "sentiment_score": [0.5],
            "news_volume": [1],
            "source": ["massive"],
            "headline": ["cached"],
        }),
    )
    settings = Settings()
    settings.data.cache_dir = str(tmp_path)
    monkeypatch.setattr("firm.config.get_settings", lambda: settings)

    sent_prov = MagicMock()
    price_prov = MagicMock()
    price_prov.get_prices.return_value = pd.DataFrame()

    from firm.live.data_feed import LiveDataFeed

    feed = LiveDataFeed(
        providers={"sentiment": sent_prov, "prices": price_prov},
        universe=["AAPL"],
    )
    view = feed.refresh(asof=datetime(2026, 7, 24, 15, 0))
    sent = view.sentiment()
    assert not sent.empty
    sent_prov.get_news_sentiment.assert_not_called()


class TestUniverseResolverWiring:
    """LiveDataFeed installs a survivorship-aware resolver on every freshly
    built pit_store, consistent with the backtest wiring — this is mainly a
    forward-compatibility / audit hook for live (see build_universe_resolver
    docstring): live always trades *today's* configured universe, so it isn't
    a survivorship-bias fix by itself, but pit_store.get_universe() must still
    be resolver-backed rather than silently falling back to "whatever prices
    happened to load"."""

    def test_resolver_installed_on_refresh(self, tmp_path, monkeypatch):
        settings = Settings()
        settings.data.cache_dir = str(tmp_path)
        monkeypatch.setattr("firm.config.get_settings", lambda: settings)

        price_prov = MagicMock()
        price_prov.get_prices.return_value = pd.DataFrame()

        from firm.live.data_feed import LiveDataFeed

        feed = LiveDataFeed(providers={"prices": price_prov}, universe=["AAPL", "MSFT"])
        feed.refresh()

        assert sorted(feed.pit_store.get_universe(datetime(2026, 1, 1))) == ["AAPL", "MSFT"]

    def test_real_membership_data_excludes_delisted_symbol(self, tmp_path, monkeypatch):
        from firm.data.cache import ParquetCache

        cache = ParquetCache(tmp_path)
        cache.put(
            "combined/universe_membership",
            pd.DataFrame({
                "symbol": ["AAPL", "DELISTED"],
                "added_date": [None, None],
                "removed_date": [None, "2020-01-01"],
            }),
        )
        settings = Settings()
        settings.data.cache_dir = str(tmp_path)
        monkeypatch.setattr("firm.config.get_settings", lambda: settings)

        price_prov = MagicMock()
        price_prov.get_prices.return_value = pd.DataFrame()

        from firm.live.data_feed import LiveDataFeed

        feed = LiveDataFeed(providers={"prices": price_prov}, universe=["AAPL", "DELISTED"])
        feed.refresh(asof=datetime(2026, 1, 1))

        assert feed.pit_store.get_universe(datetime(2026, 1, 1)) == ["AAPL"]

    def test_resolver_setup_failure_degrades_to_static_universe(self, monkeypatch):
        """A broken settings/cache lookup at feed construction must not
        prevent live trading from starting — fall back to the configured
        symbol list, matching the pre-resolver behaviour."""
        monkeypatch.setattr(
            "firm.config.get_settings",
            MagicMock(side_effect=OSError("settings unavailable")),
        )

        price_prov = MagicMock()
        price_prov.get_prices.return_value = pd.DataFrame()

        from firm.live.data_feed import LiveDataFeed

        feed = LiveDataFeed(providers={"prices": price_prov}, universe=["AAPL"])
        feed.refresh()

        assert feed.pit_store.get_universe(datetime(2026, 1, 1)) == ["AAPL"]
