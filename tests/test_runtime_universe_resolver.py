"""Tests for firm.runtime.load_universe_membership / build_universe_resolver."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from firm.config import Settings
from firm.data.cache import ParquetCache
from firm.data.universe import UniverseResolver, build_resolver
from firm.runtime import build_universe_resolver, load_universe_membership


def _settings(cache_dir) -> Settings:
    s = Settings()
    s.data.cache_dir = str(cache_dir)
    return s


class TestLoadUniverseMembership:
    def test_reads_combined_universe_membership_key(self, tmp_path):
        cache = ParquetCache(tmp_path)
        df = pd.DataFrame({
            "index": ["sp500"],
            "symbol": ["AAPL"],
            "added_date": ["2010-01-01"],
            "removed_date": [None],
        })
        cache.put("combined/universe_membership", df)

        result = load_universe_membership(_settings(tmp_path))
        assert result is not None
        assert list(result["symbol"]) == ["AAPL"]

    def test_falls_back_to_csv(self, tmp_path):
        csv_path = tmp_path / "universe_membership.csv"
        csv_path.write_text("index,symbol,added_date,removed_date\nsp500,MSFT,2010-01-01,\n")

        result = load_universe_membership(_settings(tmp_path))
        assert result is not None
        assert list(result["symbol"]) == ["MSFT"]

    def test_returns_none_when_absent(self, tmp_path):
        assert load_universe_membership(_settings(tmp_path)) is None


class TestBuildResolver:
    def test_uses_real_membership_when_present(self):
        membership = pd.DataFrame({
            "symbol": ["AAPL", "DELISTED"],
            "added_date": [pd.NaT, pd.NaT],
            "removed_date": [pd.NaT, pd.Timestamp("2019-06-01")],
        })
        resolver = build_resolver(membership, ["AAPL", "DELISTED"])
        assert isinstance(resolver, UniverseResolver)
        assert resolver.symbols_asof(datetime(2020, 1, 1)) == ["AAPL"]

    def test_falls_back_to_static_when_no_membership(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="firm.data.universe"):
            resolver = build_resolver(None, ["AAPL", "MSFT"])
        assert resolver.symbols_asof(datetime(2020, 1, 1)) == ["AAPL", "MSFT"]
        assert any("static" in r.message for r in caplog.records)

    def test_falls_back_to_static_when_membership_empty(self):
        resolver = build_resolver(pd.DataFrame(), ["AAPL"])
        assert resolver.symbols_asof(datetime(2020, 1, 1)) == ["AAPL"]


class TestBuildUniverseResolverWrapper:
    def test_prefers_cached_membership(self, tmp_path):
        cache = ParquetCache(tmp_path)
        df = pd.DataFrame({
            "symbol": ["AAPL", "DELISTED"],
            "added_date": [None, None],
            "removed_date": [None, "2019-06-01"],
        })
        cache.put("combined/universe_membership", df)

        resolver = build_universe_resolver(_settings(tmp_path), ["AAPL", "DELISTED"])
        assert resolver.symbols_asof(datetime(2020, 1, 1)) == ["AAPL"]

    def test_degrades_gracefully_on_cache_error(self, tmp_path, monkeypatch):
        def _boom(*args, **kwargs):
            raise OSError("cache unavailable")

        monkeypatch.setattr("firm.runtime.load_universe_membership", _boom)
        resolver = build_universe_resolver(_settings(tmp_path), ["AAPL"])
        assert resolver.symbols_asof(datetime(2020, 1, 1)) == ["AAPL"]
