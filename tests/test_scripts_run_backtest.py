"""Tests for firm.scripts.run_backtest's UniverseResolver wiring.

Regression coverage: the CLI path (``run-backtest`` / ``scripts/run_backtest.py``)
always calls ``pit_store.get_universe(start_dt)`` to determine the tradable set
for the *entire* run — unlike the API/job-manager path, there's no
``universe_symbols`` override to fall back on, so this is the one place a
survivorship-aware resolver has to be installed *before* that call for it to
have any effect at all.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.config import Settings


def _prices_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", "2020-06-01", freq="B")
    frames = []
    for sym in ("AAPL", "DELISTED"):
        frames.append(pd.DataFrame({
            "date": dates,
            "symbol": sym,
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
            "volume": 1.0, "adj_close": 1.0,
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(autouse=True)
def _fake_argv(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run-backtest", "--config", "config/settings.yaml",
        "--output-dir", str(tmp_path / "runs"),
    ])


class _FakeEngine:
    captured: dict = {}

    def __init__(self, bt_config):
        _FakeEngine.captured["bt_config"] = bt_config

    def setup(self, prices_df, pit_store, orchestrator, universe):
        _FakeEngine.captured["universe"] = universe
        _FakeEngine.captured["pit_store"] = pit_store

    def run(self):
        pass

    def generate_report(self):
        report = MagicMock()
        report.to_text.return_value = ""
        report.save = MagicMock()
        return report


class TestRunBacktestWiresUniverseResolver:
    def test_resolver_excludes_symbol_removed_before_start_date(self, tmp_path):
        """A membership frame that shows DELISTED removed before start_date
        must exclude it from the universe handed to BacktestEngine."""
        from firm.scripts.run_backtest import main

        settings = Settings()
        settings.data.cache_dir = str(tmp_path)
        settings.backtest.start_date = "2020-03-01"
        settings.backtest.end_date = "2020-06-01"

        membership = pd.DataFrame({
            "symbol": ["AAPL", "DELISTED"],
            "added_date": [None, None],
            "removed_date": [None, "2020-01-15"],
        })

        with patch("firm.scripts.run_backtest.get_settings", return_value=settings), \
             patch("firm.scripts.run_backtest.load_prices", return_value=_prices_df()), \
             patch("firm.scripts.run_backtest.load_fundamentals", return_value=None), \
             patch("firm.runtime.load_universe_membership", return_value=membership), \
             patch("firm.scripts.run_backtest.build_orchestrator", return_value=MagicMock()), \
             patch("firm.backtest.engine.BacktestEngine", _FakeEngine), \
             patch("builtins.print"):
            main()

        assert _FakeEngine.captured["universe"] == ["AAPL"]

    def test_static_fallback_includes_all_cached_symbols_when_no_membership_data(self, tmp_path):
        from firm.scripts.run_backtest import main

        settings = Settings()
        settings.data.cache_dir = str(tmp_path)
        settings.backtest.start_date = "2020-03-01"
        settings.backtest.end_date = "2020-06-01"

        with patch("firm.scripts.run_backtest.get_settings", return_value=settings), \
             patch("firm.scripts.run_backtest.load_prices", return_value=_prices_df()), \
             patch("firm.scripts.run_backtest.load_fundamentals", return_value=None), \
             patch("firm.scripts.run_backtest.build_orchestrator", return_value=MagicMock()), \
             patch("firm.backtest.engine.BacktestEngine", _FakeEngine), \
             patch("builtins.print"):
            main()

        assert sorted(_FakeEngine.captured["universe"]) == ["AAPL", "DELISTED"]


class TestRunBacktestLoadsFundamentalsAndSentimentWithoutCrashing:
    """Regression: a second pit_store.load(fundamentals=...) call (without
    `prices`, which has no default) used to raise TypeError whenever cached
    fundamentals were present, crashing this CLI entry point outright for
    any real dataset with a fundamentals cache."""

    def test_fundamentals_and_sentiment_both_load_into_the_same_pit_store(self):
        from firm.scripts.run_backtest import main

        settings = Settings()
        settings.backtest.start_date = "2020-03-01"
        settings.backtest.end_date = "2020-06-01"

        fund_df = pd.DataFrame({
            "date": ["2020-01-01"], "symbol": ["AAPL"], "pe_ratio": [20.0],
        })
        sentiment_df = pd.DataFrame({
            "date": ["2020-01-01"], "symbol": ["AAPL"],
            "sentiment_score": [0.5], "news_volume": [5],
        })

        with patch("firm.scripts.run_backtest.get_settings", return_value=settings), \
             patch("firm.scripts.run_backtest.load_prices", return_value=_prices_df()), \
             patch("firm.scripts.run_backtest.load_fundamentals", return_value=fund_df), \
             patch("firm.scripts.run_backtest.load_sentiment", return_value=sentiment_df), \
             patch("firm.runtime.load_universe_membership", return_value=None), \
             patch("firm.scripts.run_backtest.build_orchestrator", return_value=MagicMock()), \
             patch("firm.backtest.engine.BacktestEngine", _FakeEngine), \
             patch("builtins.print"):
            main()  # must not raise TypeError

        pit_store = _FakeEngine.captured["pit_store"]
        assert not pit_store.get_prices(["AAPL"], pd.Timestamp("2020-06-01")).empty
        assert not pit_store.get_fundamentals(["AAPL"], pd.Timestamp("2020-06-01")).empty
        assert not pit_store.get_sentiment(
            ["AAPL"], pd.Timestamp("2020-01-02"), lookback_days=30
        ).empty
