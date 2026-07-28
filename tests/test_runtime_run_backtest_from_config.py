"""Tests for firm.runtime.run_backtest_from_config.

Regression coverage: this function used to call
``pit_store.load(fundamentals=fund_df)`` in a *second* call after the
initial ``pit_store.load(prices=prices_df)`` — but ``PointInTimeDataStore
.load()``'s ``prices`` parameter has no default and each call fully replaces
prior state, so that second call raised ``TypeError: missing 1 required
positional argument: 'prices'`` whenever cached fundamentals were present.
Also covers the sentiment-loading wiring added alongside the fix.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from firm.runtime import run_backtest_from_config


def _prices_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2024-03-01", freq="B")
    return pd.DataFrame({
        "date": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume": 1.0, "adj_close": 1.0,
    })


class _FakeReport:
    def __init__(self):
        self.returns = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)


class _FakeEngine:
    captured: dict = {}

    def __init__(self, bt_config):
        pass

    def setup(self, prices_df, pit_store, orchestrator, universe, memory=None, llm_config=None):
        _FakeEngine.captured["pit_store"] = pit_store

    def run(self):
        pass

    def generate_report(self):
        return _FakeReport()


class TestRunBacktestFromConfigLoadsFundamentalsWithoutCrashing:
    def test_does_not_raise_when_fundamentals_are_cached(self):
        """The historical bug: this used to raise TypeError here."""
        fund_df = pd.DataFrame({
            "date": ["2024-01-01"], "symbol": ["AAPL"], "pe_ratio": [20.0],
        })
        with patch("firm.runtime.load_fundamentals", return_value=fund_df), \
             patch("firm.runtime.load_sentiment", return_value=None), \
             patch("firm.runtime.build_orchestrator", return_value=MagicMock()), \
             patch("firm.runtime.BacktestEngine", _FakeEngine):
            engine, report = run_backtest_from_config(
                {"strategies": ["momentum"]}, _prices_df(), ["AAPL"],
            )

        pit_store = _FakeEngine.captured["pit_store"]
        assert not pit_store.get_fundamentals(["AAPL"], pd.Timestamp("2024-06-01")).empty
        # Prices must have survived the fundamentals load, not been wiped out.
        assert not pit_store.get_prices(["AAPL"], pd.Timestamp("2024-06-01")).empty

    def test_sentiment_is_loaded_into_pit_store(self):
        sentiment_df = pd.DataFrame({
            "date": ["2024-01-01"], "symbol": ["AAPL"],
            "sentiment_score": [0.5], "news_volume": [5],
        })
        with patch("firm.runtime.load_fundamentals", return_value=None), \
             patch("firm.runtime.load_sentiment", return_value=sentiment_df), \
             patch("firm.runtime.build_orchestrator", return_value=MagicMock()), \
             patch("firm.runtime.BacktestEngine", _FakeEngine):
            run_backtest_from_config({"strategies": ["sentiment"]}, _prices_df(), ["AAPL"])

        pit_store = _FakeEngine.captured["pit_store"]
        sent = pit_store.get_sentiment(["AAPL"], pd.Timestamp("2024-01-02"), lookback_days=30)
        assert not sent.empty

    def test_missing_sentiment_does_not_crash(self):
        with patch("firm.runtime.load_fundamentals", return_value=None), \
             patch("firm.runtime.load_sentiment", return_value=None), \
             patch("firm.runtime.build_orchestrator", return_value=MagicMock()), \
             patch("firm.runtime.BacktestEngine", _FakeEngine):
            run_backtest_from_config({"strategies": ["momentum"]}, _prices_df(), ["AAPL"])

        pit_store = _FakeEngine.captured["pit_store"]
        assert pit_store.get_sentiment(["AAPL"], pd.Timestamp("2024-06-01")).empty
