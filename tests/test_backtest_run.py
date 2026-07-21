"""Tests for execute_backtest's real-data date-range filtering.

Regression coverage for a real incident: a walk-forward backtest's 5 folds,
each requesting a different start_date/end_date, produced bit-for-bit
identical metrics across every fold (down to floating point). Root cause —
execute_backtest's non-synthetic branch loaded the full cached price
history via load_prices() and handed it straight to BacktestEngine with no
date filtering at all; BacktestEngine itself has no concept of
start_date/end_date, so every fold silently ran on the entire ~6.5-year
dataset instead of its assigned window. The synthetic branch never showed
this because make_synthetic_prices() is explicitly built to generate only
the requested span.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from firm.backtest.run import execute_backtest


def _full_history_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", "2026-07-20", freq="B")
    return pd.DataFrame({
        "date": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume": 1.0, "adj_close": 1.0,
    })


class _FakeEngine:
    """Stand-in for BacktestEngine that just records what it was given."""

    captured: dict = {}

    def __init__(self, bt_config):
        pass

    def setup(self, prices_df, pit_store, orchestrator, universe, memory=None, llm_config=None):
        _FakeEngine.captured["prices_df"] = prices_df

    def run(self):
        pass

    def generate_report(self):
        return "report"


class TestExecuteBacktestFiltersRealDataByDateRange:
    def test_prices_passed_to_engine_are_restricted_to_requested_range(self):
        config = {
            "data_source": "cache",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["momentum"],
        }

        with patch("firm.runtime.load_prices", return_value=_full_history_df()), \
             patch("firm.config.get_settings"), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            result = execute_backtest(config)

        assert result == "report"
        prices_df = _FakeEngine.captured["prices_df"]
        assert prices_df["date"].min() >= pd.Timestamp("2024-01-01")
        assert prices_df["date"].max() <= pd.Timestamp("2024-03-01")
        # Far less than the ~1650 rows in the full cached history.
        assert len(prices_df) < 100

    def test_different_folds_of_the_same_cache_get_different_data(self):
        """The actual production symptom: two folds over the same cached
        dataset must not end up looking at identical price windows."""
        base_config = {
            "data_source": "cache",
            "universe_symbols": ["AAPL"],
            "strategies": ["momentum"],
        }

        seen = []
        for start, end in [("2020-06-01", "2020-09-01"), ("2025-01-01", "2025-04-01")]:
            with patch("firm.runtime.load_prices", return_value=_full_history_df()), \
                 patch("firm.config.get_settings"), \
                 patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
                 patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
                execute_backtest({**base_config, "start_date": start, "end_date": end})
            seen.append(_FakeEngine.captured["prices_df"]["date"].min())

        assert seen[0] != seen[1]
