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


class _FakeBareReport:
    """Minimal stand-in shaped enough for execute_backtest's post-hoc
    returns-trimming step (empty series -> a no-op, same as a real report
    with no trades) without needing a real BacktestReport."""

    def __init__(self):
        self.returns = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        self.snapshots = []


class _FakeEngine:
    """Stand-in for BacktestEngine that just records what it was given."""

    captured: dict = {}

    def __init__(self, bt_config):
        _FakeEngine.captured["bt_config"] = bt_config

    def setup(self, prices_df, pit_store, orchestrator, universe, memory=None, llm_config=None):
        _FakeEngine.captured["prices_df"] = prices_df

    def run(self):
        pass

    def generate_report(self):
        return _FakeBareReport()


class TestExecuteBacktestFiltersRealDataByDateRange:
    def test_prices_passed_to_engine_are_restricted_to_requested_range(self):
        config = {
            "data_source": "cache",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["momentum"],
            "warmup_days": 0,
        }

        with patch("firm.runtime.load_prices", return_value=_full_history_df()), \
             patch("firm.config.get_settings"), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            result = execute_backtest(config)

        assert isinstance(result, _FakeBareReport)
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


class TestWarmupBuffer:
    """Regression coverage: strategies with a real lookback requirement
    (regime_hmm needs 252 days to train its HMM, momentum's 12-month factor
    needs 252, gann needs 120+) got starved for a large chunk of every
    ~99-trading-day walk-forward fold once the date-range fix above landed
    — some generated zero signals until well into the fold, others
    silently degraded to a much shorter lookback than designed for.
    Loading extra history *before* start_date (but never trading on it —
    see FirmStrategy's own start_date gate) fixes this without leaking
    into reported performance.
    """

    def test_loaded_prices_extend_before_start_date_by_default(self):
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
            execute_backtest(config)

        prices_df = _FakeEngine.captured["prices_df"]
        # Default warmup_days=365 -> real history well before start_date.
        assert prices_df["date"].min() <= pd.Timestamp("2024-01-01") - pd.Timedelta(days=300)
        # But still never past end_date.
        assert prices_df["date"].max() <= pd.Timestamp("2024-03-01")

    def test_warmup_days_is_configurable(self):
        config = {
            "data_source": "cache",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["momentum"],
            "warmup_days": 10,
        }

        with patch("firm.runtime.load_prices", return_value=_full_history_df()), \
             patch("firm.config.get_settings"), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            execute_backtest(config)

        prices_df = _FakeEngine.captured["prices_df"]
        assert prices_df["date"].min() >= pd.Timestamp("2024-01-01") - pd.Timedelta(days=14)

    def test_start_date_reaches_bt_config_for_the_firm_strategy_gate(self):
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
            execute_backtest(config)

        assert _FakeEngine.captured["bt_config"]["start_date"] == "2024-01-01"

    def test_synthetic_backtests_are_unaffected(self):
        """The synthetic branch already generates exactly the requested
        span (make_synthetic_prices) — warmup_days must not change that."""
        config = {
            "data_source": "synthetic",
            "start_date": "2021-01-01",
            "end_date": "2021-03-01",
            "seed": 1,
        }
        with patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            execute_backtest(config)

        prices_df = _FakeEngine.captured["prices_df"]
        # make_synthetic_prices ends exactly at end_date and generates
        # backward from there — no artificial extension before start_date
        # should have been requested for the synthetic path.
        assert prices_df["date"].max() <= pd.Timestamp("2021-03-01")


class _FakeReport:
    def __init__(self, returns, benchmark_returns):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.snapshots = []
        self.trades = []


class _FakeEngineWithReturns(_FakeEngine):
    def generate_report(self):
        idx = pd.date_range("2023-06-01", "2024-03-01", freq="B")
        returns = pd.Series(0.001, index=idx)
        bench = pd.Series(0.0005, index=idx)
        return _FakeReport(returns, bench)


class TestReturnsTrimmedToEvaluationWindow:
    """The raw return series backtrader produces spans the full loaded
    range (warmup + eval), since BacktestEngine itself has no concept of
    an evaluation boundary — only FirmStrategy's start_date gate stops
    *trading* early. The reported returns/benchmark series must still be
    trimmed to the real evaluation window, or metrics would be computed
    over a stretch of flat, no-trade warmup days that were never meant to
    count."""

    def test_returns_start_at_start_date_not_the_warmup_buffer(self):
        config = {
            "data_source": "cache",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["momentum"],
        }
        with patch("firm.runtime.load_prices", return_value=_full_history_df()), \
             patch("firm.config.get_settings"), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngineWithReturns), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            report = execute_backtest(config)

        assert report.returns.index.min() >= pd.Timestamp("2024-01-01")
        assert report.benchmark_returns.index.min() >= pd.Timestamp("2024-01-01")


class TestFirmStrategyNeverTradesDuringWarmup:
    """End-to-end (real BacktestEngine, not mocked): confirms the
    warmup-buffer data extending before start_date is visible to
    strategies via pit_view.prices() but genuinely never traded on."""

    def test_no_trades_or_snapshots_before_start_date(self):
        from firm.data.synthetic import make_synthetic_prices

        # Real, varied multi-symbol data spanning well before and after
        # the evaluation window, so momentum has plenty to react to.
        full_df = make_synthetic_prices(
            symbols=["AAPL", "MSFT", "GOOG"], n_days=400, end_date="2024-03-01", seed=3,
        )
        config = {
            "data_source": "cache",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL", "MSFT", "GOOG"],
            "strategies": ["momentum"],
            "rebalance_frequency": "weekly",
            "warmup_days": 300,
        }

        with patch("firm.runtime.load_prices", return_value=full_df), \
             patch("firm.config.get_settings"):
            report = execute_backtest(config)

        start_ts = pd.Timestamp("2024-01-01")
        for trade in report.trades:
            assert pd.Timestamp(trade["entry_dt"]) >= start_ts, (
                f"trade entered at {trade['entry_dt']}, before start_date"
            )
        for snap in report.snapshots:
            assert pd.Timestamp(snap.asof) >= start_ts, (
                f"snapshot recorded at {snap.asof}, before start_date"
            )


class TestExecuteBacktestLoadsFundamentals:
    def test_cache_backtest_loads_fundamentals_into_pit_store(self):
        fund_df = pd.DataFrame({
            "date": ["2024-01-01"],
            "symbol": ["AAPL"],
            "pe_ratio": [20.0],
            "roe": [0.15],
        })
        load_calls: list[dict] = []

        class FakePitStore:
            def load(self, **kwargs):
                load_calls.append(kwargs)

            def get_universe(self, asof):
                return ["AAPL"]

            def set_universe_resolver(self, resolver):
                pass

        config = {
            "data_source": "cache",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["momentum"],
            "warmup_days": 0,
        }

        with patch("firm.runtime.load_prices", return_value=_full_history_df()), \
             patch("firm.runtime.load_fundamentals", return_value=fund_df), \
             patch("firm.config.get_settings"), \
             patch("firm.backtest.run.PointInTimeDataStore", FakePitStore), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            execute_backtest(config)

        assert any("prices" in call for call in load_calls)
        assert any(
            "fundamentals" in call and call["fundamentals"] is fund_df
            for call in load_calls
        )

    def test_synthetic_backtest_does_not_load_fundamentals(self):
        load_calls: list[dict] = []

        class FakePitStore:
            def load(self, **kwargs):
                load_calls.append(kwargs)

            def get_universe(self, asof):
                return ["AAPL"]

        config = {
            "data_source": "synthetic",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["momentum"],
        }

        with patch("firm.runtime.load_fundamentals") as load_fund, \
             patch("firm.backtest.run.PointInTimeDataStore", FakePitStore), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            execute_backtest(config)

        load_fund.assert_not_called()
        # pit_store.load() is always called with a `fundamentals` kwarg (it's
        # a named parameter of that call, not conditionally included) — the
        # actual behaviour under test is that no fundamentals *data* was
        # loaded for a synthetic backtest, i.e. the value is None.
        assert all(call.get("fundamentals") is None for call in load_calls)


class TestExecuteBacktestLoadsSentiment:
    def test_cache_backtest_loads_sentiment_into_pit_store(self):
        sentiment_df = pd.DataFrame({
            "date": ["2024-01-01"], "symbol": ["AAPL"],
            "sentiment_score": [0.5], "news_volume": [5],
        })
        load_calls: list[dict] = []

        class FakePitStore:
            def load(self, **kwargs):
                load_calls.append(kwargs)

            def get_universe(self, asof):
                return ["AAPL"]

            def set_universe_resolver(self, resolver):
                pass

        config = {
            "data_source": "cache",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["sentiment"],
            "warmup_days": 0,
        }

        with patch("firm.runtime.load_prices", return_value=_full_history_df()), \
             patch("firm.runtime.load_fundamentals", return_value=None), \
             patch("firm.runtime.load_sentiment", return_value=sentiment_df), \
             patch("firm.config.get_settings"), \
             patch("firm.backtest.run.PointInTimeDataStore", FakePitStore), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            execute_backtest(config)

        assert any(
            "sentiment" in call and call["sentiment"] is sentiment_df
            for call in load_calls
        )

    def test_synthetic_backtest_does_not_load_sentiment(self):
        load_calls: list[dict] = []

        class FakePitStore:
            def load(self, **kwargs):
                load_calls.append(kwargs)

            def get_universe(self, asof):
                return ["AAPL"]

        config = {
            "data_source": "synthetic",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["sentiment"],
        }

        with patch("firm.runtime.load_sentiment") as load_sent, \
             patch("firm.backtest.run.PointInTimeDataStore", FakePitStore), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            execute_backtest(config)

        load_sent.assert_not_called()
        assert all(call.get("sentiment") is None for call in load_calls)


class TestExecuteBacktestWiresUniverseResolver:
    """Regression coverage for wiring UniverseResolver into execute_backtest.

    A resolver should always be installed for real-data runs so
    ``pit_store.get_universe`` is survivorship-aware rather than falling back
    to "whatever symbols happen to have price data" — and it must actually be
    consulted (not just installed) when no explicit ``universe_symbols`` is
    configured.
    """

    def test_resolver_installed_for_cache_backtests(self):
        installed: list = []

        class FakePitStore:
            def load(self, **kwargs):
                pass

            def get_universe(self, asof):
                return ["AAPL"]

            def set_universe_resolver(self, resolver):
                installed.append(resolver)

        config = {
            "data_source": "cache",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["momentum"],
            "warmup_days": 0,
        }
        with patch("firm.runtime.load_prices", return_value=_full_history_df()), \
             patch("firm.config.get_settings"), \
             patch("firm.backtest.run.PointInTimeDataStore", FakePitStore), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            execute_backtest(config)

        assert len(installed) == 1

    def test_resolver_is_consulted_when_no_explicit_universe_symbols(self):
        """Without an explicit universe_symbols override, execute_backtest
        must ask the (resolver-backed) pit_store for the union of universe
        membership across the whole backtest window rather than silently
        using an empty/undefined symbol list or only a single snapshot."""
        get_universe_union_calls: list = []

        class FakePitStore:
            def load(self, **kwargs):
                pass

            def get_universe(self, asof):
                return ["AAPL"]

            def get_universe_union(self, start, end):
                get_universe_union_calls.append((start, end))
                return ["AAPL"]

            def set_universe_resolver(self, resolver):
                pass

        config = {
            "data_source": "cache",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "strategies": ["momentum"],
            "warmup_days": 0,
        }
        with patch("firm.runtime.load_prices", return_value=_full_history_df()), \
             patch("firm.config.get_settings"), \
             patch("firm.backtest.run.PointInTimeDataStore", FakePitStore), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            execute_backtest(config)

        assert len(get_universe_union_calls) == 1

    def test_no_resolver_installed_for_synthetic_backtests(self):
        class FakePitStore:
            def load(self, **kwargs):
                pass

            def get_universe(self, asof):
                return ["AAPL"]

            def set_universe_resolver(self, resolver):
                raise AssertionError("synthetic backtests should not install a resolver")

        config = {
            "data_source": "synthetic",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "universe_symbols": ["AAPL"],
            "strategies": ["momentum"],
        }
        with patch("firm.backtest.run.PointInTimeDataStore", FakePitStore), \
             patch("firm.backtest.run.BacktestEngine", _FakeEngine), \
             patch("firm.backtest.run.build_orchestrator", return_value=MagicMock()):
            execute_backtest(config)  # must not raise

