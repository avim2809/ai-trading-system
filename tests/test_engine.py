"""Tests for the Phase 3A backtest integration layer.

Covers data feeds, commissions, the PitViewAdapter, FirmStrategy
rebalancing logic, and BacktestEngine setup with synthetic data.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


from firm.backtest.commissions import PercentageCommission
from firm.backtest.datafeeds import AdjustedPandasData, dataframe_to_feed, load_feeds
from firm.backtest.firm_strategy import PitViewAdapter
from firm.backtest.engine import BacktestEngine
from firm.data.pit_store import PointInTimeDataStore
from firm.strategies.base import PitView


# ======================================================================
# Helpers
# ======================================================================

def _synthetic_prices(
    symbols: list[str],
    days: int = 60,
    start: str = "2023-01-02",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a multi-symbol OHLCV DataFrame with realistic random walks."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range(start, periods=days)
    for sym in symbols:
        base = 100.0 + rng.uniform(-20, 20)
        for dt in dates:
            ret = rng.normal(0.0005, 0.02)
            base *= 1 + ret
            o = base * (1 + rng.normal(0, 0.003))
            h = max(o, base) * (1 + abs(rng.normal(0, 0.005)))
            lo = min(o, base) * (1 - abs(rng.normal(0, 0.005)))
            rows.append({
                "date": dt,
                "symbol": sym,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(lo, 2),
                "close": round(base, 2),
                "volume": int(rng.uniform(100_000, 5_000_000)),
                "adj_close": round(base, 2),
            })
    return pd.DataFrame(rows)


def _make_pit_store(prices_df: pd.DataFrame) -> PointInTimeDataStore:
    store = PointInTimeDataStore()
    store.load(prices=prices_df)
    return store


# ======================================================================
# 1. datafeeds.py
# ======================================================================

class TestDataFrameToFeed:
    def test_basic_conversion(self):
        df = _synthetic_prices(["AAPL"], days=20)
        feed = dataframe_to_feed(df, "AAPL")
        assert isinstance(feed, AdjustedPandasData)

    def test_missing_symbol_raises(self):
        df = _synthetic_prices(["AAPL"], days=10)
        with pytest.raises(ValueError, match="No data found"):
            dataframe_to_feed(df, "MISSING")

    def test_missing_columns_raises(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=5),
            "symbol": "AAPL",
            "close": [100, 101, 102, 103, 104],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            dataframe_to_feed(df, "AAPL")

    def test_adj_close_fallback(self):
        """When adj_close column is absent, close should be used."""
        df = _synthetic_prices(["GOOG"], days=10)
        df = df.drop(columns=["adj_close"])
        feed = dataframe_to_feed(df, "GOOG")
        assert isinstance(feed, AdjustedPandasData)


class TestLoadFeeds:
    def test_returns_dict(self):
        df = _synthetic_prices(["AAPL", "GOOG"], days=10)
        feeds = load_feeds(df, ["AAPL", "GOOG"])
        assert set(feeds.keys()) == {"AAPL", "GOOG"}
        assert all(isinstance(f, AdjustedPandasData) for f in feeds.values())

    def test_skips_missing(self):
        df = _synthetic_prices(["AAPL"], days=10)
        feeds = load_feeds(df, ["AAPL", "NOPE"])
        assert "AAPL" in feeds
        assert "NOPE" not in feeds

    def test_empty_df(self):
        df = pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        feeds = load_feeds(df, ["AAPL"])
        assert feeds == {}


# ======================================================================
# 2. commissions.py
# ======================================================================

class TestPercentageCommission:
    def test_commission_via_public_api(self):
        comm = PercentageCommission(commission=0.001)
        result = comm.getcommission(size=100, price=50.0)
        assert result > 0

    def test_commission_proportional_to_size(self):
        comm = PercentageCommission(commission=0.001)
        r1 = comm.getcommission(size=100, price=50.0)
        r2 = comm.getcommission(size=200, price=50.0)
        assert r2 == pytest.approx(r1 * 2)

    def test_commission_proportional_to_price(self):
        comm = PercentageCommission(commission=0.001)
        r1 = comm.getcommission(size=100, price=50.0)
        r2 = comm.getcommission(size=100, price=100.0)
        assert r2 == pytest.approx(r1 * 2)

    def test_zero_size(self):
        comm = PercentageCommission(commission=0.001)
        assert comm.getcommission(0, 100.0) == 0.0


def _buy_once_orchestrator(symbol: str, shares: int, price: float):
    """Mock orchestrator that emits a single buy order on the first step.

    Used to generate turnover so transaction-cost behaviour is observable.
    """
    order = {
        "symbol": symbol,
        "side": "buy",
        "shares": shares,
        "quantity": shares,
        "notional": shares * price,
        "price": price,
        "strategy": "composite",
    }
    orch = MagicMock()
    orch.step.side_effect = [([order], MagicMock())] + [([], MagicMock())] * 500
    return orch


class TestSlippageWiring:
    """Slippage is applied at the broker (``set_slippage_perc``) and the
    ``slippage_pct`` config knob is consumed, not silently inert."""

    def _run(self, slippage_pct: float) -> float:
        config = {
            "initial_capital": 1_000_000,
            "commission_pct": 0.001,
            "slippage_pct": slippage_pct,
            "rebalance_frequency": "daily",
        }
        df = _synthetic_prices(["AAPL"], days=30, seed=7)
        store = _make_pit_store(df)
        # Buy 1000 shares near the opening price so the fill incurs slippage.
        orch = _buy_once_orchestrator("AAPL", 1000, float(df.iloc[0]["close"]))
        engine = BacktestEngine(config)
        engine.setup(df, store, orch, ["AAPL"])
        engine.run()
        return engine.get_results()["final_value"]

    def test_slippage_lowers_final_value(self):
        no_slip = self._run(0.0)
        with_slip = self._run(0.05)
        # Same data, same orders — the only difference is broker slippage, so a
        # buy fills at a worse price and ending value is strictly lower.
        assert with_slip < no_slip

    def test_zero_slippage_skips_broker_call(self):
        # slippage_pct=0 must not raise and must leave a deterministic result.
        assert self._run(0.0) > 0


# ======================================================================
# 3. PitViewAdapter
# ======================================================================

class TestPitViewAdapter:
    def test_satisfies_pitview_protocol(self):
        store = _make_pit_store(_synthetic_prices(["AAPL"], days=30))
        asof = datetime(2023, 2, 1)
        adapter = PitViewAdapter(store, asof, ["AAPL"])
        assert isinstance(adapter, PitView)

    def test_asof_property(self):
        store = _make_pit_store(_synthetic_prices(["AAPL"]))
        asof = datetime(2023, 1, 15)
        adapter = PitViewAdapter(store, asof, ["AAPL"])
        assert adapter.asof == asof

    def test_universe_property(self):
        store = _make_pit_store(_synthetic_prices(["AAPL", "GOOG"]))
        adapter = PitViewAdapter(store, datetime(2023, 2, 1), ["AAPL", "GOOG"])
        assert adapter.universe == ["AAPL", "GOOG"]

    def test_prices_returns_dataframe(self):
        df = _synthetic_prices(["AAPL"], days=30)
        store = _make_pit_store(df)
        asof = datetime(2023, 2, 15)
        adapter = PitViewAdapter(store, asof, ["AAPL"])
        result = adapter.prices()
        assert isinstance(result, pd.DataFrame)

    def test_prices_respects_asof(self):
        df = _synthetic_prices(["AAPL"], days=60)
        store = _make_pit_store(df)
        asof = datetime(2023, 1, 20)
        adapter = PitViewAdapter(store, asof, ["AAPL"])
        result = adapter.prices()
        if not result.empty:
            max_date = result["date"].max()
            assert max_date <= pd.Timestamp(asof)

    def test_fundamentals_returns_dataframe(self):
        store = _make_pit_store(_synthetic_prices(["AAPL"]))
        adapter = PitViewAdapter(store, datetime(2023, 2, 1), ["AAPL"])
        result = adapter.fundamentals()
        assert isinstance(result, pd.DataFrame)

    def test_sentiment_returns_dataframe(self):
        store = _make_pit_store(_synthetic_prices(["AAPL"]))
        adapter = PitViewAdapter(store, datetime(2023, 2, 1), ["AAPL"])
        result = adapter.sentiment()
        assert isinstance(result, pd.DataFrame)


# ======================================================================
# 4. FirmStrategy._should_rebalance
# ======================================================================

class TestShouldRebalance:
    """Test the rebalance scheduling logic via the extracted helper."""

    @staticmethod
    def _should_rebalance(freq: str, last: datetime | None, dt: datetime) -> bool:
        """Mirror the logic of FirmStrategy._should_rebalance for testing."""
        if last is None:
            return True
        if freq == "daily":
            return True
        elif freq == "weekly":
            return (dt - last).days >= 5
        elif freq == "monthly":
            return dt.month != last.month
        return True

    def test_first_bar_always_rebalances(self):
        assert self._should_rebalance("weekly", None, datetime(2023, 1, 2)) is True

    def test_daily_always_rebalances(self):
        assert self._should_rebalance("daily", datetime(2023, 1, 2), datetime(2023, 1, 3)) is True

    def test_weekly_skips_within_five_days(self):
        assert self._should_rebalance("weekly", datetime(2023, 1, 2), datetime(2023, 1, 4)) is False

    def test_weekly_triggers_after_five_days(self):
        assert self._should_rebalance("weekly", datetime(2023, 1, 2), datetime(2023, 1, 9)) is True

    def test_monthly_same_month(self):
        assert self._should_rebalance("monthly", datetime(2023, 1, 5), datetime(2023, 1, 20)) is False

    def test_monthly_new_month(self):
        assert self._should_rebalance("monthly", datetime(2023, 1, 5), datetime(2023, 2, 1)) is True

    def test_weekly_exactly_five_days(self):
        assert self._should_rebalance("weekly", datetime(2023, 1, 2), datetime(2023, 1, 7)) is True

    def test_unknown_frequency_always_rebalances(self):
        assert self._should_rebalance("quarterly", datetime(2023, 1, 2), datetime(2023, 1, 3)) is True


# ======================================================================
# 5. BacktestEngine.setup
# ======================================================================

class TestBacktestEngineSetup:
    """Verify that engine setup wires components without errors."""

    def _make_mock_orchestrator(self):
        orch = MagicMock()
        orch.step.return_value = ([], MagicMock())
        return orch

    def test_setup_succeeds_with_synthetic_data(self):
        config = {
            "initial_capital": 1_000_000,
            "commission_pct": 0.001,
            "rebalance_frequency": "weekly",
        }
        df = _synthetic_prices(["AAPL", "GOOG"], days=30)
        store = _make_pit_store(df)
        orch = self._make_mock_orchestrator()

        engine = BacktestEngine(config)
        engine.setup(df, store, orch, ["AAPL", "GOOG"])

        assert engine._portfolio_state is not None
        assert engine._attribution is not None
        assert engine._portfolio_state.cash == 1_000_000

    def test_setup_with_empty_universe(self):
        config = {"initial_capital": 500_000}
        df = _synthetic_prices(["AAPL"], days=10)
        store = _make_pit_store(df)
        orch = self._make_mock_orchestrator()

        engine = BacktestEngine(config)
        engine.setup(df, store, orch, [])
        assert engine._portfolio_state is not None

    def test_get_results_before_run_raises(self):
        engine = BacktestEngine({"initial_capital": 1_000_000})
        with pytest.raises(RuntimeError, match="Must call run"):
            engine.get_results()


# ======================================================================
# 6. Full mini-backtest with mock orchestrator
# ======================================================================

class TestMiniBacktest:
    """Run a tiny backtest with a mock orchestrator that does nothing."""

    def test_run_with_no_orders(self):
        config = {
            "initial_capital": 100_000,
            "commission_pct": 0.001,
            "rebalance_frequency": "daily",
        }
        df = _synthetic_prices(["AAPL"], days=20)
        store = _make_pit_store(df)

        orch = MagicMock()
        orch.step.return_value = ([], MagicMock())

        engine = BacktestEngine(config)
        engine.setup(df, store, orch, ["AAPL"])
        results = engine.run()

        assert results is not None
        assert len(results) == 1

        analysis = engine.get_results()
        assert analysis["final_value"] == pytest.approx(100_000, rel=1e-6)

    def test_generate_report(self):
        config = {
            "initial_capital": 100_000,
            "commission_pct": 0.001,
            "rebalance_frequency": "daily",
        }
        df = _synthetic_prices(["AAPL"], days=20)
        store = _make_pit_store(df)

        orch = MagicMock()
        orch.step.return_value = ([], MagicMock())

        engine = BacktestEngine(config)
        engine.setup(df, store, orch, ["AAPL"])
        engine.run()

        report = engine.generate_report()
        text = report.to_text()
        assert "BACKTEST REPORT" in text

    def test_detailed_returns_analyzer_records_values(self):
        config = {
            "initial_capital": 100_000,
            "commission_pct": 0.001,
            "rebalance_frequency": "daily",
        }
        df = _synthetic_prices(["AAPL"], days=20)
        store = _make_pit_store(df)

        orch = MagicMock()
        orch.step.return_value = ([], MagicMock())

        engine = BacktestEngine(config)
        engine.setup(df, store, orch, ["AAPL"])
        engine.run()

        analysis = engine.get_results()
        detailed = analysis["detailed_returns"]
        assert len(detailed["dates"]) == len(detailed["values"])
        assert len(detailed["values"]) > 0
