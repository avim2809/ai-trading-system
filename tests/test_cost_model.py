"""Tests for the extended cost model: bid-ask spread + short-borrow fees.

Ignoring these costs materially overstates backtested performance,
especially for short-heavy strategies (stat_arb, mean_reversion long/short)
— see the ``spread-impact-costs`` remediation item. Covers both the
per-trade spread cost (folded into ``PercentageCommission``, verified in
isolation and end-to-end via ``BacktestEngine``) and the daily short-borrow
accrual (verified end-to-end since it's driven by ``FirmStrategy.next()``
against a real backtrader broker).
"""

from __future__ import annotations

import pandas as pd

from firm.backtest.commissions import PercentageCommission
from firm.backtest.engine import BacktestEngine
from firm.data.pit_store import PointInTimeDataStore


class TestPercentageCommissionSpread:
    def test_spread_pct_is_additive_to_commission(self):
        comm = PercentageCommission(commission=0.001, spread_pct=0.0002)
        cost = comm._getcommission(size=100, price=50.0, pseudoexec=False)
        assert cost == 100 * 50.0 * (0.001 + 0.0002)

    def test_spread_defaults_to_zero(self):
        """Backward compatibility: existing callers that don't pass
        spread_pct must behave exactly as before."""
        comm = PercentageCommission(commission=0.001)
        cost = comm._getcommission(size=100, price=50.0, pseudoexec=False)
        assert cost == 100 * 50.0 * 0.001


def _flat_price_df(symbols: list[str], n_days: int, price: float = 100.0) -> pd.DataFrame:
    """Constant-price OHLCV data so P&L is driven purely by costs, not
    market moves — isolates the cost-model effect being tested."""
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    frames = []
    for sym in symbols:
        frames.append(pd.DataFrame({
            "date": dates,
            "symbol": sym,
            "open": price, "high": price, "low": price, "close": price,
            "volume": 1_000_000.0, "adj_close": price,
        }))
    return pd.concat(frames, ignore_index=True)


class _OpenOnceShortOrchestrator:
    """Shorts a fixed share count of one symbol on the first rebalance,
    then does nothing — so the position (and any accruing borrow cost)
    persists untouched for the rest of the backtest."""

    def __init__(self, symbol: str, shares: float):
        self.config: dict = {}
        self._symbol = symbol
        self._shares = shares
        self._opened = False

    def step(self, context):
        if self._opened:
            return [], None
        self._opened = True
        order = {
            "symbol": self._symbol,
            "shares": -self._shares,
            "quantity": self._shares,
            "price": 100.0,
            "strategy": "test",
            "notional": self._shares * 100.0,
        }
        return [order], None


def _run_short_backtest(short_borrow_annual_pct: float) -> float:
    symbols = ["AAAA"]
    n_days = 40
    prices_df = _flat_price_df(symbols, n_days)

    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices_df)

    orchestrator = _OpenOnceShortOrchestrator(symbol="AAAA", shares=1000)

    config = {
        "initial_capital": 100_000,
        "commission_pct": 0.0,
        "slippage_pct": 0.0,
        "spread_pct": 0.0,
        "short_borrow_annual_pct": short_borrow_annual_pct,
        "rebalance_frequency": "daily",
    }
    engine = BacktestEngine(config)
    engine.setup(prices_df, pit_store, orchestrator, symbols)
    engine.run()
    return engine.cerebro.broker.getvalue()


class TestShortBorrowAccrual:
    def test_short_borrow_cost_reduces_final_nav(self):
        final_with_borrow = _run_short_backtest(short_borrow_annual_pct=0.03)
        final_without_borrow = _run_short_backtest(short_borrow_annual_pct=0.0)

        assert final_with_borrow < final_without_borrow

        # 1000 shares short @ $100 = $100,000 short notional, held for
        # ~39 trading days (opened day 1) at 3%/yr / 252 trading days.
        short_notional = 1000 * 100.0
        expected_days_held = 39
        expected_total_cost = (
            short_notional * 0.03 / 252 * expected_days_held
        )
        actual_cost = final_without_borrow - final_with_borrow
        assert abs(actual_cost - expected_total_cost) < expected_total_cost * 0.05

    def test_zero_short_borrow_pct_is_a_no_op(self):
        """Backward compatibility: default (0.0) must not touch broker cash
        at all, matching pre-existing behaviour for configs that don't set
        short_borrow_annual_pct."""
        final_default = _run_short_backtest(short_borrow_annual_pct=0.0)
        # With zero commission/slippage/spread and a flat price, a short
        # position with no borrow cost should leave NAV exactly at the
        # initial capital (no P&L, no costs).
        assert final_default == 100_000.0

    def test_long_only_backtest_unaffected_by_borrow_config(self):
        """The accrual must only touch short (negative) positions —
        configuring a nonzero rate must not affect a long-only book."""
        symbols = ["AAAA"]
        prices_df = _flat_price_df(symbols, 40)
        pit_store = PointInTimeDataStore()
        pit_store.load(prices=prices_df)

        class _LongOnceOrchestrator:
            def __init__(self):
                self.config: dict = {}
                self._opened = False

            def step(self, context):
                if self._opened:
                    return [], None
                self._opened = True
                return [{
                    "symbol": "AAAA", "shares": 1000, "quantity": 1000,
                    "price": 100.0, "strategy": "test", "notional": 100_000.0,
                }], None

        config = {
            "initial_capital": 100_000,
            "commission_pct": 0.0,
            "slippage_pct": 0.0,
            "spread_pct": 0.0,
            "short_borrow_annual_pct": 0.03,
            "rebalance_frequency": "daily",
        }
        engine = BacktestEngine(config)
        engine.setup(prices_df, pit_store, _LongOnceOrchestrator(), symbols)
        engine.run()

        assert engine.cerebro.broker.getvalue() == 100_000.0


class _BuyOnceOrchestrator:
    """Buys a fixed share count of one symbol on the first rebalance, then
    does nothing — isolates the entry fill's cost from any further trading."""

    def __init__(self, symbol: str, shares: float, price: float = 100.0):
        self.config: dict = {}
        self._symbol = symbol
        self._shares = shares
        self._price = price
        self._opened = False

    def step(self, context):
        if self._opened:
            return [], None
        self._opened = True
        order = {
            "symbol": self._symbol,
            "shares": self._shares,
            "quantity": self._shares,
            "price": self._price,
            "strategy": "test",
            "notional": self._shares * self._price,
        }
        return [order], None


def _run_buy_once_backtest(*, shares: float, market_impact_coefficient: float) -> float:
    """Runs a single buy-and-hold backtest against a constant $100/share,
    1,000,000-share-volume feed (ADV = $100,000,000/day) with all flat
    costs zeroed out, so any NAV shortfall from `initial_capital` is
    attributable purely to the market-impact model."""
    symbols = ["BBBB"]
    n_days = 25
    price = 100.0
    prices_df = _flat_price_df(symbols, n_days, price=price)

    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices_df)

    orchestrator = _BuyOnceOrchestrator(symbol="BBBB", shares=shares, price=price)

    config = {
        "initial_capital": 1_000_000_000,
        "commission_pct": 0.0,
        "slippage_pct": 0.0,
        "spread_pct": 0.0,
        "market_impact_coefficient": market_impact_coefficient,
        "adv_lookback_days": 20,
        "rebalance_frequency": "daily",
    }
    engine = BacktestEngine(config)
    engine.setup(prices_df, pit_store, orchestrator, symbols)
    engine.run()
    return engine.cerebro.broker.getvalue()


class TestMarketImpactModel:
    def test_zero_coefficient_is_a_no_op(self):
        """Backward compatibility: default (0.0) must not touch NAV at all,
        matching pre-existing flat-cost-only behaviour."""
        final = _run_buy_once_backtest(shares=100_000, market_impact_coefficient=0.0)
        assert final == 1_000_000_000.0

    def test_nonzero_coefficient_reduces_nav_on_a_sizeable_order(self):
        # ADV = $100,000,000/day; 500,000 shares @ $100 = $50,000,000 = 50%
        # participation, comfortably large enough that sqrt-law impact is
        # material.
        final = _run_buy_once_backtest(shares=500_000, market_impact_coefficient=0.01)
        assert final < 1_000_000_000.0

        # Expected: coefficient * sqrt(participation) * notional.
        notional = 500_000 * 100.0
        participation = notional / 100_000_000.0
        expected_cost = 0.01 * (participation ** 0.5) * notional
        actual_cost = 1_000_000_000.0 - final
        assert abs(actual_cost - expected_cost) < expected_cost * 0.05

    def test_impact_cost_rate_grows_with_participation(self):
        """The whole point of a size-aware model: a bigger order should pay
        a *higher effective rate* (cost / notional), not just a bigger
        absolute cost — a flat-pct model would charge the same rate
        regardless of size."""
        small_final = _run_buy_once_backtest(shares=1_000, market_impact_coefficient=0.01)
        large_final = _run_buy_once_backtest(shares=500_000, market_impact_coefficient=0.01)

        small_notional = 1_000 * 100.0
        large_notional = 500_000 * 100.0
        small_rate = (1_000_000_000.0 - small_final) / small_notional
        large_rate = (1_000_000_000.0 - large_final) / large_notional

        assert large_rate > small_rate
