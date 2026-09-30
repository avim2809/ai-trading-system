"""Tests for strategy_mode: allocation (firm.allocation + LiveTradingEngine wiring)."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.allocation import (
    SLEEVE_REGISTRY,
    Allocator,
    Sleeve,
    SleeveConfigError,
    StaticSleeve,
    build_allocator,
    build_sleeves,
    is_calendar_rebalance_due,
)
from firm.allocation.calendar import first_trading_day_of_month, is_us_trading_day, nyse_holidays
from firm.brokers.base import BrokerError, BrokerPosition, OrderRequest, OrderStatus
from firm.live.approval import ApprovalQueue
from firm.live.engine import LiveTradingEngine
from tests.test_brokers import MockBroker

# Mid-session US/Eastern timestamps expressed as naive UTC (engine convention).
# 2026-09-01 is a Tuesday (first trading day of September; Labor Day is 9/7).
SEP1_1430_UTC = datetime(2026, 9, 1, 14, 30)   # 10:30 ET
SEP1_1530_UTC = datetime(2026, 9, 1, 15, 30)   # 11:30 ET, same trading day
SEP2_1430_UTC = datetime(2026, 9, 2, 14, 30)
OCT1_1430_UTC = datetime(2026, 10, 1, 14, 30)  # Thursday, first trading day of October

CORE = {"name": "core", "type": "static", "weight": 0.92,
        "weights": {"SPY": 0.6, "IEF": 0.4}, "rebalance": "monthly"}


def _alloc_cfg(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "sleeves": [dict(CORE)],
        "band_abs": 0.02,
        "min_order_notional": 100.0,
        "liquidate_unmanaged": False,
        "max_order_notional": 80_000,
        "kill_switch_drawdown": 0.25,
        "max_daily_turnover": 3.0,
    }
    cfg.update(overrides)
    return cfg


def _history(symbols: list[str], price: float = 100.0, n: int = 300) -> dict[str, pd.Series]:
    idx = pd.bdate_range(end="2026-08-31", periods=n)
    return {s: pd.Series([price] * n, index=idx, name=s) for s in symbols}


class StubFeed:
    """Minimal data feed: fetch_close_history only; refresh() must never run in allocation mode."""

    def __init__(self, universe: list[str], history: dict[str, pd.Series] | None = None) -> None:
        self._universe = list(universe)
        self._history = history or {}
        self.refresh_calls = 0
        self.history_calls = 0

    def refresh(self, asof=None):
        self.refresh_calls += 1
        raise AssertionError("allocation mode must not run the full data refresh")

    def fetch_close_history(self, symbols, asof=None, lookback_days=None):
        self.history_calls += 1
        return {s: self._history[s] for s in symbols if s in self._history}


class AllocBroker(MockBroker):
    def __init__(self, initial_cash: float = 100_000) -> None:
        super().__init__(initial_cash)
        self._prices.update({"SPY": 500.0, "IEF": 95.0, "TLT": 90.0})
        self.submitted: list[OrderRequest] = []
        self.fail_symbols: set[str] = set()

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        self.submitted.append(order)
        if order.symbol in self.fail_symbols:
            raise BrokerError(f"insufficient buying power for {order.symbol}")
        return super().submit_order(order)

    def seed_position(self, symbol: str, qty: float) -> None:
        price = self._prices[symbol]
        self._positions[symbol] = BrokerPosition(
            symbol=symbol, quantity=qty, avg_cost=price, market_value=qty * price,
        )
        self._cash -= qty * price


def _make_engine(
    tmp_path,
    broker: MockBroker | None = None,
    *,
    strategy_mode: str | None = "allocation",
    allocation: dict[str, Any] | None = None,
    state_db: bool = True,
    extra: dict[str, Any] | None = None,
    approval_mode: str = "full_auto",
) -> tuple[LiveTradingEngine, MockBroker, StubFeed, MagicMock]:
    broker = broker or AllocBroker()
    feed = StubFeed(["AAPL", "MSFT"], _history(["SPY", "IEF", "TLT"]))
    config: dict[str, Any] = {
        "initial_capital": 100_000,
        "memory_log_path": str(tmp_path / "decisions.jsonl"),
        "max_position_pct": 0.05,
        "kill_switch_drawdown": 0.08,
        "max_daily_turnover": 0.25,
        "capital_allocation_mode": "sleeved",
    }
    if strategy_mode is not None:
        config["strategy_mode"] = strategy_mode
    if allocation is not None:
        config["allocation"] = allocation
    config.update(extra or {})
    with patch("firm.live.engine.build_orchestrator") as mock_build:
        orch = MagicMock()
        orch.capital_allocation_mode = "blended"
        orch.sleeve_traders = {}
        orch.trader = None
        orch.step.side_effect = AssertionError("orchestrator.step must not run in allocation mode")
        mock_build.return_value = orch
        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=ApprovalQueue(broker=broker),
            approval_mode=approval_mode,
            state_db_path=(tmp_path / "live_state.db") if state_db else None,
        )
    engine.start()
    return engine, broker, feed, orch


def _run_at(engine: LiveTradingEngine, when: datetime, **kwargs):
    with patch("firm.live.engine.utcnow", return_value=when):
        return engine.run_cycle(**kwargs)


# ---------------------------------------------------------------------------
# Calendar + rebalance timing
# ---------------------------------------------------------------------------

class TestCalendar:
    def test_2026_holidays(self):
        h = nyse_holidays(2026)
        for d in [date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
                  date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
                  date(2026, 11, 26), date(2026, 12, 25)]:
            assert d in h, d
        assert len(h) == 10

    def test_observed_rules(self):
        # 2027: Christmas on Saturday -> Friday Dec 24 closed.
        assert date(2027, 12, 24) in nyse_holidays(2027)
        # 2022: New Year's on Saturday is NOT observed on Fri Dec 31, 2021.
        assert is_us_trading_day(date(2021, 12, 31))
        # Good Friday 2025 = April 18.
        assert not is_us_trading_day(date(2025, 4, 18))
        # 2023: New Year's on Sunday -> Monday Jan 2 closed.
        assert not is_us_trading_day(date(2023, 1, 2))

    def test_first_trading_day_of_month(self):
        assert first_trading_day_of_month(2026, 1) == date(2026, 1, 2)   # Jan 1 holiday
        assert first_trading_day_of_month(2026, 8) == date(2026, 8, 3)   # Aug 1 is Saturday
        assert first_trading_day_of_month(2026, 9) == date(2026, 9, 1)
        assert first_trading_day_of_month(2027, 1) == date(2027, 1, 4)   # Fri Jan 1 holiday


class TestRebalanceTiming:
    def test_monthly_due_on_first_trading_day_of_new_month(self):
        last = datetime(2026, 8, 3, 14, 30)
        assert not is_calendar_rebalance_due("monthly", datetime(2026, 8, 31, 18, 0), last)
        assert is_calendar_rebalance_due("monthly", SEP1_1430_UTC, last)

    def test_monthly_not_due_on_holiday_or_weekend_rolls_to_next_trading_day(self):
        last = datetime(2025, 12, 1, 15, 0)
        assert not is_calendar_rebalance_due("monthly", datetime(2026, 1, 1, 16, 0), last)  # holiday
        assert is_calendar_rebalance_due("monthly", datetime(2026, 1, 2, 16, 0), last)
        last = datetime(2026, 7, 1, 15, 0)
        assert not is_calendar_rebalance_due("monthly", datetime(2026, 8, 1, 16, 0), last)  # Saturday
        assert is_calendar_rebalance_due("monthly", datetime(2026, 8, 3, 16, 0), last)

    def test_month_boundary_uses_eastern_time(self):
        last = datetime(2026, 8, 3, 14, 30)
        # 02:00 UTC Sep 1 is still Aug 31 (Monday) evening in New York.
        assert not is_calendar_rebalance_due("monthly", datetime(2026, 9, 1, 2, 0), last)
        # Rebalanced late on Aug 31 ET (= Sep 1 UTC): still counts as August.
        last_late = datetime(2026, 9, 1, 1, 0)
        assert is_calendar_rebalance_due("monthly", SEP1_1430_UTC, last_late)

    def test_monthly_missed_first_day_still_due_later_in_month_but_once(self):
        last = datetime(2026, 8, 3, 14, 30)
        assert is_calendar_rebalance_due("monthly", SEP2_1430_UTC, last)
        assert not is_calendar_rebalance_due("monthly", SEP2_1430_UTC, SEP1_1430_UTC)

    def test_never_rebalanced_is_due_on_trading_days_only(self):
        assert is_calendar_rebalance_due("monthly", SEP2_1430_UTC, None)
        assert not is_calendar_rebalance_due("monthly", datetime(2026, 9, 5, 15, 0), None)  # Sat

    def test_weekly_and_daily(self):
        mon = datetime(2026, 9, 14, 15, 0)
        fri = datetime(2026, 9, 18, 15, 0)
        next_mon = datetime(2026, 9, 21, 15, 0)
        assert not is_calendar_rebalance_due("weekly", fri, mon)
        assert is_calendar_rebalance_due("weekly", next_mon, fri)
        assert is_calendar_rebalance_due("daily", fri, mon)
        assert not is_calendar_rebalance_due("daily", datetime(2026, 9, 18, 19, 0), fri)

    def test_static_sleeve_validation(self):
        with pytest.raises(SleeveConfigError):
            StaticSleeve("x", 0.5, {"SPY": 0.7, "IEF": 0.4})
        with pytest.raises(SleeveConfigError):
            StaticSleeve("x", 0.5, {"SPY": -0.1})
        with pytest.raises(SleeveConfigError):
            StaticSleeve("x", 0.5, {"SPY": 1.0}, rebalance="yearly")


class TestBuildSleeves:
    def test_builds_static(self):
        sleeves = build_sleeves({"sleeves": [dict(CORE)]})
        assert len(sleeves) == 1
        assert sleeves[0].name == "core" and sleeves[0].weight == 0.92
        assert sleeves[0].symbols() == ["SPY", "IEF"]

    def test_total_weight_over_one_rejected(self):
        with pytest.raises(SleeveConfigError, match="exceeds 1.0"):
            build_sleeves({"sleeves": [dict(CORE), {**CORE, "name": "core2", "weight": 0.2}]})

    def test_duplicate_and_unknown_type_rejected(self):
        with pytest.raises(SleeveConfigError, match="Duplicate"):
            build_sleeves({"sleeves": [dict(CORE), dict(CORE, weight=0.01)]})
        with pytest.raises(SleeveConfigError, match="Unknown sleeve type"):
            build_sleeves({"sleeves": [{"name": "a", "type": "nope", "weight": 0.1}]})

    def test_missing_sleeve_module_fails_loudly(self, monkeypatch):
        monkeypatch.setitem(SLEEVE_REGISTRY, "ghost", "firm.allocation._does_not_exist:Ghost")
        with pytest.raises(SleeveConfigError, match="not available"):
            build_sleeves({"sleeves": [{"name": "g", "type": "ghost", "weight": 0.08}]})

    def test_btc_trend_type_is_lazily_registered(self):
        assert SLEEVE_REGISTRY["btc_trend"] == "firm.allocation.btc_trend:BtcTrendSleeve"


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------

PRICES = {"SPY": 500.0, "IEF": 95.0}
LAST_THIS_MONTH = {"core": datetime(2026, 9, 1, 14, 30)}


def _core_allocator(**kw) -> Allocator:
    return Allocator(build_sleeves({"sleeves": [dict(CORE)]}), **kw)


class _WeirdSleeve(Sleeve):
    def __init__(self, name, weight, weights, fractional=False, due=True):
        self.name, self.weight, self._w, self.fractional, self._due = name, weight, weights, fractional, due

    def target_weights(self, asof, history):
        return dict(self._w)

    def is_rebalance_due(self, asof, last_rebalance):
        return self._due

    def symbols(self):
        return list(self._w)


class TestAllocator:
    def test_initial_build_from_cash(self):
        plan = _core_allocator().plan(SEP1_1430_UTC, 100_000, {}, PRICES, {}, {"core": None})
        assert plan.due_sleeves == ["core"] and plan.rebalanced_sleeves == ["core"]
        assert plan.targets == {"SPY": pytest.approx(0.552), "IEF": pytest.approx(0.368)}
        by_sym = {o["symbol"]: o for o in plan.orders}
        assert by_sym["SPY"]["quantity"] == 110 and by_sym["SPY"]["side"] == "buy"
        assert by_sym["IEF"]["quantity"] == 387
        for o in plan.orders:
            assert o["strategy"] == "core" and o["order_type"] == "market"
            assert o["price"] == PRICES[o["symbol"]]
            assert "fractional" not in o
        assert plan.gross_after <= 1.0

    def test_within_band_and_not_due_trades_nothing(self):
        positions = {"SPY": 0.56 * 100_000, "IEF": 0.36 * 100_000}
        plan = _core_allocator().plan(SEP2_1430_UTC, 100_000, positions, PRICES, {}, LAST_THIS_MONTH)
        assert plan.due_sleeves == [] and plan.orders == []

    def test_drift_band_trades_only_drifted_symbol(self):
        positions = {"SPY": 0.60 * 100_000, "IEF": 0.36 * 100_000}
        plan = _core_allocator().plan(SEP2_1430_UTC, 100_000, positions, PRICES, {}, LAST_THIS_MONTH)
        assert plan.drift_symbols == ["SPY"]
        assert [(o["symbol"], o["side"]) for o in plan.orders] == [("SPY", "sell")]
        assert plan.rebalanced_sleeves == []  # drift trade, not a scheduled rebalance

    def test_sells_before_buys(self):
        positions = {"SPY": 0.80 * 100_000, "IEF": 0.10 * 100_000}
        plan = _core_allocator().plan(OCT1_1430_UTC, 100_000, positions, PRICES, {}, LAST_THIS_MONTH)
        sides = [o["side"] for o in plan.orders]
        assert sides == ["sell", "buy"]
        assert plan.orders[0]["symbol"] == "SPY"

    def test_never_shorts_or_levers(self):
        sleeves = [
            _WeirdSleeve("a", 0.5, {"SPY": -0.5, "IEF": 1.5}),  # negative + over-1 weights
            _WeirdSleeve("b", 0.5, {"TLT": 0.0}),
        ]
        alloc = Allocator(sleeves)
        prices = {**PRICES, "TLT": 90.0}
        positions = {"TLT": 9_000.0}  # target 0 -> full exit, never below zero
        plan = alloc.plan(SEP1_1430_UTC, 100_000, positions, prices, {}, {},
                          quantities={"TLT": 100.0})
        by_sym = {o["symbol"]: o for o in plan.orders}
        assert "SPY" not in by_sym  # clamped to 0, nothing held -> nothing to sell
        assert by_sym["TLT"]["side"] == "sell" and by_sym["TLT"]["quantity"] == 100.0
        assert plan.targets["IEF"] == pytest.approx(0.5)  # 1.5 scaled to 1.0 within sleeve
        assert plan.gross_after <= 1.0 + 1e-9

    def test_gross_cap_scales_buys_when_unmanaged_positions_remain(self):
        alloc = _core_allocator()
        positions = {"AAPL": 30_000.0}  # unmanaged, left alone
        plan = alloc.plan(SEP1_1430_UTC, 100_000, positions, {**PRICES, "AAPL": 150.0}, {},
                          {"core": None})
        assert plan.unmanaged == {"AAPL": pytest.approx(0.3)}
        assert all(o["symbol"] != "AAPL" for o in plan.orders)
        assert plan.buy_scale < 1.0
        assert plan.gross_after <= 1.0 + 1e-9

    def test_unmanaged_positions_untouched_by_default(self):
        positions = {"SPY": 55_000.0, "IEF": 36_800.0, "AAPL": 1_500.0}
        plan = _core_allocator().plan(SEP2_1430_UTC, 100_000, positions,
                                      {**PRICES, "AAPL": 150.0}, {}, LAST_THIS_MONTH)
        assert plan.orders == []
        assert "AAPL" in plan.unmanaged

    def test_liquidate_unmanaged_closes_longs_and_covers_shorts(self):
        alloc = _core_allocator(liquidate_unmanaged=True)
        positions = {"SPY": 55_000.0, "IEF": 36_800.0, "AAPL": 1_500.0, "TSLA": -2_500.0}
        plan = alloc.plan(
            SEP2_1430_UTC, 100_000, positions, {**PRICES, "AAPL": 150.0, "TSLA": 250.0}, {},
            LAST_THIS_MONTH, quantities={"SPY": 110, "IEF": 387, "AAPL": 10, "TSLA": -10},
        )
        by_sym = {o["symbol"]: o for o in plan.orders}
        assert by_sym["AAPL"]["side"] == "sell" and by_sym["AAPL"]["quantity"] == 10
        assert by_sym["TSLA"]["side"] == "buy" and by_sym["TSLA"]["quantity"] == 10
        assert by_sym["AAPL"]["strategy"] == "allocation_unmanaged"

    def test_fractional_flag_only_for_fractional_sleeves(self):
        sleeves = [
            StaticSleeve("core", 0.92, {"SPY": 0.6, "IEF": 0.4}),
            _WeirdSleeve("btc", 0.08, {"BTC/USD": 1.0}, fractional=True),
        ]
        plan = Allocator(sleeves).plan(SEP1_1430_UTC, 100_000, {},
                                       {**PRICES, "BTC/USD": 60_000.0}, {}, {})
        by_sym = {o["symbol"]: o for o in plan.orders}
        assert by_sym["BTC/USD"]["fractional"] is True
        assert by_sym["BTC/USD"]["quantity"] == pytest.approx(0.133333, abs=1e-6)
        assert "fractional" not in by_sym["SPY"]

    def test_broker_crypto_symbol_alias_is_managed(self):
        sleeves = [_WeirdSleeve("btc", 0.08, {"BTC/USD": 1.0}, fractional=True, due=False)]
        plan = Allocator(sleeves, liquidate_unmanaged=True).plan(
            SEP2_1430_UTC, 100_000, {"BTCUSD": 8_000.0}, {"BTCUSD": 60_000.0}, {}, {},
            quantities={"BTCUSD": 8_000 / 60_000},
        )
        assert plan.unmanaged == {} and plan.orders == []
        assert plan.actual_weights == {"BTC/USD": pytest.approx(0.08)}

    def test_failed_sleeve_holds_its_symbols(self):
        class Broken(_WeirdSleeve):
            def target_weights(self, asof, history):
                raise RuntimeError("no history")

        sleeves = [StaticSleeve("core", 0.9, {"SPY": 1.0}), Broken("sat", 0.1, {"TLT": 1.0})]
        plan = Allocator(sleeves, liquidate_unmanaged=True).plan(
            SEP1_1430_UTC, 100_000, {"TLT": 5_000.0}, {**PRICES, "TLT": 90.0}, {}, {},
        )
        assert all(o["symbol"] != "TLT" for o in plan.orders)  # not liquidated as "unmanaged"
        assert plan.rebalanced_sleeves == ["core"]
        assert plan.errors

    def test_total_sleeve_weight_validated(self):
        with pytest.raises(SleeveConfigError):
            Allocator([_WeirdSleeve("a", 0.7, {"SPY": 1}), _WeirdSleeve("b", 0.4, {"IEF": 1})])


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineAllocationMode:
    def test_cycle_bypasses_pipeline_and_submits_plan(self, tmp_path):
        engine, broker, feed, orch = _make_engine(tmp_path, allocation=_alloc_cfg())
        result = _run_at(engine, SEP1_1430_UTC)
        assert result.error is None, result.error
        assert result.orders_submitted == 2 and result.orders_failed == 0
        assert orch.step.call_count == 0
        assert feed.refresh_calls == 0 and feed.history_calls == 1
        assert {o.symbol for o in broker.submitted} == {"SPY", "IEF"}
        assert all(isinstance(o.quantity, int) for o in broker.submitted)
        assert all(o.strategy == "core" for o in broker.submitted)
        assert engine._allocation_state["last_rebalance"]["core"] == SEP1_1430_UTC.isoformat()
        # Orchestrator forced blended: dormant pipeline sleeves never drive reconciliation.
        assert engine._config["capital_allocation_mode"] == "blended"
        assert engine._config["pipeline_warmup"] is False

    def test_once_per_trading_day(self, tmp_path):
        engine, broker, _feed, _orch = _make_engine(tmp_path, allocation=_alloc_cfg())
        _run_at(engine, SEP1_1430_UTC)
        n = len(broker.submitted)
        second = _run_at(engine, SEP1_1530_UTC)
        assert second.orders_generated == 0 and len(broker.submitted) == n
        assert engine._allocation_state["day_status"] == "settled"
        # Next day, same month: not due, within band -> runs but trades nothing.
        third = _run_at(engine, SEP2_1430_UTC)
        assert third.error is None and third.orders_generated == 0
        assert engine._allocation_state["day"] == "2026-09-02"

    def test_same_day_retry_only_after_failed_order(self, tmp_path):
        broker = AllocBroker()
        broker.fail_symbols = {"IEF"}
        engine, broker, _feed, _orch = _make_engine(tmp_path, broker, allocation=_alloc_cfg())
        first = _run_at(engine, SEP1_1430_UTC)
        assert first.orders_failed == 1 and first.orders_submitted == 1
        assert engine._allocation_state["day_status"] == "retry"
        assert "core" not in engine._allocation_state["last_rebalance"]
        broker.fail_symbols = set()
        second = _run_at(engine, SEP1_1530_UTC)
        assert [o.symbol for o in broker.submitted[2:]] == ["IEF"]  # SPY already at target
        assert second.orders_submitted == 1
        assert engine._allocation_state["last_rebalance"]["core"] == SEP1_1530_UTC.isoformat()

    def test_retry_limit(self, tmp_path):
        broker = AllocBroker()
        broker.fail_symbols = {"IEF"}
        engine, broker, _f, _o = _make_engine(
            tmp_path, broker, allocation=_alloc_cfg(max_attempts_per_day=2),
        )
        _run_at(engine, SEP1_1430_UTC)
        _run_at(engine, datetime(2026, 9, 1, 15, 0))
        n = len(broker.submitted)
        third = _run_at(engine, SEP1_1530_UTC)
        assert third.orders_generated == 0 and len(broker.submitted) == n

    def test_rejected_after_submission_reverts_and_retries(self, tmp_path):
        engine, broker, _f, _o = _make_engine(tmp_path, allocation=_alloc_cfg())
        _run_at(engine, SEP1_1430_UTC)
        assert engine._allocation_state["day_status"] == "complete"
        ief = next(s for s in broker._orders.values() if s.symbol == "IEF")
        ief.status = "rejected"
        # Simulate the broker never having filled it.
        broker._cash += broker._positions["IEF"].market_value
        del broker._positions["IEF"]
        second = _run_at(engine, SEP1_1530_UTC)
        assert second.orders_submitted == 1
        assert broker.submitted[-1].symbol == "IEF"

    def test_last_rebalance_survives_restart(self, tmp_path):
        engine, broker, _f, _o = _make_engine(tmp_path, allocation=_alloc_cfg())
        _run_at(engine, SEP1_1430_UTC)
        engine.stop()
        engine2, _b, _f2, _o2 = _make_engine(tmp_path, broker, allocation=_alloc_cfg())
        assert engine2._allocation_state["last_rebalance"]["core"] == SEP1_1430_UTC.isoformat()
        assert engine2.allocation_status()["plan"]["targets"]["SPY"] == pytest.approx(0.552)
        n = len(broker.submitted)
        same_day = _run_at(engine2, SEP1_1530_UTC)  # restart mid-day: no second allocation
        assert same_day.orders_generated == 0
        next_day = _run_at(engine2, SEP2_1430_UTC)  # not due again this month
        assert next_day.orders_generated == 0 and len(broker.submitted) == n
        october = _run_at(engine2, OCT1_1430_UTC)
        assert october.error is None
        assert engine2._allocation_state["last_rebalance"]["core"] == OCT1_1430_UTC.isoformat()

    def test_unmanaged_positions_left_alone_by_engine(self, tmp_path):
        broker = AllocBroker()
        broker.seed_position("AAPL", 20)
        engine, broker, _f, _o = _make_engine(tmp_path, broker, allocation=_alloc_cfg())
        _run_at(engine, SEP1_1430_UTC)
        assert "AAPL" not in {o.symbol for o in broker.submitted}
        assert broker._positions["AAPL"].quantity == 20

    def test_liquidate_unmanaged_sells_first(self, tmp_path):
        broker = AllocBroker()
        broker.seed_position("AAPL", 20)
        engine, broker, _f, _o = _make_engine(
            tmp_path, broker, allocation=_alloc_cfg(liquidate_unmanaged=True),
        )
        _run_at(engine, SEP1_1430_UTC)
        assert broker.submitted[0].symbol == "AAPL" and broker.submitted[0].side == "sell"
        assert "AAPL" not in broker._positions

    def test_daily_turnover_cap_defers_without_same_day_retry(self, tmp_path):
        engine, broker, _f, _o = _make_engine(
            tmp_path, allocation=_alloc_cfg(max_daily_turnover=0.25),
        )
        first = _run_at(engine, SEP1_1430_UTC)
        assert first.orders_submitted == 2
        total = sum(o.quantity * broker._prices[o.symbol] for o in broker.submitted)
        assert total <= 0.25 * 100_000 + 1_000  # scaled to the budget (whole-share rounding)
        assert "core" not in engine._allocation_state["last_rebalance"]  # still due
        assert engine._allocation_state["day_status"] == "complete"  # no same-day retry
        n = len(broker.submitted)
        assert _run_at(engine, SEP1_1530_UTC).orders_generated == 0
        assert len(broker.submitted) == n
        # Next trading day the sleeve is still due and continues the transition.
        nxt = _run_at(engine, SEP2_1430_UTC)
        assert nxt.orders_submitted >= 1

    def test_planning_and_extended_hours_cycles_skipped(self, tmp_path):
        engine, broker, _f, _o = _make_engine(tmp_path, allocation=_alloc_cfg())
        result = _run_at(engine, SEP1_1430_UTC, cycle_type="planning")
        assert result.skipped and broker.submitted == []

    def test_kill_switch_and_notional_overrides_in_allocation_mode(self, tmp_path):
        engine, *_ = _make_engine(tmp_path, allocation=_alloc_cfg(max_daily_turnover=2.0))
        assert engine._kill_switch_drawdown == 0.25
        assert engine._allocation_max_order_notional == 80_000
        assert engine._max_daily_turnover == 2.0

    def test_overrides_ignored_in_pipeline_mode(self, tmp_path):
        engine, *_ = _make_engine(tmp_path, strategy_mode=None, allocation=_alloc_cfg())
        assert engine.strategy_mode == "pipeline"
        assert engine._allocator is None
        assert engine._kill_switch_drawdown == 0.08
        assert engine._allocation_max_order_notional is None
        assert engine._max_daily_turnover == 0.25
        assert engine._config["capital_allocation_mode"] == "sleeved"
        assert engine.allocation_status() is None

    def test_without_notional_override_pipeline_cap_blocks_core_order(self, tmp_path):
        cfg = _alloc_cfg()
        cfg.pop("max_order_notional")
        engine, broker, _f, _o = _make_engine(tmp_path, allocation=cfg)
        result = _run_at(engine, SEP1_1430_UTC)
        assert result.orders_failed == 2  # 55k/37k orders vs a 10k (2 x 5% x NAV) cap
        assert broker.submitted == []
        assert engine._allocation_state["day_status"] == "retry"

    def test_kill_switch_halts_allocation(self, tmp_path):
        engine, broker, _f, _o = _make_engine(tmp_path, allocation=_alloc_cfg())
        engine._peak_equity = 200_000  # 50% drawdown vs 25% allocation kill switch
        result = _run_at(engine, SEP1_1430_UTC)
        assert result.halted and broker.submitted == []

    def test_missing_sleeve_module_fails_engine_start(self, tmp_path, monkeypatch):
        monkeypatch.setitem(SLEEVE_REGISTRY, "ghost", "firm.allocation._does_not_exist:Ghost")
        cfg = _alloc_cfg(sleeves=[dict(CORE, weight=0.9),
                                  {"name": "g", "type": "ghost", "weight": 0.08}])
        with pytest.raises(SleeveConfigError):
            _make_engine(tmp_path, allocation=cfg)

    def test_unknown_strategy_mode_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="strategy_mode"):
            _make_engine(tmp_path, strategy_mode="yolo", allocation=_alloc_cfg())

    def test_status_endpoint_exposes_mode_and_plan(self, tmp_path):
        from firm.api.routers.live import live_status

        engine, *_ = _make_engine(tmp_path, allocation=_alloc_cfg())
        _run_at(engine, SEP1_1430_UTC)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            live_engine=engine, live_scheduler=None,
        )))
        status = live_status(request)
        assert status["strategy_mode"] == "allocation"
        alloc = status["allocation"]
        assert alloc["sleeves"][0]["name"] == "core"
        assert alloc["sleeves"][0]["last_rebalance"] == SEP1_1430_UTC.isoformat()
        assert alloc["plan"]["targets"] == {"SPY": pytest.approx(0.552), "IEF": pytest.approx(0.368)}
        assert set(alloc["plan"]["actual_weights"]) == set()  # planned from all-cash
        import json
        json.dumps(status)  # must be JSON-serializable

    def test_status_endpoint_pipeline_mode(self, tmp_path):
        from firm.api.routers.live import live_status

        engine, *_ = _make_engine(tmp_path, strategy_mode=None)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            live_engine=engine, live_scheduler=None,
        )))
        status = live_status(request)
        assert status["strategy_mode"] == "pipeline" and status["allocation"] is None


class TestFractionalExecution:
    def _engine(self, tmp_path):
        engine, broker, _f, _o = _make_engine(tmp_path, strategy_mode=None, state_db=False)
        broker._prices["BTC/USD"] = 60_000.0
        engine._data_feed._universe.append("BTC/USD")
        engine._max_position_pct = 1.0
        return engine, broker

    def test_fractional_quantity_kept_only_when_flagged(self, tmp_path):
        engine, broker = self._engine(tmp_path)
        orders = [
            {"symbol": "BTC/USD", "side": "buy", "quantity": 0.1234567, "price": 60_000.0,
             "strategy": "btc", "fractional": True, "time_in_force": "gtc"},
            {"symbol": "AAPL", "side": "buy", "quantity": 2.6, "price": 150.0, "strategy": "x"},
            {"symbol": "MSFT", "side": "buy", "quantity": 0.4, "price": 300.0, "strategy": "x"},
        ]
        _statuses, failed = engine._execute_orders(orders, cycle_id=1)
        assert failed == []
        sent = {o.symbol: o for o in broker.submitted}
        assert sent["BTC/USD"].quantity == pytest.approx(0.1234567)
        assert sent["BTC/USD"].time_in_force == "gtc"
        assert sent["AAPL"].quantity == 3 and isinstance(sent["AAPL"].quantity, int)
        assert sent["AAPL"].time_in_force == "day"
        assert "MSFT" not in sent  # 0.4 shares rounds to 0 -> dust

    def test_dust_check_respects_flag(self):
        assert LiveTradingEngine._is_dust_order({"quantity": 0.3})
        assert not LiveTradingEngine._is_dust_order({"quantity": 0.3, "fractional": True})
        assert LiveTradingEngine._is_dust_order({"quantity": 0.0, "fractional": True})


# ---------------------------------------------------------------------------
# Config plumbing + data feed
# ---------------------------------------------------------------------------

class TestConfigPlumbing:
    def test_resolve_live_startup_carries_strategy_mode_and_allocation(self):
        from firm.live.provider_utils import resolve_live_startup

        fake_yaml = {"risk": {"kill_switch_drawdown": 0.08}, "strategy_mode": "allocation",
                     "allocation": _alloc_cfg()}
        with patch("firm.live.provider_utils.load_live_yaml_defaults", return_value=fake_yaml):
            resolved = resolve_live_startup()
        cfg = resolved["engine_config"]
        assert cfg["strategy_mode"] == "allocation"
        assert cfg["allocation"]["sleeves"][0]["name"] == "core"
        assert cfg["kill_switch_drawdown"] == 0.08  # pipeline value; the engine overrides it

    def test_absent_strategy_mode_not_injected(self):
        from firm.live.provider_utils import resolve_live_startup

        with patch("firm.live.provider_utils.load_live_yaml_defaults", return_value={"risk": {}}):
            resolved = resolve_live_startup()
        assert "strategy_mode" not in resolved["engine_config"]
        assert "allocation" not in resolved["engine_config"]

    def test_example_config_builds_core_sleeve(self):
        """config/live_alpaca_allocation.example.yaml's core sleeve is valid
        (the btc_trend sleeve comes from a separately-built module)."""
        from pathlib import Path

        import yaml

        path = (Path(__file__).resolve().parents[1] / "config"
                / "live_alpaca_allocation.example.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert cfg["strategy_mode"] == "allocation"
        alloc = cfg["allocation"]
        weights = {s["name"]: s["weight"] for s in alloc["sleeves"]}
        assert sum(weights.values()) <= 1.0
        core = [s for s in alloc["sleeves"] if s["type"] == "static"]
        allocator = build_allocator({**alloc, "sleeves": core})
        assert allocator.symbols() == ["SPY", "IEF"]
        assert alloc["kill_switch_drawdown"] >= 0.2


class TestFetchCloseHistory:
    def test_uses_prices_provider_only_and_drops_forming_bar(self):
        from firm.live.data_feed import LiveDataFeed

        calls: list[list[str]] = []

        class Prov:
            def get_prices(self, symbols, start, end):
                calls.append(list(symbols))
                rows = []
                for sym in symbols:
                    for d, px in [("2026-08-28", 1.0), ("2026-08-31", 2.0), ("2026-09-01", 3.0)]:
                        rows.append({"date": d, "symbol": sym, "close": px, "adj_close": px * 10})
                return pd.DataFrame(rows)

        feed = LiveDataFeed(providers={"prices": Prov()}, universe=["AAPL"])
        hist = feed.fetch_close_history(["SPY", "IEF"], asof=SEP1_1430_UTC)
        assert calls == [["SPY", "IEF"]]
        assert list(hist["SPY"].values) == [10.0, 20.0]  # adj_close; Sep 1 forming bar dropped
        assert hist["IEF"].index[-1] == pd.Timestamp("2026-08-31")

    def test_no_provider_returns_empty(self):
        from firm.live.data_feed import LiveDataFeed

        assert LiveDataFeed(providers={}, universe=[]).fetch_close_history(["SPY"]) == {}
