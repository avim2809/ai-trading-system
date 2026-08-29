"""Tests for firm.live.capital_gate (real-capital allocation gate).

Pure-function tests only — no live engine required; synthetic
``PortfolioSnapshot`` lists stand in for engine.portfolio.history.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from firm.contracts.models import PortfolioSnapshot
from firm.live.capital_gate import (
    MAX_DRAWDOWN_LIMIT,
    MIN_EXECUTED_ORDERS,
    MIN_TRADING_DAYS,
    _daily_nav_series,
    compute_capital_gate,
)
from firm.live.trade_history import TradeHistoryStore


def _snapshot(day: int, nav: float, hour: int = 15) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        asof=datetime(2026, 1, 1) + timedelta(days=day, hours=hour),
        nav=nav,
    )


def _rising_nav_snapshots(
    n_days: int,
    start: float = 100_000.0,
    daily_return: float = 0.003,
    seed: int = 0,
) -> list[PortfolioSnapshot]:
    """Synthetic NAV series with a steady positive drift + small noise —
    long/duration/trade-count criteria pass, and the Sharpe CI's lower bound
    should be comfortably positive."""
    rng = np.random.default_rng(seed)
    navs = [start]
    for _ in range(n_days - 1):
        navs.append(navs[-1] * (1 + daily_return + rng.normal(0, 0.0005)))
    return [_snapshot(i, nav) for i, nav in enumerate(navs)]


class TestDailyNavSeries:
    def test_empty_snapshots(self):
        assert _daily_nav_series([]).empty

    def test_collapses_intraday_duplicates_to_last(self):
        snapshots = [
            _snapshot(0, 100_000.0, hour=9),
            _snapshot(0, 100_500.0, hour=15),  # same day, later — should win
            _snapshot(1, 101_000.0, hour=9),
        ]
        series = _daily_nav_series(snapshots)
        assert len(series) == 2
        assert list(series.values) == [100_500.0, 101_000.0]

    def test_sorted_regardless_of_input_order(self):
        snapshots = [_snapshot(2, 102.0), _snapshot(0, 100.0), _snapshot(1, 101.0)]
        series = _daily_nav_series(snapshots)
        assert list(series.values) == [100.0, 101.0, 102.0]


class TestComputeCapitalGate:
    def test_clearly_passing_scenario(self):
        snapshots = _rising_nav_snapshots(90, daily_return=0.003, seed=1)
        result = compute_capital_gate(
            snapshots=snapshots,
            executed_order_count=150,
            alerts=[],
            halted=False,
            broker="ibkr_paper",
        )
        assert result["overall_passing"] is True
        assert result["blocking"] == []
        assert result["n_passing"] == 5
        criteria = result["criteria"]
        assert criteria["duration"]["passing"] is True
        assert criteria["trade_count"]["passing"] is True
        assert criteria["realized_sharpe"]["passing"] is True
        assert criteria["max_drawdown"]["passing"] is True
        assert criteria["kill_switch_trips"]["passing"] is True
        assert criteria["llm_ab"] == {
            "label": "LLM A/B", "passing": None, "applicable": False,
        }

    def test_insufficient_data_scenario(self):
        snapshots = _rising_nav_snapshots(5, seed=2)
        result = compute_capital_gate(
            snapshots=snapshots, executed_order_count=3, alerts=[], halted=False,
        )
        realized = result["criteria"]["realized_sharpe"]
        assert realized["passing"] is None
        assert "caveat" in realized
        assert result["overall_passing"] is False
        assert "duration" in result["blocking"]
        assert "realized_sharpe" in result["blocking"]

    def test_duration_and_trade_count_thresholds(self):
        snapshots = _rising_nav_snapshots(MIN_TRADING_DAYS - 1, seed=3)
        result = compute_capital_gate(
            snapshots=snapshots,
            executed_order_count=MIN_EXECUTED_ORDERS - 1,
            alerts=[],
            halted=False,
        )
        assert result["criteria"]["duration"]["passing"] is False
        assert result["criteria"]["trade_count"]["passing"] is False

    def test_max_drawdown_over_limit_fails(self):
        navs = [100_000.0] * 30 + [100_000.0 * (1 - MAX_DRAWDOWN_LIMIT - 0.05)] * 30
        snapshots = [_snapshot(i, nav) for i, nav in enumerate(navs)]
        result = compute_capital_gate(
            snapshots=snapshots, executed_order_count=0, alerts=[], halted=False,
        )
        assert result["criteria"]["max_drawdown"]["passing"] is False
        assert "max_drawdown" in result["blocking"]

    def test_kill_switch_trip_marks_ambiguous_not_failing(self):
        snapshots = _rising_nav_snapshots(90, daily_return=0.003, seed=4)
        alerts = [{"kind": "drawdown_breach", "severity": "critical", "message": "x"}]
        result = compute_capital_gate(
            snapshots=snapshots, executed_order_count=150, alerts=alerts, halted=False,
        )
        criterion = result["criteria"]["kill_switch_trips"]
        assert criterion["value"] == 1
        assert criterion["passing"] is None
        assert criterion["durable"] is False
        assert "kill_switch_trips" in result["blocking"]

    def test_currently_halted_fails_kill_switch_criterion(self):
        snapshots = _rising_nav_snapshots(90, daily_return=0.003, seed=5)
        result = compute_capital_gate(
            snapshots=snapshots, executed_order_count=150, alerts=[], halted=True,
        )
        assert result["criteria"]["kill_switch_trips"]["passing"] is False

    def test_kill_switch_reset_alert_alone_does_not_count_as_trip(self):
        snapshots = _rising_nav_snapshots(90, daily_return=0.003, seed=6)
        alerts = [{"kind": "kill_switch_reset", "severity": "warning", "message": "x"}]
        result = compute_capital_gate(
            snapshots=snapshots, executed_order_count=150, alerts=alerts, halted=False,
        )
        assert result["criteria"]["kill_switch_trips"]["value"] == 0
        assert result["criteria"]["kill_switch_trips"]["passing"] is True

    def test_llm_ab_always_not_applicable(self):
        result = compute_capital_gate(
            snapshots=[], executed_order_count=0, alerts=[], halted=False,
        )
        assert result["criteria"]["llm_ab"] == {
            "label": "LLM A/B", "passing": None, "applicable": False,
        }

    def test_no_snapshots_degrades_safely(self):
        result = compute_capital_gate(
            snapshots=[], executed_order_count=0, alerts=[], halted=False,
        )
        assert result["overall_passing"] is False
        assert result["criteria"]["duration"]["value"] == 0


class TestCountOrders:
    def test_counts_only_matching_status(self, tmp_path):
        store = TradeHistoryStore(
            orders_path=tmp_path / "orders.json", cycles_path=tmp_path / "cycles.json",
        )
        store.record_orders([
            {"order_id": "o1", "symbol": "AAPL", "status": "filled"},
            {"order_id": "o2", "symbol": "MSFT", "status": "cancelled"},
            {"order_id": "o3", "symbol": "GOOG", "status": "filled"},
            {"order_id": "o4", "symbol": "TSLA", "status": "rejected"},
        ])
        assert store.count_orders(status="filled") == 2
        assert store.count_orders() == 4
        assert store.count_orders(status="rejected") == 1

    def test_counts_beyond_list_orders_cap(self, tmp_path):
        store = TradeHistoryStore(
            orders_path=tmp_path / "orders.json", cycles_path=tmp_path / "cycles.json",
        )
        store.record_orders([
            {"order_id": f"o{i}", "symbol": "AAPL", "status": "filled"} for i in range(600)
        ])
        # list_orders caps at 500 by default — count_orders must not.
        assert len(store.list_orders()) == 500
        assert store.count_orders(status="filled") == 600
