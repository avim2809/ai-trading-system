"""Tests for the live trading engine, approval queue, and scheduler.

Uses the MockBroker from test_brokers and patches the orchestrator to
avoid real agent pipeline execution.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.agents.blackboard import Blackboard
from firm.brokers.base import (
    BrokerError,
    OrderRequest,
    OrderStatus,
)
from firm.contracts.models import ExecutionReport, RiskDecision, TradeProposal
from firm.live.approval import ApprovalQueue, PendingApproval
from firm.live.data_feed import LiveDataFeed, LivePitViewAdapter
from firm.live.engine import CycleResult, LiveTradingEngine
from firm.live.portfolio_sync import sync_portfolio_from_broker
from firm.portfolio.state import PortfolioState
from firm.time_utils import utcnow

# Re-use MockBroker from test_brokers
from tests.test_brokers import MockBroker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blackboard(asof: datetime | None = None) -> Blackboard:
    bb = Blackboard(asof=asof or utcnow())
    bb.proposal = TradeProposal(
        asof=bb.asof,
        targets={"AAPL": 0.05, "MSFT": 0.05},
    )
    bb.risk_decision = RiskDecision(
        approved=True,
        adjusted_targets={"AAPL": 0.05, "MSFT": 0.05},
    )
    bb.execution_report = ExecutionReport(
        fills=[
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0, "strategy": "momentum"},
            {"symbol": "MSFT", "side": "buy", "quantity": 5, "price": 300.0, "strategy": "trend"},
        ],
        turnover=0.10,
        costs=5.0,
    )
    return bb


def _make_orders() -> list[dict[str, Any]]:
    return [
        {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0, "strategy": "momentum"},
        {"symbol": "MSFT", "side": "buy", "quantity": 5, "price": 300.0, "strategy": "trend"},
    ]


@pytest.fixture()
def engine_components(tmp_path):
    broker = MockBroker()
    feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
    queue = ApprovalQueue(broker=broker)
    # Isolated from the real data/memory/decisions.jsonl — a test writing a
    # decision there (any run_cycle() with a proposal set) would otherwise
    # pollute production data and, via store_decision()'s same-day
    # idempotency check, could silently block a real decision from ever
    # being recorded for that date.
    config = {"initial_capital": 100_000, "memory_log_path": str(tmp_path / "decisions.jsonl")}
    return broker, feed, queue, config


# ---------------------------------------------------------------------------
# ApprovalQueue tests
# ---------------------------------------------------------------------------

class TestApprovalQueue:
    def test_add_and_get_pending(self):
        broker = MockBroker()
        broker.connect()
        queue = ApprovalQueue(broker=broker)
        orders = _make_orders()
        bb = _make_blackboard()

        aid = queue.add(orders=orders, blackboard=bb, strategy="momentum")
        assert len(aid) == 12

        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0].approval_id == aid
        assert pending[0].status == "pending"

    def test_approve_executes_orders(self):
        broker = MockBroker()
        broker.connect()
        queue = ApprovalQueue(broker=broker)

        aid = queue.add(orders=_make_orders(), blackboard=_make_blackboard())
        statuses = queue.approve(aid)

        assert len(statuses) == 2
        assert all(s.status == "filled" for s in statuses)

        approval = queue.get_by_id(aid)
        assert approval.status == "approved"

    def test_reject_marks_rejected(self):
        queue = ApprovalQueue(broker=MockBroker())
        aid = queue.add(orders=_make_orders(), blackboard=_make_blackboard())
        queue.reject(aid, reason="Too risky")

        approval = queue.get_by_id(aid)
        assert approval.status == "rejected"
        assert approval.reject_reason == "Too risky"

    def test_double_approve_raises(self):
        broker = MockBroker()
        broker.connect()
        queue = ApprovalQueue(broker=broker)
        aid = queue.add(orders=_make_orders(), blackboard=_make_blackboard())
        queue.approve(aid)

        with pytest.raises(ValueError, match="not pending"):
            queue.approve(aid)

    def test_approve_nonexistent_raises(self):
        queue = ApprovalQueue(broker=MockBroker())
        with pytest.raises(ValueError, match="not found"):
            queue.approve("doesnotexist")

    def test_expire_stale(self):
        queue = ApprovalQueue(broker=MockBroker(), expiry_minutes=60)
        aid = queue.add(orders=_make_orders(), blackboard=_make_blackboard())

        # Manually backdate the expiry so it's already past
        approval = queue.get_by_id(aid)
        approval.expires_at = utcnow() - timedelta(seconds=5)

        expired = queue.expire_stale()
        assert expired == 1
        assert approval.status == "expired"

    def test_persistence(self, tmp_path):
        persist = tmp_path / "approvals.json"
        broker = MockBroker()
        broker.connect()

        queue1 = ApprovalQueue(broker=broker, persist_path=persist)
        aid = queue1.add(orders=_make_orders(), blackboard=_make_blackboard())

        assert persist.exists()
        data = json.loads(persist.read_text())
        assert len(data) == 1

        queue2 = ApprovalQueue(broker=broker, persist_path=persist)
        assert len(queue2.get_all()) == 1
        assert queue2.get_by_id(aid) is not None

    def test_clear_wipes_pending_and_historical(self):
        broker = MockBroker()
        broker.connect()
        queue = ApprovalQueue(broker=broker)
        aid1 = queue.add(orders=_make_orders(), blackboard=_make_blackboard())
        queue.add(orders=_make_orders(), blackboard=_make_blackboard())
        queue.reject(aid1, reason="test")

        count = queue.clear()

        assert count == 2
        assert queue.get_all() == []
        assert queue.get_pending() == []

    def test_clear_persists_to_disk(self, tmp_path):
        persist = tmp_path / "approvals.json"
        broker = MockBroker()
        broker.connect()
        queue = ApprovalQueue(broker=broker, persist_path=persist)
        queue.add(orders=_make_orders(), blackboard=_make_blackboard())

        queue.clear()

        reloaded = ApprovalQueue(broker=broker, persist_path=persist)
        assert reloaded.get_all() == []

    def test_clear_empty_queue_returns_zero(self):
        queue = ApprovalQueue(broker=MockBroker())
        assert queue.clear() == 0


class TestPendingApproval:
    def test_is_expired(self):
        a = PendingApproval(
            approval_id="test",
            created_at=utcnow() - timedelta(hours=2),
            expires_at=utcnow() - timedelta(hours=1),
            orders=[],
            blackboard_snapshot={},
        )
        assert a.is_expired() is True

    def test_not_expired(self):
        a = PendingApproval(
            approval_id="test",
            created_at=utcnow(),
            expires_at=utcnow() + timedelta(hours=1),
            orders=[],
            blackboard_snapshot={},
        )
        assert a.is_expired() is False


# ---------------------------------------------------------------------------
# PortfolioSync tests
# ---------------------------------------------------------------------------

class TestPortfolioSync:
    def test_sync_detects_cash_mismatch(self):
        broker = MockBroker(initial_cash=50_000)
        broker.connect()
        portfolio = PortfolioState(initial_capital=100_000)

        discreps = sync_portfolio_from_broker(broker, portfolio)
        assert any(d["type"] == "cash_mismatch" for d in discreps)
        assert portfolio.cash == 50_000

    def test_sync_detects_position_mismatch(self):
        broker = MockBroker()
        broker.connect()
        broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=10))

        portfolio = PortfolioState(initial_capital=100_000)
        discreps = sync_portfolio_from_broker(broker, portfolio)

        pos_mismatches = [d for d in discreps if d["type"] == "position_mismatch"]
        assert len(pos_mismatches) >= 1
        assert portfolio.holdings.get("AAPL") == 10

    def test_sync_no_discrepancies(self):
        broker = MockBroker()
        broker.connect()
        portfolio = PortfolioState(initial_capital=100_000)

        discreps = sync_portfolio_from_broker(broker, portfolio)
        assert len(discreps) == 0

    def test_sync_flags_unavailable_open_orders(self):
        """When open orders can't be fetched, reconciliation is flagged
        degraded rather than silently trusting an empty in-flight view."""
        broker = MockBroker()
        broker.connect()

        def _boom():
            raise RuntimeError("broker API down")

        broker.get_open_orders = _boom  # type: ignore[method-assign]
        portfolio = PortfolioState(initial_capital=100_000)

        discreps = sync_portfolio_from_broker(broker, portfolio)
        assert any(d["type"] == "open_orders_unavailable" for d in discreps)


# ---------------------------------------------------------------------------
# LiveDataFeed tests
# ---------------------------------------------------------------------------

class TestLiveDataFeed:
    def test_refresh_returns_pit_view(self):
        feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
        pit_view = feed.refresh()

        assert isinstance(pit_view, LivePitViewAdapter)
        assert pit_view.universe == ["AAPL", "MSFT"]
        assert pit_view.asof is not None

    def test_pit_view_protocol_methods(self, tmp_path, monkeypatch):
        from firm.config import Settings

        settings = Settings()
        settings.data.cache_dir = str(tmp_path)
        monkeypatch.setattr("firm.config.get_settings", lambda: settings)

        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        view = feed.refresh()

        # These should return empty DataFrames when no providers supplied
        prices = view.prices()
        assert prices.empty

        funds = view.fundamentals()
        assert funds.empty

        sent = view.sentiment()
        assert sent.empty

    def test_refresh_merges_cached_fundamentals(self, tmp_path, monkeypatch):
        from firm.data.cache import ParquetCache
        from firm.config import Settings

        monkeypatch.setenv("FIRM_LIVE_FETCH_FUNDAMENTALS", "1")
        monkeypatch.setenv("FIRM_FUNDAMENTALS_REFRESH_MAX_AGE_HOURS", "0")
        cache = ParquetCache(tmp_path)
        cache.put(
            "combined/fundamentals",
            pd.DataFrame({
                "date": ["2024-06-01"],
                "symbol": ["MSFT"],
                "pe_ratio": [30.0],
            }),
        )

        settings = Settings()
        settings.data.cache_dir = str(tmp_path)
        monkeypatch.setattr("firm.config.get_settings", lambda: settings)

        class FundProv:
            def get_fundamentals(self, symbols, start, end):
                return pd.DataFrame({
                    "date": ["2024-06-01"],
                    "symbol": ["AAPL"],
                    "pe_ratio": [25.0],
                })

        class PriceProv:
            def get_prices(self, symbols, start, end):
                return pd.DataFrame()

        feed = LiveDataFeed(
            providers={"fundamentals": FundProv(), "prices": PriceProv()},
            universe=["AAPL", "MSFT"],
        )
        view = feed.refresh()
        funds = view.fundamentals()
        assert set(funds["symbol"]) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# LiveTradingEngine tests
# ---------------------------------------------------------------------------

class TestLiveTradingEngine:
    @pytest.fixture()
    def engine_components(self, tmp_path):
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "memory_log_path": str(tmp_path / "decisions.jsonl")}
        return broker, feed, queue, config

    def _make_engine(self, broker, feed, queue, config, **kwargs):
        return LiveTradingEngine(
            config=config,
            broker=broker,
            data_feed=feed,
            approval_queue=queue,
            **kwargs,
        )

    @patch("firm.live.engine.build_orchestrator")
    def test_start_stop(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config)
        assert engine._started_at is None
        engine.start()
        assert engine.is_running
        assert broker.is_connected()
        # Regression: _started_at was referenced by /live/status's uptime
        # calculation but never actually set anywhere, so uptime_seconds
        # was always null even for a genuinely running engine.
        assert engine._started_at is not None

        engine.stop()
        assert not engine.is_running
        assert not broker.is_connected()
        assert engine._started_at is None

    @patch("firm.live.engine.build_orchestrator")
    def test_run_cycle_full_auto(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        orders = _make_orders()
        bb = _make_blackboard()

        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, bb)
        mock_build.return_value = mock_orch

        engine = self._make_engine(
            broker, feed, queue, config, approval_mode="full_auto"
        )
        engine.start()

        result = engine.run_cycle()
        assert isinstance(result, CycleResult)
        assert result.orders_generated == 2
        assert result.orders_submitted == 2
        assert result.orders_queued == 0
        assert result.error is None

    @patch("firm.live.engine.build_orchestrator")
    def test_run_cycle_semi_auto(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        orders = _make_orders()
        bb = _make_blackboard()

        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, bb)
        mock_build.return_value = mock_orch

        engine = self._make_engine(
            broker, feed, queue, config,
            approval_mode="semi_auto",
            auto_approve_strategies=["trend"],
        )
        engine.start()

        result = engine.run_cycle()
        # "trend" strategy auto-approved, "momentum" queued
        assert result.orders_submitted == 1
        assert result.orders_queued == 1
        assert len(result.approval_ids) == 1

    @patch("firm.live.engine.build_orchestrator")
    def test_run_cycle_no_orders(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components

        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], Blackboard(asof=utcnow()))
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config)
        engine.start()

        result = engine.run_cycle()
        assert result.orders_generated == 0
        assert result.orders_submitted == 0

    @patch("firm.live.engine.build_orchestrator")
    def test_cycle_error_captured(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components

        mock_orch = MagicMock()
        mock_orch.step.side_effect = RuntimeError("pipeline boom")
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config)
        engine.start()

        result = engine.run_cycle()
        assert result.error is not None
        assert "pipeline boom" in result.error

    @patch("firm.live.engine.build_orchestrator")
    def test_drawdown_kill_switch_trips_and_halts(self, mock_build, tmp_path):
        # Broker cash (50k) is far below the 100k starting equity, so the
        # peak-to-trough drawdown (50%) breaches the 10% kill switch.
        broker = MockBroker(initial_cash=50_000)
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "kill_switch_drawdown": 0.1, "memory_log_path": str(tmp_path / "decisions.jsonl")}

        mock_orch = MagicMock()
        mock_orch.step.return_value = (_make_orders(), _make_blackboard())
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
        )
        engine.start()

        result = engine.run_cycle()
        assert result.halted is True
        assert engine.halted is True
        assert result.orders_submitted == 0  # no new orders once halted
        assert any(a["kind"] == "drawdown_breach" for a in result.alerts)
        assert any(a["kind"] == "drawdown_breach" for a in engine.alerts)

    @patch("firm.live.engine.build_orchestrator")
    def test_drawdown_kill_switch_persists_and_survives_restart(self, mock_build, tmp_path):
        state_path = tmp_path / "kill_switch_state.json"
        broker = MockBroker(initial_cash=50_000)
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "kill_switch_drawdown": 0.1, "memory_log_path": str(tmp_path / "decisions.jsonl")}

        mock_orch = MagicMock()
        mock_orch.step.return_value = (_make_orders(), _make_blackboard())
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
            kill_switch_state_path=state_path,
        )
        engine.start()
        engine.run_cycle()
        assert engine.halted is True
        assert state_path.exists()
        saved = json.loads(state_path.read_text())
        assert saved["halted"] is True
        assert "reason" in saved

        # A fresh engine instance (simulating a process restart) pointed at
        # the same state file must come up already halted.
        engine2 = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
            kill_switch_state_path=state_path,
        )
        assert engine2.halted is True

    def test_kill_switch_state_not_persisted_by_default(self, tmp_path):
        # Default construction (no kill_switch_state_path) must not touch
        # disk at all — every other test in this module relies on this.
        broker = MockBroker(initial_cash=100_000)
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "memory_log_path": str(tmp_path / "decisions.jsonl")}
        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )
        assert engine._kill_switch_state_path is None

    @patch("firm.live.engine.build_orchestrator")
    def test_reset_kill_switch_rearms_trading(self, mock_build, tmp_path):
        state_path = tmp_path / "kill_switch_state.json"
        broker = MockBroker(initial_cash=50_000)
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "kill_switch_drawdown": 0.1, "memory_log_path": str(tmp_path / "decisions.jsonl")}

        mock_orch = MagicMock()
        mock_orch.step.return_value = (_make_orders(), _make_blackboard())
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
            kill_switch_state_path=state_path,
        )
        engine.start()
        engine.run_cycle()
        assert engine.halted is True

        result = engine.reset_kill_switch()
        assert result["halted"] is False
        assert engine.halted is False
        saved = json.loads(state_path.read_text())
        assert saved["halted"] is False
        assert any(a["kind"] == "kill_switch_reset" for a in engine.alerts)

        # Re-armed: a subsequent cycle can submit orders again (nav hasn't
        # moved further, so the new peak == current nav means no immediate
        # re-trip).
        result2 = engine.run_cycle()
        assert result2.halted is False

    @patch("firm.live.engine.build_orchestrator")
    def test_alert_callback_invoked(self, mock_build, tmp_path):
        received: list[dict] = []
        broker = MockBroker(initial_cash=50_000)
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "kill_switch_drawdown": 0.1, "memory_log_path": str(tmp_path / "decisions.jsonl")}

        mock_build.return_value = MagicMock()
        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, alert_callback=received.append,
        )
        engine.start()
        engine.run_cycle()
        assert any(a["kind"] == "drawdown_breach" for a in received)

    @patch("firm.live.engine.build_orchestrator")
    def test_broker_unavailable_alert(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_build.return_value = MagicMock()
        engine = self._make_engine(broker, feed, queue, config)
        engine.start()

        def _down(*_a, **_k):
            raise BrokerError("connection lost")

        broker.get_current_prices = _down  # type: ignore[method-assign]
        result = engine.run_cycle()
        assert result.error is not None
        assert any(a["kind"] == "broker_unavailable" for a in result.alerts)

    @patch("firm.live.engine.build_orchestrator")
    def test_healthy_cycle_emits_no_alerts(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], _make_blackboard())
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config)
        engine.start()
        result = engine.run_cycle()
        assert result.alerts == []
        assert engine.halted is False

    @patch("firm.live.engine.build_orchestrator")
    def test_cycle_history_tracking(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components

        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], Blackboard(asof=utcnow()))
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config)
        engine.start()

        engine.run_cycle()
        engine.run_cycle()
        engine.run_cycle()

        assert len(engine.cycle_history) == 3
        assert engine.cycle_history[0].cycle_id == 1
        assert engine.cycle_history[2].cycle_id == 3

    @patch("firm.live.engine.build_orchestrator")
    def test_clear_cycle_history_wipes_history(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components

        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], Blackboard(asof=utcnow()))
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config)
        engine.start()
        engine.run_cycle()
        engine.run_cycle()

        count = engine.clear_cycle_history()

        assert count == 2
        assert engine.cycle_history == []

    @patch("firm.live.engine.build_orchestrator")
    def test_clear_cycle_history_empty_returns_zero(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_build.return_value = MagicMock()
        engine = self._make_engine(broker, feed, queue, config)
        engine.start()

        assert engine.clear_cycle_history() == 0

    @patch("firm.live.engine.build_orchestrator")
    def test_full_approval_flow(self, mock_build, engine_components):
        """Semi-auto cycle -> queue -> approve -> orders execute."""
        broker, feed, queue, config = engine_components
        orders = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0, "strategy": "momentum"},
        ]
        bb = _make_blackboard()

        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, bb)
        mock_build.return_value = mock_orch

        engine = self._make_engine(
            broker, feed, queue, config, approval_mode="semi_auto"
        )
        engine.start()

        result = engine.run_cycle()
        assert result.orders_queued == 1
        aid = result.approval_ids[0]

        pending = queue.get_pending()
        assert len(pending) == 1

        statuses = queue.approve(aid)
        assert len(statuses) == 1
        assert statuses[0].status == "filled"

        assert broker.get_position("AAPL") is not None


class TestOrdersToFillsAttributionWiring:
    """Coverage for LiveTradingEngine._orders_to_fills, the defensive
    normalizer between order dicts (``side`` + unsigned ``quantity``) and
    the signed-``shares`` fill format PerformanceAttribution.record_trades
    expects. Real ExecutionAgent-produced orders already carry ``shares``,
    but a bare mocked orchestrator (as used throughout this test module)
    does not — without this normalizer, record_trades() would silently
    KeyError (swallowed by a broad except) and record nothing."""

    def test_orders_to_fills_signs_shares_by_side(self):
        orders = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0, "strategy": "momentum"},
            {"symbol": "MSFT", "side": "sell", "quantity": 5, "price": 300.0, "strategy": "trend"},
        ]
        fills = LiveTradingEngine._orders_to_fills(orders)
        assert fills[0] == {"symbol": "AAPL", "shares": 10.0, "price": 150.0, "strategy": "momentum"}
        assert fills[1] == {"symbol": "MSFT", "shares": -5.0, "price": 300.0, "strategy": "trend"}

    def test_orders_to_fills_defaults_missing_strategy_to_composite(self):
        fills = LiveTradingEngine._orders_to_fills(
            [{"symbol": "AAPL", "side": "buy", "quantity": 1, "price": 100.0}]
        )
        assert fills[0]["strategy"] == "composite"

    @patch("firm.live.engine.build_orchestrator")
    def test_full_auto_cycle_populates_attribution_strategy_holdings(
        self, mock_build, engine_components,
    ):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_orch.step.return_value = (_make_orders(), _make_blackboard())
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
        )
        engine.start()
        engine.run_cycle()

        assert engine._attribution.trade_log, (
            "record_trades() must succeed (not be silently swallowed) so "
            "live trade attribution is actually recorded"
        )
        holdings = engine._attribution._strategy_holdings
        assert holdings.get("momentum", {}).get("AAPL") == 10.0
        assert holdings.get("trend", {}).get("MSFT") == 5.0


class TestDurableLiveState:
    """SQLite-backed persistence of portfolio history + attribution state
    (firm.live.state_store.LiveStateStore), wired via LiveTradingEngine's
    optional ``state_db_path``."""

    def _make_engine(self, broker, feed, queue, config, **kwargs):
        return LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, **kwargs,
        )

    def test_disabled_by_default_no_disk_access(self, engine_components):
        broker, feed, queue, config = engine_components
        engine = self._make_engine(broker, feed, queue, config)
        assert engine._state_store is None

    @patch("firm.live.engine.build_orchestrator")
    def test_portfolio_history_and_attribution_persist_across_restart(
        self, mock_build, tmp_path,
    ):
        db_path = tmp_path / "live_state.db"
        broker = MockBroker(initial_cash=100_000)
        feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "memory_log_path": str(tmp_path / "decisions.jsonl")}

        mock_orch = MagicMock()
        mock_orch.step.return_value = (_make_orders(), _make_blackboard())
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
            state_db_path=db_path,
        )
        engine.start()
        engine.run_cycle()
        engine.run_cycle()

        assert db_path.exists()
        assert len(engine.portfolio.history) == 2
        assert "momentum" in engine._attribution.strategies

        engine.stop()

        # A fresh engine instance (simulating a process restart) pointed at
        # the same database must come up with the prior run's NAV history
        # and attribution state already restored — cash/holdings are still
        # (correctly) re-seeded from the broker, not from this database.
        engine2 = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
            state_db_path=db_path,
        )
        assert len(engine2.portfolio.history) == 2
        assert "momentum" in engine2._attribution.strategies
        assert engine2._attribution.trade_log == engine._attribution.trade_log

    def test_no_state_db_path_never_creates_file(self, tmp_path, engine_components):
        broker, feed, queue, config = engine_components
        would_be_path = tmp_path / "should_not_exist.db"
        engine = self._make_engine(broker, feed, queue, config)
        engine.start()
        engine.stop()
        assert not would_be_path.exists()

    @patch("firm.live.engine.build_orchestrator")
    def test_missing_state_db_starts_clean_without_error(self, mock_build, tmp_path):
        # First-run case: state_db_path is configured but the file doesn't
        # exist yet — must not raise, and must start with empty history.
        db_path = tmp_path / "fresh_live_state.db"
        broker = MockBroker(initial_cash=100_000)
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "memory_log_path": str(tmp_path / "decisions.jsonl")}
        mock_build.return_value = MagicMock()

        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, state_db_path=db_path,
        )
        assert engine.portfolio.history == []
        assert engine._attribution.strategies == []


class TestReflectionPersistence:
    """Regression tests: reflection must survive an engine process restart.

    Previously the "previous decision" pointer used for deferred reflection
    was in-memory only (_prev_cycle_date/_prev_cycle_nav), so a restart
    between the decision and its outcome silently dropped the reflection
    forever. The fix persists nav_at_decision in the memory log itself and
    has _maybe_reflect read pending decisions back from disk.
    """

    @patch("firm.live.engine.build_orchestrator")
    def test_reflection_survives_engine_restart(self, mock_build, tmp_path):
        memory_path = tmp_path / "decisions.jsonl"
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "memory_log_path": str(memory_path)}

        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], _make_blackboard())
        mock_build.return_value = mock_orch

        # "Process 1": makes a decision, then the process is discarded —
        # simulated by simply never reusing this engine instance again.
        engine1 = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine1.start()
        engine1.run_cycle()

        entries = [json.loads(line) for line in memory_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["status"] == "pending"
        assert entries[0]["nav_at_decision"] == pytest.approx(engine1.portfolio.nav)

        # "Process 2": a brand-new engine instance with no in-memory
        # knowledge of process 1's decision, pointed at the same log file.
        engine2 = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine2._llm_service = MagicMock()
        engine2._llm_service.chat_json.return_value = {
            "verdict": "correct",
            "what_worked": "the thesis held",
            "what_failed": "",
            "lesson": "trust the signal",
        }
        engine2.start()
        engine2.run_cycle()

        entries = [json.loads(line) for line in memory_path.read_text().splitlines()]
        reflected = [e for e in entries if e["status"] == "reflected"]
        assert len(reflected) == 1
        assert reflected[0]["reflection"] == (
            "CORRECT. What worked: the thesis held Lesson: trust the signal"
        )
        assert reflected[0]["verdict"] == "correct"
        assert reflected[0]["lesson"] == "trust the signal"
        assert reflected[0]["date"] == entries[0]["date"]

    @patch("firm.live.engine.build_orchestrator")
    def test_missing_llm_service_logs_warning(self, mock_build, tmp_path, caplog):
        memory_path = tmp_path / "decisions.jsonl"
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "memory_log_path": str(memory_path)}

        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], _make_blackboard())
        mock_build.return_value = mock_orch

        engine1 = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine1.start()
        engine1.run_cycle()

        engine2 = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )
        # Force LLM construction to fail, as it would with no reachable
        # provider/credentials — reflection must be skipped loudly, not
        # silently, so an operator can notice via the logs.
        with patch("firm.llm.provider.LLMService", side_effect=RuntimeError("no creds")):
            engine2.start()
            with caplog.at_level("WARNING"):
                engine2.run_cycle()

        assert any("LLM service unavailable" in r.message for r in caplog.records)
        assert any("Skipping reflection" in r.message for r in caplog.records)


class TestEngineConfigUpdates:
    """Regression tests: strategies/risk must be genuinely mutable on a
    running engine, not silently ignored (previously PUT /live/config
    accepted these fields but the handler never applied them, and
    /live/start never threaded them into the engine at all).
    """

    @patch("firm.live.engine.build_orchestrator")
    def test_update_strategies_rebuilds_orchestrator(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_build.return_value = MagicMock()
        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)

        assert mock_build.call_count == 1
        engine.update_strategies(["momentum", "trend"])
        assert engine.enabled_strategies == ["momentum", "trend"]
        assert mock_build.call_count == 2
        # New config passed to build_orchestrator reflects the update.
        assert mock_build.call_args[0][0]["strategies"] == ["momentum", "trend"]

    @patch("firm.live.engine.build_orchestrator")
    def test_update_strategies_empty_defaults_to_all(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_build.return_value = MagicMock()
        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)

        engine.update_strategies([])
        assert len(engine.enabled_strategies) > 1  # falls back to all registered strategies

    @patch("firm.live.engine.build_orchestrator")
    def test_update_risk(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_build.return_value = MagicMock()
        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)

        engine.update_risk(kill_switch_drawdown=0.2, max_daily_trades=5, max_daily_turnover=0.3)
        assert engine.risk_config == {
            "kill_switch_drawdown": 0.2, "max_daily_trades": 5, "max_daily_turnover": 0.3,
        }

    @patch("firm.live.engine.build_orchestrator")
    def test_daily_trade_limit_forces_manual_approval(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        orders = _make_orders()
        bb = _make_blackboard()
        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, bb)
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config={**config, "max_daily_trades": 1}, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="semi_auto",
        )
        engine.start()

        result = engine.run_cycle()
        assert result.orders_submitted == 0
        assert result.orders_queued == 2
        assert any(a["kind"] == "daily_limit_breach" for a in result.alerts)
        # Safely routed to manual approval — noteworthy, not an emergency.
        alert = next(a for a in result.alerts if a["kind"] == "daily_limit_breach")
        assert alert["severity"] == "warning"

    @patch("firm.live.engine.build_orchestrator")
    def test_daily_trade_limit_full_auto_still_submits(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        orders = _make_orders()
        bb = _make_blackboard()
        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, bb)
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config={**config, "max_daily_trades": 1}, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
        )
        engine.start()

        result = engine.run_cycle()
        assert result.orders_submitted == 2
        assert result.orders_queued == 0
        assert any(a["kind"] == "daily_limit_breach" for a in result.alerts)
        # full_auto means these orders proceeded unchecked past a configured
        # risk guardrail with no human in the loop — that's "at risk".
        alert = next(a for a in result.alerts if a["kind"] == "daily_limit_breach")
        assert alert["severity"] == "critical"

    @patch("firm.live.engine.build_orchestrator")
    def test_daily_turnover_limit_forces_manual_approval(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        orders = _make_orders()  # 10*150 + 5*300 = 3000 notional
        bb = _make_blackboard()
        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, bb)
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config={**config, "max_daily_turnover": 0.01}, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="semi_auto",
        )
        engine.start()

        result = engine.run_cycle()
        assert result.orders_submitted == 0
        assert result.orders_queued == 2
        assert any(a["kind"] == "daily_limit_breach" for a in result.alerts)

    @patch("firm.live.engine.build_orchestrator")
    def test_daily_turnover_limit_full_auto_still_submits(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        orders = _make_orders()
        bb = _make_blackboard()
        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, bb)
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config={**config, "max_daily_turnover": 0.01}, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
        )
        engine.start()

        result = engine.run_cycle()
        assert result.orders_submitted == 2
        assert result.orders_queued == 0
        assert any(a["kind"] == "daily_limit_breach" for a in result.alerts)
        alert = next(a for a in result.alerts if a["kind"] == "daily_limit_breach")
        assert alert["severity"] == "critical"

    @patch("firm.live.engine.build_orchestrator")
    def test_within_daily_limits_executes_normally(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        orders = _make_orders()
        bb = _make_blackboard()
        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, bb)
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
        )
        engine.start()

        result = engine.run_cycle()
        assert result.orders_submitted == 2
        assert result.orders_queued == 0
        assert not any(a["kind"] == "daily_limit_breach" for a in result.alerts)


class TestTradingDayTimezone:
    def test_trading_day_key_uses_session_timezone_not_utc(self):
        from datetime import datetime, timezone as dt_tz

        from firm.live.scheduler import trading_day_key

        # 2026-07-26 03:00 UTC = 2026-07-25 23:00 US/Eastern (EDT)
        ts = datetime(2026, 7, 26, 3, 0, tzinfo=dt_tz.utc)
        assert trading_day_key(ts, "US/Eastern") == "2026-07-25"
        assert trading_day_key(ts.replace(tzinfo=None), "US/Eastern") == "2026-07-25"
        # Same instant, UTC calendar date would be 2026-07-26
        assert ts.astimezone(dt_tz.utc).strftime("%Y-%m-%d") == "2026-07-26"

    @patch("firm.live.engine.build_orchestrator")
    def test_daily_limits_reset_on_trading_day_boundary(self, mock_build, engine_components):
        from datetime import datetime

        broker, feed, queue, config = engine_components
        mock_build.return_value = MagicMock()
        engine = LiveTradingEngine(
            config={**config, "schedule_timezone": "US/Eastern", "max_daily_trades": 100},
            broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine._daily_date = "2026-07-25"
        engine._daily_trade_count = 7

        # Still July 25 in US/Eastern — counters must not reset.
        late_et = datetime(2026, 7, 26, 3, 0)  # naive UTC
        engine._check_daily_limits(late_et, [], {})
        assert engine._daily_date == "2026-07-25"
        assert engine._daily_trade_count == 7

        # US/Eastern midnight passed — new trading day.
        new_day = datetime(2026, 7, 26, 5, 0)  # 01:00 ET on Jul 26
        engine._check_daily_limits(new_day, [], {})
        assert engine._daily_date == "2026-07-26"
        assert engine._daily_trade_count == 0


class TestCycleWatchdog:
    """Regression coverage for a real 34-hour incident: a scheduled cycle
    hung on a stale IBKR connection (after IB Gateway's mandatory daily
    restart) with zero error, zero alert, and zero log output — silently
    blocking every subsequent cycle indefinitely with no visibility at all.
    """

    @patch("firm.live.engine.build_orchestrator")
    def test_normal_cycle_cancels_watchdog_without_alert(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], _make_blackboard())
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config={**config, "cycle_watchdog_seconds": 5.0},
            broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine.start()
        engine.run_cycle()

        # Cancellation happens synchronously in run_cycle()'s finally block,
        # so this is deterministic — no need to race a sleep against the
        # 5s watchdog to prove it didn't fire.
        assert engine._watchdog_timer is None
        assert not any(a["kind"] == "cycle_watchdog_timeout" for a in engine.alerts)

    @patch("firm.live.engine.build_orchestrator")
    def test_hung_cycle_triggers_watchdog_alert(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()

        def _hang(*_a, **_k):
            time.sleep(0.5)
            return ([], _make_blackboard())

        mock_orch.step.side_effect = _hang
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config={**config, "cycle_watchdog_seconds": 0.1},
            broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine.start()
        engine.run_cycle()

        alerts = [a for a in engine.alerts if a["kind"] == "cycle_watchdog_timeout"]
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"
        assert "Cycle 1" in alerts[0]["message"]

    def test_watchdog_alert_fires_directly(self, engine_components):
        broker, feed, queue, config = engine_components
        with patch("firm.live.engine.build_orchestrator", return_value=MagicMock()):
            engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        engine._on_cycle_watchdog_timeout(7)
        alerts = engine.alerts
        assert len(alerts) == 1
        assert alerts[0]["kind"] == "cycle_watchdog_timeout"
        assert alerts[0]["severity"] == "critical"


class TestCurrentCycleRunningSeconds:
    """Regression coverage for a real incident where the watchdog *alert*
    itself never fired during a genuine 24+ hour hung cycle (found while
    reading production logs — a scheduled market-open cycle silently
    blocked every cycle for the rest of that day and all of the next with
    zero alert, zero error, zero log output). Whatever caused that thread
    callback to go silent, an operator/GUI still needs a way to notice a
    stuck cycle that doesn't depend on that same alert path succeeding —
    a plain clock read of when the current cycle started, independent of
    threading.Timer, answers that.
    """

    @patch("firm.live.engine.build_orchestrator")
    def test_none_when_idle(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_build.return_value = MagicMock()
        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        assert engine.current_cycle_running_seconds is None

    @patch("firm.live.engine.build_orchestrator")
    def test_reports_elapsed_time_while_a_cycle_is_running(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()

        def _hang(*_a, **_k):
            # While the pipeline is "running", the running-seconds clock
            # must already report elapsed time — not just after the fact.
            assert engine.current_cycle_running_seconds >= 0
            time.sleep(0.05)
            return ([], _make_blackboard())

        mock_orch.step.side_effect = _hang
        mock_build.return_value = mock_orch
        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        engine.start()
        engine.run_cycle()

        # Cleared again once the cycle finishes.
        assert engine.current_cycle_running_seconds is None


class TestMarketSessionSync:
    def test_had_cycle_today_false_when_history_empty(self, engine_components):
        broker, feed, queue, config = engine_components
        with patch("firm.live.engine.build_orchestrator", return_value=MagicMock()):
            engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        assert engine.had_cycle_today() is False

    def test_had_cycle_today_true_after_run(self, engine_components):
        broker, feed, queue, config = engine_components
        with patch("firm.live.engine.build_orchestrator", return_value=MagicMock()):
            engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        engine._cycle_history.append(
            CycleResult(cycle_id=1, timestamp=utcnow())
        )
        assert engine.had_cycle_today() is True

    # NOTE: a test_catch_up_starts_cycle_when_market_open test used to live
    # here, exercising maybe_catch_up_session_cycle() against a real
    # LiveTradingEngine via a timed 5s poll loop for engine.had_cycle_today().
    # It was flaky under CI load (racing a background daemon thread against
    # the poll deadline) and has been replaced by deterministic,
    # threading.Event-based coverage of the same function in
    # tests/test_scheduler.py (see test_starts_catch_up_cycle_when_market_open),
    # which uses a duck-typed mock engine instead of a full one.


class TestCycleHardTimeout:
    @patch("firm.live.engine.build_orchestrator")
    def test_hard_timeout_releases_lock_and_alerts(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()

        def _hang(*_a, **_k):
            time.sleep(2.0)
            return ([], _make_blackboard())

        mock_orch.step.side_effect = _hang
        mock_build.return_value = mock_orch

        engine = LiveTradingEngine(
            config={**config, "cycle_hard_timeout_seconds": 0.2},
            broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine.start()
        result = engine.run_cycle()
        assert "hard timeout" in (result.error or "")
        assert any(a["kind"] == "cycle_hard_timeout" for a in result.alerts)

        # Lock must be released — a second cycle should not be skipped.
        mock_orch.step.side_effect = None
        mock_orch.step.return_value = ([], _make_blackboard())
        result2 = engine.run_cycle()
        assert result2.skipped is False


class TestEngineShutdownHardening:
    @patch("firm.live.engine.build_orchestrator")
    def test_stop_returns_while_cycle_worker_hung(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        release = threading.Event()
        mock_build.return_value = MagicMock()

        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine.start()
        engine._cycle_executor.submit(lambda: release.wait(timeout=30))
        time.sleep(0.1)

        t0 = time.time()
        engine.stop()
        elapsed = time.time() - t0
        release.set()

        assert elapsed < 10
        assert not engine.is_running

    @patch("firm.live.engine.build_orchestrator")
    def test_run_cycle_skipped_when_shutting_down(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_build.return_value = MagicMock()
        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine.start()
        engine._shutting_down = True
        result = engine.run_cycle()
        assert result.skipped is True
        assert result.error == "skipped: engine shutting down"


class TestMarketHoursGate:
    """Regression tests: a cycle must not burn a full pipeline pass (and
    IBKR's misleading off-hours quotes) against a closed market.
    """

    @patch("firm.live.engine.build_orchestrator")
    def test_skips_cleanly_when_market_closed(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_build.return_value = mock_orch
        broker._market_open = False

        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        engine.start()

        result = engine.run_cycle()
        assert result.skipped is True
        assert result.error == "skipped: market closed"
        # No pipeline work should have happened at all.
        mock_orch.step.assert_not_called()

    @patch("firm.live.engine.build_orchestrator")
    def test_force_bypasses_market_hours_check(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], _make_blackboard())
        mock_build.return_value = mock_orch
        broker._market_open = False

        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        engine.start()

        result = engine.run_cycle(force=True)
        assert result.skipped is False
        mock_orch.step.assert_called_once()

    @patch("firm.live.engine.build_orchestrator")
    def test_respect_market_hours_false_always_runs(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], _make_blackboard())
        mock_build.return_value = mock_orch
        broker._market_open = False

        engine = LiveTradingEngine(
            config={**config, "respect_market_hours": False},
            broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine.start()

        result = engine.run_cycle()
        assert result.skipped is False
        mock_orch.step.assert_called_once()

    @patch("firm.live.engine.build_orchestrator")
    def test_broken_market_hours_check_fails_open(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], _make_blackboard())
        mock_build.return_value = mock_orch
        broker.is_market_open = MagicMock(side_effect=RuntimeError("boom"))

        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        engine.start()

        result = engine.run_cycle()
        # A broken check must never silently prevent every future cycle.
        assert result.skipped is False
        mock_orch.step.assert_called_once()

    @patch("firm.live.engine.build_orchestrator")
    def test_runs_normally_when_market_open(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], _make_blackboard())
        mock_build.return_value = mock_orch
        assert broker._market_open is True

        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        engine.start()

        result = engine.run_cycle()
        assert result.skipped is False
        mock_orch.step.assert_called_once()


class TestLiveEngineHardening:
    """Regression tests for the live-execution audit fixes."""

    @pytest.fixture()
    def engine_components(self, tmp_path):
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "memory_log_path": str(tmp_path / "decisions.jsonl")}
        return broker, feed, queue, config

    def _make_engine(self, broker, feed, queue, config, **kwargs):
        return LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, **kwargs,
        )

    @patch("firm.live.engine.build_orchestrator")
    def test_orders_carry_idempotency_token(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components

        class RecordingBroker(MockBroker):
            def __init__(self):
                super().__init__()
                self.client_ids = []

            def submit_order(self, order):
                self.client_ids.append(order.client_order_id)
                return super().submit_order(order)

        broker = RecordingBroker()
        queue = ApprovalQueue(broker=broker)
        mock_orch = MagicMock()
        mock_orch.step.return_value = (_make_orders(), _make_blackboard())
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config, approval_mode="full_auto")
        engine.start()
        engine.run_cycle()

        assert len(broker.client_ids) == 2
        assert all(cid for cid in broker.client_ids), "client_order_id must be set"
        assert len(set(broker.client_ids)) == 2, "ids must be distinct per order"

    @patch("firm.live.engine.build_orchestrator")
    def test_concurrent_cycle_is_skipped(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        mock_orch = MagicMock()
        mock_orch.step.return_value = (_make_orders(), _make_blackboard())
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config, approval_mode="full_auto")
        engine.start()

        # Simulate a cycle already in progress by holding the lock.
        engine._cycle_lock.acquire()
        try:
            result = engine.run_cycle()
        finally:
            engine._cycle_lock.release()

        assert result.skipped is True
        assert result.orders_submitted == 0

    @patch("firm.live.engine.build_orchestrator")
    def test_manual_orders_grouped_by_strategy(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        orders = _make_orders()  # momentum + trend
        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, _make_blackboard())
        mock_build.return_value = mock_orch

        # semi_auto with nothing auto-approved -> both queued, but separately.
        engine = self._make_engine(
            broker, feed, queue, config,
            approval_mode="semi_auto", auto_approve_strategies=[],
        )
        engine.start()
        result = engine.run_cycle()

        assert result.orders_queued == 2
        assert len(result.approval_ids) == 2  # one approval per strategy
        strategies = {a.strategy for a in queue.get_pending()}
        assert strategies == {"momentum", "trend"}

    @patch("firm.live.engine.build_orchestrator")
    def test_broker_failures_are_surfaced(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components

        class FlakyBroker(MockBroker):
            def submit_order(self, order):
                if order.symbol == "MSFT":
                    raise BrokerError("insufficient buying power")
                return super().submit_order(order)

        broker = FlakyBroker()
        mock_orch = MagicMock()
        mock_orch.step.return_value = (_make_orders(), _make_blackboard())
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config, approval_mode="full_auto")
        engine.start()
        result = engine.run_cycle()

        assert result.orders_submitted == 1  # AAPL went through
        assert result.orders_failed == 1     # MSFT failed and is visible
        assert result.failed_orders[0]["symbol"] == "MSFT"

    def test_reconcile_respects_in_flight_orders(self):
        """An unsettled open order must not cause internal holdings to be
        wiped (which would re-submit the order next cycle)."""

        class InFlightBroker(MockBroker):
            def get_positions(self):
                return []  # broker hasn't booked the fill yet

            def get_open_orders(self):
                return [OrderStatus(
                    order_id="x", symbol="AAPL", side="buy",
                    quantity=10, filled_quantity=0.0, status="pending",
                )]

        broker = InFlightBroker()
        broker.connect()
        portfolio = PortfolioState(initial_capital=100_000)
        portfolio.holdings = {"AAPL": 10}  # we already booked the intended buy

        discrepancies = sync_portfolio_from_broker(
            broker, portfolio, prices={"AAPL": 150.0}
        )
        # No spurious position_mismatch, and holdings preserved.
        assert portfolio.holdings.get("AAPL") == 10
        assert not any(d["type"] == "position_mismatch" for d in discrepancies)


# ---------------------------------------------------------------------------
# CycleResult tests
# ---------------------------------------------------------------------------

class TestCycleResult:
    def test_defaults(self):
        r = CycleResult(cycle_id=1, timestamp=utcnow())
        assert r.orders_generated == 0
        assert r.error is None
        assert r.approval_ids == []


# ---------------------------------------------------------------------------
# News-guard blackout gate
# ---------------------------------------------------------------------------

class TestNewsGuardGate:
    @patch("firm.live.engine.build_orchestrator")
    def _engine(self, mock_build, tmp_path, news_guard):
        mock_build.return_value = MagicMock()
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["SPY"])
        queue = ApprovalQueue(broker=broker)
        config = {
            "initial_capital": 100_000,
            "memory_log_path": str(tmp_path / "decisions.jsonl"),
            "news_guard": news_guard,
        }
        return LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )

    def test_blocks_orders_in_event_window(self, tmp_path):
        engine = self._engine(
            tmp_path=tmp_path,
            news_guard={"enabled": True, "offline": True},
        )
        result = CycleResult(cycle_id=1, timestamp=utcnow())
        # 18:05 UTC on FOMC day is inside the bundled-CSV blackout window.
        at = datetime(2026, 7, 29, 18, 5)
        allowed = engine._apply_news_guard(
            [{"symbol": "SPY", "side": "buy", "quantity": 1}], at, result
        )
        assert allowed == []
        assert any(a["kind"] == "news_guard_blackout" for a in result.alerts)

    def test_allows_orders_in_quiet_window(self, tmp_path):
        engine = self._engine(
            tmp_path=tmp_path,
            news_guard={"enabled": True, "offline": True},
        )
        result = CycleResult(cycle_id=1, timestamp=utcnow())
        at = datetime(2026, 7, 20, 12, 0)  # no event nearby
        orders = [{"symbol": "SPY", "side": "buy", "quantity": 1}]
        assert engine._apply_news_guard(orders, at, result) == orders

    def test_disabled_is_noop(self, tmp_path):
        engine = self._engine(
            tmp_path=tmp_path,
            news_guard={"enabled": False},
        )
        result = CycleResult(cycle_id=1, timestamp=utcnow())
        at = datetime(2026, 7, 29, 18, 5)
        orders = [{"symbol": "SPY", "side": "buy", "quantity": 1}]
        assert engine._apply_news_guard(orders, at, result) == orders

    def test_calendar_totally_unavailable_fails_closed(self, tmp_path):
        """If the calendar can't be loaded at all (live fetch AND the
        bundled CSV both failed), every order must be held — approving
        them blind would defeat the entire point of the gate."""
        engine = self._engine(
            tmp_path=tmp_path,
            news_guard={"enabled": True, "offline": False},
        )
        result = CycleResult(cycle_id=1, timestamp=utcnow())
        at = datetime(2026, 7, 29, 18, 5)
        orders = [
            {"symbol": "SPY", "side": "buy", "quantity": 1},
            {"symbol": "QQQ", "side": "sell", "quantity": 2},
        ]
        with patch("firm.live.news_guard.load_events", side_effect=RuntimeError("disk full")):
            allowed = engine._apply_news_guard(orders, at, result)
        assert allowed == []
        alerts = [a for a in result.alerts if a["kind"] == "news_guard_calendar_unavailable"]
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"
        assert "disk full" in alerts[0]["message"]

    def test_live_fetch_failure_falling_back_to_csv_raises_stale_alert(self, tmp_path):
        from firm.live import news_guard as ng

        engine = self._engine(
            tmp_path=tmp_path,
            news_guard={"enabled": True, "offline": False},
        )
        result = CycleResult(cycle_id=1, timestamp=utcnow())
        at = datetime(2026, 7, 20, 12, 0)  # quiet window in the bundled CSV
        orders = [{"symbol": "SPY", "side": "buy", "quantity": 1}]
        # offline=False but load_events still lands on the bundled CSV, as it
        # genuinely would after a real live-fetch failure (real CSV events,
        # not a mock, so the quiet-window decision below is meaningful —
        # only the source label is what load_events would already produce).
        csv_events = ng.load_from_csv()
        with patch("firm.live.news_guard.load_events", return_value=(csv_events, "bundled-csv")):
            allowed = engine._apply_news_guard(orders, at, result)
        assert allowed == orders  # quiet window — not held
        stale_alerts = [a for a in result.alerts if a["kind"] == "news_guard_stale_calendar"]
        assert len(stale_alerts) == 1
        assert stale_alerts[0]["severity"] == "warning"

    def test_deliberate_offline_mode_does_not_raise_stale_alert(self, tmp_path):
        """offline=True is an intentional configuration, not a degradation —
        alerting every cycle for it would just be noise."""
        engine = self._engine(
            tmp_path=tmp_path,
            news_guard={"enabled": True, "offline": True},
        )
        result = CycleResult(cycle_id=1, timestamp=utcnow())
        at = datetime(2026, 7, 20, 12, 0)
        orders = [{"symbol": "SPY", "side": "buy", "quantity": 1}]
        engine._apply_news_guard(orders, at, result)
        assert not any(a["kind"] == "news_guard_stale_calendar" for a in result.alerts)


# ---------------------------------------------------------------------------
# Execution-safety live lock in _execute_orders
# ---------------------------------------------------------------------------

class TestExecutionSafetyGate:
    @patch("firm.live.engine.build_orchestrator")
    def _engine(self, mock_build, tmp_path, broker_type):
        mock_build.return_value = MagicMock()
        broker = MockBroker()
        broker.connect()
        feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
        queue = ApprovalQueue(broker=broker)
        config = {
            "initial_capital": 100_000,
            "memory_log_path": str(tmp_path / "decisions.jsonl"),
        }
        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine._broker_type = broker_type
        return engine, broker

    def test_live_broker_blocked_without_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FIRM_ALLOW_TRADING", raising=False)
        monkeypatch.setenv("FIRM_EXECUTION_AUDIT", str(tmp_path / "audit.jsonl"))
        engine, broker = self._engine(tmp_path=tmp_path, broker_type="ibkr_live")
        statuses, failed = engine._execute_orders(_make_orders())
        assert statuses == []
        assert len(failed) == 2
        assert all("FIRM_ALLOW_TRADING" in f["error"] for f in failed)
        assert any(a["kind"] == "live_trading_locked" for a in engine.alerts)

    def test_paper_broker_submits(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FIRM_ALLOW_TRADING", raising=False)
        monkeypatch.setenv("FIRM_EXECUTION_AUDIT", str(tmp_path / "audit.jsonl"))
        engine, broker = self._engine(tmp_path=tmp_path, broker_type="ibkr_paper")
        statuses, failed = engine._execute_orders(_make_orders())
        assert len(statuses) == 2
        assert failed == []

    def test_live_broker_submits_with_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FIRM_ALLOW_TRADING", "1")
        monkeypatch.setenv("FIRM_EXECUTION_AUDIT", str(tmp_path / "audit.jsonl"))
        engine, broker = self._engine(tmp_path=tmp_path, broker_type="alpaca_live")
        statuses, failed = engine._execute_orders(_make_orders())
        assert len(statuses) == 2
        assert failed == []


# ---------------------------------------------------------------------------
# guard_order/RiskProfile hard-cap gate in _execute_orders
# ---------------------------------------------------------------------------

class TestOrderRiskCapGate:
    @patch("firm.live.engine.build_orchestrator")
    def _engine(self, mock_build, tmp_path, universe=("AAPL", "MSFT"), max_position_pct=None):
        mock_build.return_value = MagicMock()
        broker = MockBroker()
        broker.connect()
        feed = LiveDataFeed(providers={}, universe=list(universe))
        queue = ApprovalQueue(broker=broker)
        config = {
            "initial_capital": 100_000,
            "memory_log_path": str(tmp_path / "decisions.jsonl"),
        }
        if max_position_pct is not None:
            config["max_position_pct"] = max_position_pct
        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )
        engine._broker_type = "ibkr_paper"
        return engine, broker

    def test_no_config_cap_is_a_noop(self, tmp_path, monkeypatch):
        """Default (unconfigured) max_position_pct must not block orders
        that every other existing test already exercises without a cap."""
        monkeypatch.setenv("FIRM_EXECUTION_AUDIT", str(tmp_path / "audit.jsonl"))
        engine, broker = self._engine(tmp_path=tmp_path)
        statuses, failed = engine._execute_orders(_make_orders())
        assert len(statuses) == 2
        assert failed == []

    def test_order_for_symbol_outside_universe_is_blocked(self, tmp_path, monkeypatch):
        """Defense-in-depth: an order for a symbol the engine wasn't even
        configured to trade must never reach the broker, regardless of how
        it got produced upstream."""
        monkeypatch.setenv("FIRM_EXECUTION_AUDIT", str(tmp_path / "audit.jsonl"))
        engine, broker = self._engine(tmp_path=tmp_path, universe=["AAPL"])
        orders = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0, "strategy": "momentum"},
            {"symbol": "TSLA", "side": "buy", "quantity": 5, "price": 300.0, "strategy": "momentum"},
        ]
        statuses, failed = engine._execute_orders(orders)
        assert len(statuses) == 1
        assert statuses[0].symbol == "AAPL"
        assert len(failed) == 1
        assert failed[0]["symbol"] == "TSLA"
        assert "allowlist" in failed[0]["error"]
        assert any(a["kind"] == "order_risk_cap_blocked" for a in engine.alerts)

    def test_order_exceeding_notional_cap_is_blocked(self, tmp_path, monkeypatch):
        """A per-trade cap derived from RiskAgent's own max_position_pct
        (doubled for a legitimate full-position flip — see engine.py) must
        catch an order far outside anything the portfolio construction
        pipeline would ever legitimately produce."""
        monkeypatch.setenv("FIRM_EXECUTION_AUDIT", str(tmp_path / "audit.jsonl"))
        # cap = 2 * 0.01 * 100_000 = 2_000; this order's notional is 50_000.
        engine, broker = self._engine(tmp_path=tmp_path, max_position_pct=0.01)
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 100, "price": 500.0, "strategy": "momentum"}]
        statuses, failed = engine._execute_orders(orders)
        assert statuses == []
        assert len(failed) == 1
        assert "notional" in failed[0]["error"].lower()
        assert any(a["kind"] == "order_risk_cap_blocked" for a in engine.alerts)

    def test_order_within_notional_cap_still_submits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FIRM_EXECUTION_AUDIT", str(tmp_path / "audit.jsonl"))
        # cap = 2 * 0.5 * 100_000 = 100_000; this order's notional is 1_500.
        engine, broker = self._engine(tmp_path=tmp_path, max_position_pct=0.5)
        statuses, failed = engine._execute_orders(_make_orders())
        assert len(statuses) == 2
        assert failed == []

    def test_missing_stop_does_not_block_orders(self, tmp_path, monkeypatch):
        """This engine rebalances to target weights — orders never carry a
        protective-stop field — so the RiskProfile gate must run with
        require_stop=False, not the CLI/discretionary skill's stricter
        default (which would block every single order)."""
        monkeypatch.setenv("FIRM_EXECUTION_AUDIT", str(tmp_path / "audit.jsonl"))
        engine, broker = self._engine(tmp_path=tmp_path)
        orders = _make_orders()
        assert all("stop" not in o for o in orders)
        statuses, failed = engine._execute_orders(orders)
        assert len(statuses) == 2
        assert failed == []

    def test_blocked_order_is_audited(self, tmp_path, monkeypatch):
        import json

        audit_path = tmp_path / "audit.jsonl"
        monkeypatch.setenv("FIRM_EXECUTION_AUDIT", str(audit_path))
        engine, broker = self._engine(tmp_path=tmp_path, universe=["AAPL"])
        orders = [{"symbol": "TSLA", "side": "buy", "quantity": 5, "price": 300.0, "strategy": "momentum"}]
        engine._execute_orders(orders)
        records = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert any(r.get("decision") == "blocked" and r.get("symbol") == "TSLA" for r in records)


class TestBrokerReconnect:
    """Coverage for mid-session broker disconnect/reconnect handling — see
    docs/PROJECT_CONTEXT.md 'Broker & host failover'."""

    @pytest.fixture()
    def engine_components(self, tmp_path):
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "memory_log_path": str(tmp_path / "decisions.jsonl")}
        return broker, feed, queue, config

    def _make_engine(self, broker, feed, queue, config, **kwargs):
        return LiveTradingEngine(
            config=config, broker=broker, data_feed=feed,
            approval_queue=queue, **kwargs,
        )

    @patch("firm.live.engine.build_orchestrator")
    def test_reconnect_is_attempted_and_succeeds_inline(self, mock_build, engine_components):
        """A broker whose reconnect() call succeeds must not escalate to the
        sustained-disconnect alert — a single dropped socket should just
        self-heal within the same cycle's error handling."""
        broker, feed, queue, config = engine_components

        class DropsOnceBroker(MockBroker):
            def __init__(self):
                super().__init__()
                self.reconnect_calls = 0

            def get_current_prices(self, symbols):
                raise BrokerError("connection reset by peer")

            def reconnect(self):
                self.reconnect_calls += 1
                super().reconnect()

        broker = DropsOnceBroker()
        mock_build.return_value = MagicMock()

        engine = self._make_engine(broker, feed, queue, config)
        engine.start()
        result = engine.run_cycle()

        assert broker.reconnect_calls == 1
        assert any(a["kind"] == "broker_unavailable" for a in result.alerts)
        assert not any(a["kind"] == "broker_disconnected_sustained" for a in result.alerts)
        # reconnected=True must be recorded on the alert context.
        alert = next(a for a in result.alerts if a["kind"] == "broker_unavailable")
        assert alert["reconnected"] is True
        assert alert["consecutive_failures"] == 1
        # A self-healed reconnect is routine (IB Gateway's own daily restart
        # is a normal cause) — must not page a human as "critical".
        assert alert["severity"] == "warning"

    @patch("firm.live.engine.build_orchestrator")
    def test_sustained_disconnect_escalates_past_threshold(self, mock_build, engine_components):
        """When reconnect() itself keeps failing (e.g. IB Gateway is
        actually down, not just a dropped socket), consecutive cycles must
        escalate to a distinct, louder alert past the threshold — this is
        the "needs a human" signal the runbook tells operators to watch
        for."""
        broker, feed, queue, config = engine_components
        config = {**config, "broker_disconnect_alert_threshold": 2}

        class AlwaysDownBroker(MockBroker):
            def get_current_prices(self, symbols):
                raise BrokerError("no route to host")

            def reconnect(self):
                raise BrokerError("IB Gateway not reachable")

        broker = AlwaysDownBroker()
        mock_build.return_value = MagicMock()

        engine = self._make_engine(broker, feed, queue, config)
        engine.start()

        result1 = engine.run_cycle()
        assert any(a["kind"] == "broker_unavailable" for a in result1.alerts)
        assert not any(a["kind"] == "broker_disconnected_sustained" for a in result1.alerts)
        # Not yet past the threshold — "building toward an incident", not
        # the incident itself — must not page a human as "critical" yet.
        first_alert = next(a for a in result1.alerts if a["kind"] == "broker_unavailable")
        assert first_alert["severity"] == "warning"

        result2 = engine.run_cycle()
        assert any(a["kind"] == "broker_disconnected_sustained" for a in result2.alerts)
        sustained = next(a for a in result2.alerts if a["kind"] == "broker_disconnected_sustained")
        assert sustained["severity"] == "critical"
        assert sustained["consecutive_failures"] == 2
        assert sustained["reconnected"] is False

    @patch("firm.live.engine.build_orchestrator")
    def test_recovery_emits_reconnected_alert_and_resets_counter(self, mock_build, engine_components):
        """Once a cycle's broker calls actually succeed again, the engine
        must announce recovery and reset the failure counter — otherwise a
        transient blip would falsely count toward a later, unrelated
        outage's sustained-disconnect threshold."""
        broker, feed, queue, config = engine_components
        config = {**config, "broker_disconnect_alert_threshold": 5}

        state = {"down": True}

        class FlakyThenHealthyBroker(MockBroker):
            def get_current_prices(self, symbols):
                if state["down"]:
                    raise BrokerError("temporarily unavailable")
                return super().get_current_prices(symbols)

            def reconnect(self):
                raise BrokerError("still down")

        broker = FlakyThenHealthyBroker()
        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], _make_blackboard())
        mock_build.return_value = mock_orch

        engine = self._make_engine(broker, feed, queue, config)
        engine.start()

        down_result = engine.run_cycle()
        assert down_result.error is not None
        assert engine._consecutive_broker_failures == 1

        state["down"] = False
        up_result = engine.run_cycle()
        assert up_result.error is None
        assert engine._consecutive_broker_failures == 0
        assert any(a["kind"] == "broker_reconnected" for a in up_result.alerts)

    @patch("firm.live.engine.build_orchestrator")
    def test_default_broker_reconnect_calls_disconnect_then_connect(self, mock_build, engine_components):
        """The Broker ABC's default reconnect() (used by IBKRBroker/
        AlpacaBroker, which don't override it) must tear down and
        re-establish the connection rather than being a no-op."""
        broker, feed, queue, config = engine_components
        calls: list[str] = []
        broker.disconnect = lambda: calls.append("disconnect")  # type: ignore[method-assign]
        broker.connect = lambda: calls.append("connect")  # type: ignore[method-assign]

        broker.reconnect()

        assert calls == ["disconnect", "connect"]

    def test_default_broker_reconnect_survives_disconnect_failure(self, engine_components):
        """A broker whose connection is already dead may raise from
        disconnect() itself (e.g. the socket is gone) — reconnect() must
        still proceed to connect() rather than propagating that error."""
        broker, _feed, _queue, _config = engine_components
        broker.disconnect = MagicMock(side_effect=RuntimeError("socket already closed"))  # type: ignore[method-assign]
        connected = {"value": False}

        def _connect():
            connected["value"] = True

        broker.connect = _connect  # type: ignore[method-assign]

        broker.reconnect()

        assert connected["value"] is True


# ---------------------------------------------------------------------------
# Strategy circuit breaker config update (rebuild-on-change knob)
# ---------------------------------------------------------------------------

class TestUpdateStrategyCircuitBreaker:
    @patch("firm.live.engine.build_orchestrator")
    def _engine(self, mock_build, tmp_path):
        mock_build.return_value = MagicMock()
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {
            "initial_capital": 100_000,
            "memory_log_path": str(tmp_path / "decisions.jsonl"),
        }
        return LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )

    def test_updates_config_and_rebuilds_orchestrator(self, tmp_path):
        engine = self._engine(tmp_path=tmp_path)
        original_orchestrator = engine._orchestrator

        with patch("firm.live.engine.build_orchestrator") as mock_build:
            rebuilt = MagicMock()
            mock_build.return_value = rebuilt
            engine.update_strategy_circuit_breaker({"enabled": True, "trigger_sharpe": -0.3})

        assert engine._config["strategy_circuit_breaker"] == {
            "enabled": True, "trigger_sharpe": -0.3,
        }
        assert engine._orchestrator is rebuilt
        assert engine._orchestrator is not original_orchestrator

    def test_default_config_has_no_circuit_breaker_key(self, tmp_path):
        engine = self._engine(tmp_path=tmp_path)
        assert "strategy_circuit_breaker" not in engine._config


class TestUpdateStrategyRegimeWeights:
    @patch("firm.live.engine.build_orchestrator")
    def _engine(self, mock_build, tmp_path):
        mock_build.return_value = MagicMock()
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {
            "initial_capital": 100_000,
            "memory_log_path": str(tmp_path / "decisions.jsonl"),
        }
        return LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=queue,
        )

    def test_updates_config_and_rebuilds_orchestrator(self, tmp_path):
        engine = self._engine(tmp_path=tmp_path)
        original_orchestrator = engine._orchestrator

        with patch("firm.live.engine.build_orchestrator") as mock_build:
            rebuilt = MagicMock()
            mock_build.return_value = rebuilt
            engine.update_strategy_regime_weights({
                "enabled": True,
                "weights": {"Bull": {"momentum": 1.2}},
            })

        assert engine._config["strategy_regime_weights"] == {
            "enabled": True,
            "weights": {"Bull": {"momentum": 1.2}},
        }
        assert engine._orchestrator is rebuilt
        assert engine._orchestrator is not original_orchestrator

    def test_default_config_has_no_regime_weights_key(self, tmp_path):
        engine = self._engine(tmp_path=tmp_path)
        assert "strategy_regime_weights" not in engine._config
