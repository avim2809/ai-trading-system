"""Tests for the live trading engine, approval queue, and scheduler.

Uses the MockBroker from test_brokers and patches the orchestrator to
avoid real agent pipeline execution.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from firm.agents.blackboard import Blackboard
from firm.brokers.base import (
    Broker,
    BrokerError,
    BrokerPosition,
    OrderRequest,
    OrderStatus,
)
from firm.contracts.models import ExecutionReport, RiskDecision, TradeProposal
from firm.live.approval import ApprovalQueue, PendingApproval
from firm.live.data_feed import LiveDataFeed, LivePitViewAdapter
from firm.live.engine import CycleResult, LiveTradingEngine
from firm.live.portfolio_sync import sync_portfolio_from_broker
from firm.portfolio.state import PortfolioState

# Re-use MockBroker from test_brokers
from tests.test_brokers import MockBroker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blackboard(asof: datetime | None = None) -> Blackboard:
    bb = Blackboard(asof=asof or datetime.utcnow())
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
        approval.expires_at = datetime.utcnow() - timedelta(seconds=5)

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


class TestPendingApproval:
    def test_is_expired(self):
        a = PendingApproval(
            approval_id="test",
            created_at=datetime.utcnow() - timedelta(hours=2),
            expires_at=datetime.utcnow() - timedelta(hours=1),
            orders=[],
            blackboard_snapshot={},
        )
        assert a.is_expired() is True

    def test_not_expired(self):
        a = PendingApproval(
            approval_id="test",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
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

    def test_pit_view_protocol_methods(self):
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        view = feed.refresh()

        # These should return empty DataFrames when no providers supplied
        prices = view.prices()
        assert prices.empty

        funds = view.fundamentals()
        assert funds.empty

        sent = view.sentiment()
        assert sent.empty


# ---------------------------------------------------------------------------
# LiveTradingEngine tests
# ---------------------------------------------------------------------------

class TestLiveTradingEngine:
    @pytest.fixture()
    def engine_components(self):
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000}
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
        engine.start()
        assert engine.is_running
        assert broker.is_connected()

        engine.stop()
        assert not engine.is_running
        assert not broker.is_connected()

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
        mock_orch.step.return_value = ([], Blackboard(asof=datetime.utcnow()))
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
    def test_cycle_history_tracking(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components

        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], Blackboard(asof=datetime.utcnow()))
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


# ---------------------------------------------------------------------------
# CycleResult tests
# ---------------------------------------------------------------------------

class TestCycleResult:
    def test_defaults(self):
        r = CycleResult(cycle_id=1, timestamp=datetime.utcnow())
        assert r.orders_generated == 0
        assert r.error is None
        assert r.approval_ids == []
