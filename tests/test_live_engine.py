"""Tests for the live trading engine, approval queue, and scheduler.

Uses the MockBroker from test_brokers and patches the orchestrator to
avoid real agent pipeline execution.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

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
    def test_drawdown_kill_switch_trips_and_halts(self, mock_build):
        # Broker cash (50k) is far below the 100k starting equity, so the
        # peak-to-trough drawdown (50%) breaches the 10% kill switch.
        broker = MockBroker(initial_cash=50_000)
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "kill_switch_drawdown": 0.1}

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
    def test_alert_callback_invoked(self, mock_build):
        received: list[dict] = []
        broker = MockBroker(initial_cash=50_000)
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {"initial_capital": 100_000, "kill_switch_drawdown": 0.1}

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
    def test_clear_cycle_history_wipes_history(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components

        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], Blackboard(asof=datetime.utcnow()))
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
        engine2._llm_service.chat.return_value = "Reflection text."
        engine2.start()
        engine2.run_cycle()

        entries = [json.loads(line) for line in memory_path.read_text().splitlines()]
        reflected = [e for e in entries if e["status"] == "reflected"]
        assert len(reflected) == 1
        assert reflected[0]["reflection"] == "Reflection text."
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
            approval_queue=queue, approval_mode="full_auto",
        )
        engine.start()

        result = engine.run_cycle()
        # 2 orders this cycle > max_daily_trades=1 -> forced to manual despite full_auto.
        assert result.orders_submitted == 0
        assert result.orders_queued == 2
        assert any(a["kind"] == "daily_limit_breach" for a in result.alerts)

    @patch("firm.live.engine.build_orchestrator")
    def test_daily_turnover_limit_forces_manual_approval(self, mock_build, engine_components):
        broker, feed, queue, config = engine_components
        orders = _make_orders()  # 10*150 + 5*300 = 3000 notional
        bb = _make_blackboard()
        mock_orch = MagicMock()
        mock_orch.step.return_value = (orders, bb)
        mock_build.return_value = mock_orch

        # NAV is 100_000 (initial_capital) -> turnover_frac ~= 0.03; cap it far below that.
        engine = LiveTradingEngine(
            config={**config, "max_daily_turnover": 0.01}, broker=broker, data_feed=feed,
            approval_queue=queue, approval_mode="full_auto",
        )
        engine.start()

        result = engine.run_cycle()
        assert result.orders_submitted == 0
        assert result.orders_queued == 2
        assert any(a["kind"] == "daily_limit_breach" for a in result.alerts)

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
        r = CycleResult(cycle_id=1, timestamp=datetime.utcnow())
        assert r.orders_generated == 0
        assert r.error is None
        assert r.approval_ids == []
