"""Tests for per-strategy capital sleeves (capital_allocation_mode: "sleeved").

Uses real (not mocked) BullResearcher/BearResearcher/DebateAgent/TraderAgent/
RiskAgent/ExecutionAgent instances -- these quant agents are lightweight,
deterministic, and already covered individually elsewhere (test_agents.py);
what's new here is the Orchestrator's sleeved-mode wiring: per-sleeve
PortfolioState compounding, capital-weight splitting, netting into one real
order set, and the restart-persistence round trip.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from firm.agents.base import Agent
from firm.agents.execution import ExecutionAgent
from firm.agents.orchestrator import Orchestrator
from firm.agents.research.bear import BearResearcher
from firm.agents.research.bull import BullResearcher
from firm.agents.research.debate import DebateAgent
from firm.agents.risk import RiskAgent
from firm.agents.trader import TraderAgent
from firm.contracts.models import Signal, SignalSet
from firm.live.approval import ApprovalQueue
from firm.live.data_feed import LiveDataFeed
from firm.live.engine import LiveTradingEngine
from firm.portfolio.state import PortfolioState
from tests.test_brokers import MockBroker


@pytest.fixture()
def engine_components(tmp_path):
    broker = MockBroker()
    feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
    queue = ApprovalQueue(broker=broker)
    config = {"initial_capital": 100_000, "memory_log_path": str(tmp_path / "decisions.jsonl")}
    return broker, feed, queue, config

NOW = datetime(2023, 6, 15, 16, 0)

# Permissive risk config -- these tests are about sleeve wiring, not cap
# edge cases (already covered by TestRiskManager in test_agents.py).
_PERMISSIVE_RISK_CFG = {
    "max_position_pct": 1.0,
    "max_gross_exposure": 10.0,
    "max_net_exposure": 10.0,
    "max_sector_pct": 1.0,
}
# Zero-cost execution config for clean NAV-compounding arithmetic.
_ZERO_COST_EXEC_CFG = {"commission_pct": 0.0, "slippage_pct": 0.0, "spread_pct": 0.0}


def _sig(symbol: str, strategy: str, score: float, confidence: float = 0.8) -> Signal:
    return Signal(
        symbol=symbol, strategy=strategy, score=score, confidence=confidence,
        horizon="5d", asof=NOW,
    )


def _analyst_with_signals(*signals: Signal) -> Agent:
    analyst = MagicMock(spec=Agent)
    analyst.name = "mock_analyst"
    analyst.run.return_value = SignalSet(domain="technical", asof=NOW, signals=list(signals))
    return analyst


def _pit_view():
    pv = MagicMock()
    pv.asof = NOW
    return pv


def _make_orchestrator(
    *,
    analysts,
    sleeve_traders: dict[str, Agent] | None = None,
    trader: Agent | None = None,
    execution: Agent | None = None,
    config: dict | None = None,
) -> Orchestrator:
    cfg = {
        "capital_allocation_mode": "sleeved" if sleeve_traders is not None else "blended",
        "initial_capital": 1_000_000.0,
        **(config or {}),
    }
    return Orchestrator(
        analysts=analysts,
        bull=BullResearcher(config={}),
        bear=BearResearcher(config={}),
        debate=DebateAgent(config={}),
        trader=trader or TraderAgent(config={"allocation_method": "conviction_weighted"}),
        risk=RiskAgent(config=_PERMISSIVE_RISK_CFG),
        execution=execution or ExecutionAgent(config=_ZERO_COST_EXEC_CFG),
        config=cfg,
        sleeve_traders=sleeve_traders,
    )


class TestBuildOrchestratorSleeving:
    """firm.runtime.build_orchestrator's sleeved-mode wiring."""

    def test_blended_by_default_no_sleeve_traders(self):
        from firm.runtime import build_orchestrator

        orch = build_orchestrator({"strategies": ["momentum", "trend"]})
        assert orch.capital_allocation_mode == "blended"
        assert orch.sleeve_traders == {}

    def test_sleeved_mode_builds_one_trader_per_strategy(self):
        from firm.agents.trader import TraderAgent
        from firm.runtime import build_orchestrator

        orch = build_orchestrator({
            "strategies": ["momentum", "trend"],
            "capital_allocation_mode": "sleeved",
        })
        assert orch.capital_allocation_mode == "sleeved"
        assert set(orch.sleeve_traders.keys()) == {"momentum", "trend"}
        for trader in orch.sleeve_traders.values():
            assert isinstance(trader, TraderAgent)
        # Independent instances -- not the same object shared across sleeves.
        assert orch.sleeve_traders["momentum"] is not orch.sleeve_traders["trend"]

    def test_sleeved_mode_still_raises_on_unsafe_agent_modes(self):
        from firm.runtime import build_orchestrator

        with pytest.raises(ValueError, match="debate"):
            build_orchestrator({
                "strategies": ["momentum"],
                "capital_allocation_mode": "sleeved",
                "agent_modes": {"debate": "llm_enhanced"},
            })


class TestSleeveCapitalWeights:
    def test_equal_split_default(self):
        orch = _make_orchestrator(
            analysts=[],
            sleeve_traders={"momentum": TraderAgent(), "trend": TraderAgent()},
        )
        weights = orch._sleeve_capital_weights()
        assert weights == {"momentum": 0.5, "trend": 0.5}

    def test_explicit_override_splits_remainder_equally(self):
        orch = _make_orchestrator(
            analysts=[],
            sleeve_traders={"momentum": TraderAgent(), "trend": TraderAgent(), "stat_arb": TraderAgent()},
            config={"strategy_capital_weights": {"momentum": 0.5}},
        )
        weights = orch._sleeve_capital_weights()
        assert weights["momentum"] == pytest.approx(0.5)
        assert weights["trend"] == pytest.approx(0.25)
        assert weights["stat_arb"] == pytest.approx(0.25)


class TestSleeveLlmCostSafety:
    def test_raises_when_bull_researcher_is_llm_enhanced(self):
        with pytest.raises(ValueError, match="bull_researcher"):
            _make_orchestrator(
                analysts=[],
                sleeve_traders={"momentum": TraderAgent()},
                config={"agent_modes": {"bull_researcher": "llm_enhanced"}},
            )

    def test_quant_mode_is_fine(self):
        _make_orchestrator(
            analysts=[],
            sleeve_traders={"momentum": TraderAgent()},
            config={"agent_modes": {"bull_researcher": "quant"}},
        )  # must not raise

    def test_blended_mode_ignores_agent_modes_entirely(self):
        _make_orchestrator(
            analysts=[],
            sleeve_traders=None,
            config={"agent_modes": {"bull_researcher": "llm_enhanced"}},
        )  # must not raise -- guard only applies when sleeving is on


class TestSleevedModeSingleStrategyMatchesBlended:
    """Migration-safety sanity check from the plan: sleeving one strategy
    alone must reproduce today's blended-mode orders exactly."""

    def test_single_sleeve_matches_blended_orders(self):
        signals = [_sig("AAPL", "momentum", 1.0), _sig("MSFT", "momentum", -0.5)]
        prices = {"AAPL": 150.0, "MSFT": 300.0}

        blended = _make_orchestrator(analysts=[_analyst_with_signals(*signals)])
        blended_orders, _ = blended.step(
            {"pit_view": _pit_view(), "portfolio": PortfolioState(initial_capital=1_000_000.0), "prices": prices},
        )

        sleeved = _make_orchestrator(
            analysts=[_analyst_with_signals(*signals)],
            sleeve_traders={"momentum": TraderAgent(config={"allocation_method": "conviction_weighted"})},
        )
        sleeved_orders, _ = sleeved.step(
            {"pit_view": _pit_view(), "portfolio": PortfolioState(initial_capital=1_000_000.0), "prices": prices},
        )

        def _by_symbol(orders):
            return {o["symbol"]: (o["side"], round(o["quantity"], 4)) for o in orders}

        assert len(blended_orders) == len(sleeved_orders) == 2
        assert _by_symbol(blended_orders) == _by_symbol(sleeved_orders)


class TestSleevedModeIndependentCompounding:
    def test_sleeve_portfolio_updates_from_its_own_fills(self):
        signals = [_sig("AAPL", "momentum", 1.0)]
        orch = _make_orchestrator(
            analysts=[_analyst_with_signals(*signals)],
            sleeve_traders={"momentum": TraderAgent(config={"allocation_method": "conviction_weighted"})},
        )
        real_portfolio = PortfolioState(initial_capital=1_000_000.0)
        orders, _ = orch.step(
            {"pit_view": _pit_view(), "portfolio": real_portfolio, "prices": {"AAPL": 100.0}},
        )
        assert len(orders) == 1

        sleeve = orch._sleeve_portfolios["momentum"]
        # Momentum is the only sleeve -> got 100% of initial capital, fully
        # long AAPL (conviction-weighted single-name = weight 1.0).
        assert sleeve.holdings.get("AAPL") == pytest.approx(10_000.0)  # $1M / $100
        assert sleeve.cash == pytest.approx(0.0, abs=1.0)

    def test_two_sleeves_split_capital_and_compound_independently(self):
        analyst = _analyst_with_signals(
            _sig("AAPL", "momentum", 1.0), _sig("MSFT", "trend", 1.0),
        )
        orch = _make_orchestrator(
            analysts=[analyst],
            sleeve_traders={
                "momentum": TraderAgent(config={"allocation_method": "conviction_weighted"}),
                "trend": TraderAgent(config={"allocation_method": "conviction_weighted"}),
            },
        )
        real_portfolio = PortfolioState(initial_capital=1_000_000.0)
        orch.step({"pit_view": _pit_view(), "portfolio": real_portfolio, "prices": {"AAPL": 100.0, "MSFT": 200.0}})

        momentum = orch._sleeve_portfolios["momentum"]
        trend = orch._sleeve_portfolios["trend"]
        # Equal 50/50 split of $1M initial capital, each fully in its own name.
        assert momentum.holdings.get("AAPL") == pytest.approx(5_000.0)  # $500k / $100
        assert trend.holdings.get("MSFT") == pytest.approx(2_500.0)  # $500k / $200
        assert momentum.nav == pytest.approx(500_000.0)
        assert trend.nav == pytest.approx(500_000.0)

    def test_no_signal_sleeve_gets_snapshot_not_a_trade(self):
        # Only momentum has a signal this cycle; trend sleeve exists but is silent.
        analyst = _analyst_with_signals(_sig("AAPL", "momentum", 1.0))
        orch = _make_orchestrator(
            analysts=[analyst],
            sleeve_traders={
                "momentum": TraderAgent(config={"allocation_method": "conviction_weighted"}),
                "trend": TraderAgent(config={"allocation_method": "conviction_weighted"}),
            },
        )
        real_portfolio = PortfolioState(initial_capital=1_000_000.0)
        orch.step({"pit_view": _pit_view(), "portfolio": real_portfolio, "prices": {"AAPL": 100.0}})

        trend = orch._sleeve_portfolios["trend"]
        assert trend.holdings == {}
        assert trend.cash == pytest.approx(500_000.0)  # untouched initial split
        assert len(trend.history) == 1  # still snapshotted for return continuity


class TestRealExecutionRebalanceBand:
    """Regression coverage for a real bug found via a blended-vs-sleeved A/B
    backtest (2024-Q1 cached data): splitting capital across ~18 sleeves,
    each further diversifying across several names, meant no single symbol's
    combined weight exceeded ~1% of total NAV -- so the blended book's
    validated 5% rebalance_band_pct silently filtered out every symbol on
    every day, zero real turnover for an entire quarter, even though every
    individual sleeve traded and compounded correctly. Fix: the final netted
    real-execution pass gets its own (smaller) band, independent of each
    sleeve's own internal execution band."""

    def test_real_execution_band_defaults_to_shared_band_over_sleeve_count(self):
        orch = _make_orchestrator(
            analysts=[],
            sleeve_traders={"momentum": TraderAgent(), "trend": TraderAgent()},
            execution=ExecutionAgent(config={**_ZERO_COST_EXEC_CFG, "rebalance_band_pct": 0.05}),
        )
        assert orch._real_execution.rebalance_band_pct == pytest.approx(0.025)

    def test_real_execution_band_explicit_override_respected(self):
        orch = _make_orchestrator(
            analysts=[],
            sleeve_traders={"momentum": TraderAgent(), "trend": TraderAgent()},
            execution=ExecutionAgent(config={**_ZERO_COST_EXEC_CFG, "rebalance_band_pct": 0.05}),
            config={"real_rebalance_band_pct": 0.001},
        )
        assert orch._real_execution.rebalance_band_pct == pytest.approx(0.001)

    def test_sleeve_internal_execution_band_is_unmodified(self):
        """Each sleeve's own virtual trading must keep using the original,
        already-validated band -- only the real netted pass changes."""
        orch = _make_orchestrator(
            analysts=[],
            sleeve_traders={"momentum": TraderAgent(), "trend": TraderAgent()},
            execution=ExecutionAgent(config={**_ZERO_COST_EXEC_CFG, "rebalance_band_pct": 0.05}),
        )
        assert orch.execution.rebalance_band_pct == pytest.approx(0.05)

    def test_real_trades_clear_a_band_the_shared_band_would_have_blocked(self):
        """The exact bug scenario: a small sleeve's combined weight (3% of
        total NAV, from an explicit small capital split) sits below the
        shared 5% band but above a real-execution override band -- so the
        real pass must trade even though reusing the shared band verbatim
        would have silently filtered it out for the entire run."""
        analyst = _analyst_with_signals(
            _sig("AAPL", "momentum", 1.0), _sig("MSFT", "trend", 1.0),
        )
        orch = _make_orchestrator(
            analysts=[analyst],
            sleeve_traders={
                "momentum": TraderAgent(config={"allocation_method": "conviction_weighted"}),
                "trend": TraderAgent(config={"allocation_method": "conviction_weighted"}),
            },
            execution=ExecutionAgent(config={**_ZERO_COST_EXEC_CFG, "rebalance_band_pct": 0.05}),
            config={
                "real_rebalance_band_pct": 0.01,  # would not block a 3% weight
                "strategy_capital_weights": {"momentum": 0.03, "trend": 0.03},
            },
        )
        real_portfolio = PortfolioState(initial_capital=1_000_000.0)
        orders, _ = orch.step({
            "pit_view": _pit_view(), "portfolio": real_portfolio,
            "prices": {"AAPL": 100.0, "MSFT": 100.0},
        })
        # Each sleeve: 3% of $1M = $30k fully in its one name -> 3% of real
        # NAV -- below the 5% shared band, above the 1% real-execution band.
        assert len(orders) == 2
        assert {o["symbol"] for o in orders} == {"AAPL", "MSFT"}


class TestSleevedModeNetting:
    def test_opposing_sleeves_net_to_a_smaller_real_order(self):
        analyst = _analyst_with_signals(
            _sig("AAPL", "momentum", 1.0), _sig("AAPL", "trend", -1.0),
        )
        orch = _make_orchestrator(
            analysts=[analyst],
            sleeve_traders={
                "momentum": TraderAgent(config={"allocation_method": "conviction_weighted"}),
                "trend": TraderAgent(config={"allocation_method": "conviction_weighted"}),
            },
            config={"strategy_capital_weights": {"momentum": 0.6, "trend": 0.4}},
        )
        real_portfolio = PortfolioState(initial_capital=1_000_000.0)
        orders, _ = orch.step(
            {"pit_view": _pit_view(), "portfolio": real_portfolio, "prices": {"AAPL": 100.0}},
        )
        # momentum wants +$600k of AAPL, trend wants -$400k -> net +$200k, one order.
        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["side"] == "buy"
        assert orders[0]["quantity"] == pytest.approx(2_000.0)  # $200k / $100

    def test_fully_offsetting_sleeves_produce_no_order(self):
        analyst = _analyst_with_signals(
            _sig("AAPL", "momentum", 1.0), _sig("AAPL", "trend", -1.0),
        )
        orch = _make_orchestrator(
            analysts=[analyst],
            sleeve_traders={
                "momentum": TraderAgent(config={"allocation_method": "conviction_weighted"}),
                "trend": TraderAgent(config={"allocation_method": "conviction_weighted"}),
            },
            config={"strategy_capital_weights": {"momentum": 0.5, "trend": 0.5}},
        )
        real_portfolio = PortfolioState(initial_capital=1_000_000.0)
        orders, _ = orch.step(
            {"pit_view": _pit_view(), "portfolio": real_portfolio, "prices": {"AAPL": 100.0}},
        )
        assert orders == []


class TestSleevePersistenceRoundTrip:
    def test_export_restore_round_trip(self):
        orch = _make_orchestrator(
            analysts=[],
            sleeve_traders={"momentum": TraderAgent(), "trend": TraderAgent()},
        )
        orch._sleeve_portfolios["momentum"] = orch._get_or_create_sleeve_portfolio("momentum", 0.5)
        orch._sleeve_portfolios["momentum"].cash = 400_000.0
        orch._sleeve_portfolios["momentum"].holdings = {"AAPL": 1_000.0}

        exported = orch.export_sleeve_portfolios()
        assert exported["momentum"] == {"cash": 400_000.0, "holdings": {"AAPL": 1_000.0}}

        restored = _make_orchestrator(
            analysts=[],
            sleeve_traders={"momentum": TraderAgent(), "trend": TraderAgent()},
        )
        restored.restore_sleeve_portfolios(exported)
        assert restored._sleeve_portfolios["momentum"].cash == 400_000.0
        assert restored._sleeve_portfolios["momentum"].holdings == {"AAPL": 1_000.0}

    def test_restart_does_not_reset_a_sleeve_to_initial_capital(self):
        analyst = _analyst_with_signals(_sig("AAPL", "momentum", 1.0))
        orch = _make_orchestrator(
            analysts=[analyst],
            sleeve_traders={"momentum": TraderAgent(config={"allocation_method": "conviction_weighted"})},
        )
        real_portfolio = PortfolioState(initial_capital=1_000_000.0)
        orch.step({"pit_view": _pit_view(), "portfolio": real_portfolio, "prices": {"AAPL": 100.0}})
        exported = orch.export_sleeve_portfolios()

        # Simulate a restart: a fresh Orchestrator restores from the export
        # before its first cycle, instead of lazily creating a fresh sleeve
        # at the full initial-capital split.
        restarted = _make_orchestrator(
            analysts=[analyst],
            sleeve_traders={"momentum": TraderAgent(config={"allocation_method": "conviction_weighted"})},
        )
        restarted.restore_sleeve_portfolios(exported)
        assert restarted._sleeve_portfolios["momentum"].holdings.get("AAPL") == pytest.approx(10_000.0)


class TestLiveAttributionEndpointSleeveMerge:
    """firm.api.routers.live.live_attribution's sleeved-mode merge."""

    @staticmethod
    def _fake_request(engine):
        request = MagicMock()
        request.app.state.live_engine = engine
        return request

    def test_blended_mode_returns_heuristic_attribution_unchanged(self):
        from firm.api.routers.live import live_attribution

        engine = MagicMock()
        engine._attribution.get_strategy_metrics.return_value = {
            "momentum": {"sharpe_ratio": 1.2},
        }
        engine._orchestrator.capital_allocation_mode = "blended"

        result = live_attribution(self._fake_request(engine))
        assert result == {"momentum": {"sharpe_ratio": 1.2}}
        engine._orchestrator.get_sleeve_metrics.assert_not_called()

    def test_sleeved_mode_prefers_exact_sleeve_metrics(self):
        from firm.api.routers.live import live_attribution

        engine = MagicMock()
        engine._attribution.get_strategy_metrics.return_value = {
            "momentum": {"sharpe_ratio": 1.2},  # stale heuristic value
            "danelfin_ai_score": {"sharpe_ratio": 0.4},  # no sleeve for this one
        }
        engine._orchestrator.capital_allocation_mode = "sleeved"
        engine._orchestrator.get_sleeve_metrics.return_value = {
            "momentum": {"sharpe_ratio": 2.5},  # exact value wins
        }

        result = live_attribution(self._fake_request(engine))
        assert result["momentum"]["sharpe_ratio"] == 2.5
        assert result["danelfin_ai_score"]["sharpe_ratio"] == 0.4

    def test_no_engine_returns_empty(self):
        from firm.api.routers.live import live_attribution

        request = MagicMock()
        request.app.state.live_engine = None
        assert live_attribution(request) == {}


class TestSeedSleevesFromAttribution:
    """Best-effort seeding when switching a running engine from "blended"
    to "sleeved" mid-history (docs/capital_sleeves_plan.md §5's cutover
    step) -- without this, every sleeve would start from zero positions
    while the real book still holds actual current positions."""

    @staticmethod
    def _attribution_with_holdings(holdings: dict[str, dict[str, float]]):
        from firm.portfolio.attribution import PerformanceAttribution

        attribution = PerformanceAttribution()
        for strategy, sym_shares in holdings.items():
            fills = [
                {"symbol": sym, "shares": shares, "price": 1.0, "strategy": strategy}
                for sym, shares in sym_shares.items()
            ]
            attribution.record_trades(fills, {sym: 1.0 for sym in sym_shares})
        return attribution

    def test_get_strategy_holdings_returns_attributed_shares(self):
        attribution = self._attribution_with_holdings({"momentum": {"AAPL": 100.0}})
        assert attribution.get_strategy_holdings("momentum") == {"AAPL": 100.0}
        assert attribution.get_strategy_holdings("trend") == {}

    def test_seeds_cash_and_holdings_to_match_target_capital(self):
        orch = _make_orchestrator(
            analysts=[],
            sleeve_traders={"momentum": TraderAgent(), "trend": TraderAgent()},
        )
        attribution = self._attribution_with_holdings({"momentum": {"AAPL": 100.0}})
        # Equal split: each sleeve's target = 50% of total_nav ($1M) = $500k.
        # momentum's seeded holdings (100 AAPL @ $100) are worth $10k, so its
        # seeded cash should be $500k - $10k = $490k; trend has no
        # attributed holdings, so its seeded cash is its full $500k share.
        summary = orch.seed_sleeve_portfolios_from_attribution(
            attribution, {"AAPL": 100.0}, total_nav=1_000_000.0,
        )

        momentum = orch._sleeve_portfolios["momentum"]
        assert momentum.holdings == {"AAPL": 100.0}
        assert momentum.cash == pytest.approx(490_000.0)
        assert momentum.nav == pytest.approx(500_000.0)

        trend = orch._sleeve_portfolios["trend"]
        assert trend.holdings == {}
        assert trend.cash == pytest.approx(500_000.0)

        assert summary["momentum"]["target_capital"] == pytest.approx(500_000.0)
        assert summary["momentum"]["seeded_cash"] == pytest.approx(490_000.0)
        assert summary["momentum"]["seeded_holdings"] == {"AAPL": 100.0}

    def test_over_attributed_sleeve_seeds_negative_cash_not_clipped(self):
        """A strategy attributed more notional than its own capital share
        (possible: blended mode never capped any strategy's contribution)
        seeds a real negative cash balance -- an honest reflection of the
        approximation, not silently clipped to zero."""
        orch = _make_orchestrator(
            analysts=[], sleeve_traders={"momentum": TraderAgent(), "trend": TraderAgent()},
        )
        attribution = self._attribution_with_holdings({"momentum": {"AAPL": 8_000.0}})
        orch.seed_sleeve_portfolios_from_attribution(
            attribution, {"AAPL": 100.0}, total_nav=1_000_000.0,
        )
        momentum = orch._sleeve_portfolios["momentum"]
        # 8,000 shares @ $100 = $800k >> momentum's $500k target share.
        assert momentum.cash == pytest.approx(500_000.0 - 800_000.0)
        assert momentum.nav == pytest.approx(500_000.0)  # still exactly its target


class TestEngineSeedSleevesFromAttribution:
    """LiveTradingEngine.seed_sleeves_from_attribution -- the deliberate,
    one-time operator action wired to POST /api/live/sleeves/seed."""

    def _sleeved_engine(self, broker, feed, queue, config):
        cfg = {**config, "capital_allocation_mode": "sleeved", "strategies": ["momentum", "trend"]}
        return LiveTradingEngine(config=cfg, broker=broker, data_feed=feed, approval_queue=queue)

    def test_refuses_when_not_sleeved(self, engine_components):
        broker, feed, queue, config = engine_components
        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        with pytest.raises(ValueError, match="not in sleeved mode"):
            engine.seed_sleeves_from_attribution()

    def test_refuses_when_sleeves_already_have_state(self, engine_components):
        broker, feed, queue, config = engine_components
        engine = self._sleeved_engine(broker, feed, queue, config)
        engine._orchestrator._get_or_create_sleeve_portfolio("momentum", 0.5)
        with pytest.raises(ValueError, match="already have state"):
            engine.seed_sleeves_from_attribution()

    def test_seeds_from_real_broker_positions_and_persists(self, engine_components, tmp_path):
        from firm.brokers.base import OrderRequest

        broker, feed, queue, config = engine_components
        broker.connect()
        broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=10))
        engine = self._sleeved_engine(broker, feed, queue, config)
        engine._attribution.record_trades(
            [{"symbol": "AAPL", "shares": 10, "price": 1.0, "strategy": "momentum"}],
            {"AAPL": 1.0},
        )

        summary = engine.seed_sleeves_from_attribution()

        assert "momentum" in summary
        momentum = engine._orchestrator._sleeve_portfolios["momentum"]
        assert momentum.holdings == {"AAPL": 10.0}


class TestSleeveMetrics:
    def test_get_sleeve_metrics_needs_at_least_two_snapshots(self):
        orch = _make_orchestrator(analysts=[], sleeve_traders={"momentum": TraderAgent()})
        portfolio = orch._get_or_create_sleeve_portfolio("momentum", 1.0)
        portfolio.record_snapshot(NOW, {})
        assert orch.get_sleeve_metrics() == {}

    def test_get_sleeve_metrics_computes_real_metrics_from_nav_history(self):
        orch = _make_orchestrator(analysts=[], sleeve_traders={"momentum": TraderAgent()})
        portfolio = orch._get_or_create_sleeve_portfolio("momentum", 1.0)
        portfolio.holdings = {"AAPL": 100.0}
        for price in (100.0, 105.0, 110.0):
            portfolio.record_snapshot(NOW, {"AAPL": price})

        metrics = orch.get_sleeve_metrics()
        assert "momentum" in metrics
        assert metrics["momentum"]["total_return"] > 0
        assert "sharpe_ratio" in metrics["momentum"]
