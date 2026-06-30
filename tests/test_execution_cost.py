"""Pre-trade transaction-cost estimation in the ExecutionAgent (Tier A)."""

from __future__ import annotations

from datetime import datetime

from firm.agents.base import AgentContext
from firm.agents.execution import ExecutionAgent
from firm.contracts.models import RiskDecision

NOW = datetime(2023, 6, 1)


class TestPreTradeCost:
    def test_per_order_cost_breakdown(self):
        agent = ExecutionAgent({"commission_pct": 0.001, "slippage_pct": 0.0005})
        ctx = AgentContext(now=NOW, config={"initial_capital": 1_000_000})
        decision = RiskDecision(approved=True, adjusted_targets={"AAPL": 0.1})

        report = agent.run(ctx, decision=decision, prices={"AAPL": 100.0})

        assert len(report.fills) == 1
        order = report.fills[0]
        notional = order["notional"]  # 0.1 * 1_000_000 = 100_000
        assert notional == 100_000.0
        assert order["est_commission"] == notional * 0.001
        assert order["est_slippage"] == notional * 0.0005
        assert order["est_cost"] == order["est_commission"] + order["est_slippage"]

    def test_aggregate_costs_equals_sum_of_orders(self):
        agent = ExecutionAgent({"commission_pct": 0.001, "slippage_pct": 0.0005})
        ctx = AgentContext(now=NOW, config={"initial_capital": 1_000_000})
        decision = RiskDecision(approved=True, adjusted_targets={"AAPL": 0.1, "MSFT": 0.2})

        report = agent.run(
            ctx, decision=decision, prices={"AAPL": 100.0, "MSFT": 50.0}
        )

        assert report.costs == sum(o["est_cost"] for o in report.fills)
        # Matches the prior aggregate formulation: notional * (comm + slip).
        total_notional = sum(o["notional"] for o in report.fills)
        assert abs(report.costs - total_notional * (0.001 + 0.0005)) < 1e-9
