"""Execution agent.

Takes an approved ``RiskDecision``, diffs its adjusted target weights
against current portfolio holdings, and produces an order list plus
turnover and cost estimates in an ``ExecutionReport``.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import Agent, AgentContext
from firm.contracts.models import ExecutionReport, RiskDecision

log = logging.getLogger(__name__)


class ExecutionAgent(Agent):
    """Translates approved target weights into executable orders."""

    role = "execution"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="execution", config=config)
        cfg = config or {}
        self.commission_pct: float = cfg.get("commission_pct", 0.001)
        self.slippage_pct: float = cfg.get("slippage_pct", 0.0005)

    def run(self, ctx: AgentContext, **inputs: Any) -> ExecutionReport:
        decision: RiskDecision = inputs["decision"]
        portfolio = inputs.get("portfolio")
        prices: dict[str, float] = inputs.get("prices", {})

        target_weights = decision.adjusted_targets

        if portfolio is not None and prices:
            current_weights = portfolio.get_weights(prices)
            nav = portfolio.cash + sum(
                shares * prices.get(sym, 0.0)
                for sym, shares in portfolio.holdings.items()
            )
        else:
            current_weights = {}
            nav = ctx.config.get("initial_capital", 10_000_000)

        orders: list[dict[str, Any]] = []
        turnover = 0.0

        all_symbols = sorted(set(target_weights) | set(current_weights))
        for sym in all_symbols:
            target_w = target_weights.get(sym, 0.0)
            current_w = current_weights.get(sym, 0.0)
            diff_w = target_w - current_w

            if abs(diff_w) < 1e-6:
                continue

            price = prices.get(sym, 0.0)
            if price <= 0:
                log.warning("No price for %s – skipping order", sym)
                continue

            dollar_amount = diff_w * nav
            quantity = abs(dollar_amount / price)
            side = "buy" if dollar_amount > 0 else "sell"

            orders.append(
                {
                    "symbol": sym,
                    "side": side,
                    "quantity": quantity,
                    "notional": abs(dollar_amount),
                    "price": price,
                    "strategy": "composite",
                }
            )
            turnover += abs(diff_w)

        total_notional = sum(o["notional"] for o in orders)
        costs = total_notional * (self.commission_pct + self.slippage_pct)

        return ExecutionReport(fills=orders, turnover=turnover, costs=costs)
