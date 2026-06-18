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
        per_strategy: dict[str, dict[str, float]] = inputs.get("per_strategy", {})

        target_weights = decision.adjusted_targets
        symbol_strategy = self._dominant_strategy_by_symbol(per_strategy)

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
                    # Signed share count: the canonical field consumed by
                    # PortfolioState.update and PerformanceAttribution.
                    # ``quantity`` stays absolute for broker order requests.
                    "shares": quantity if side == "buy" else -quantity,
                    "quantity": quantity,
                    "notional": abs(dollar_amount),
                    "price": price,
                    "strategy": symbol_strategy.get(sym, "composite"),
                }
            )
            turnover += abs(diff_w)

        total_notional = sum(o["notional"] for o in orders)
        costs = total_notional * (self.commission_pct + self.slippage_pct)

        return ExecutionReport(fills=orders, turnover=turnover, costs=costs)

    @staticmethod
    def _dominant_strategy_by_symbol(
        per_strategy: dict[str, dict[str, float]],
    ) -> dict[str, str]:
        """Map each symbol to the strategy that contributed the most weight.

        Used to tag each order with a real originating strategy so the live
        engine's per-strategy approval routing is meaningful.  Symbols absent
        from the attribution fall back to ``"composite"`` at the call site.
        """
        best: dict[str, tuple[float, str]] = {}
        for strat, weights in (per_strategy or {}).items():
            for sym, weight in weights.items():
                contribution = abs(weight)
                if sym not in best or contribution > best[sym][0]:
                    best[sym] = (contribution, strat)
        return {sym: strat for sym, (_, strat) in best.items()}
