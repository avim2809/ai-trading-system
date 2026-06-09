"""Trader / portfolio manager agent.

Converts ``DebateResult`` conviction scores into target portfolio weights
via a configurable allocation method (conviction-weighted, risk-parity, or
equal-weight), respecting per-strategy capital budgets and a maximum
number of positions.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import Agent, AgentContext
from firm.contracts.models import DebateResult, TradeProposal

log = logging.getLogger(__name__)


class TraderAgent(Agent):
    """Produces a TradeProposal from debate-synthesised convictions."""

    role = "trader"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="trader", config=config)
        cfg = config or {}
        self.allocation_method: str = cfg.get("allocation_method", "conviction_weighted")
        self.max_positions: int = cfg.get("max_positions", 20)
        self.strategy_budgets: dict[str, float] = cfg.get("strategy_budgets", {})

    def run(self, ctx: AgentContext, **inputs: Any) -> TradeProposal:
        debate_results: list[DebateResult] = inputs.get("debate_results", [])
        blackboard = inputs.get("blackboard")

        ranked = sorted(debate_results, key=lambda r: abs(r.net_conviction), reverse=True)
        selected = [r for r in ranked if abs(r.net_conviction) > 1e-8][: self.max_positions]

        if self.allocation_method == "equal_weight":
            targets = self._equal_weight(selected)
        elif self.allocation_method == "risk_parity":
            targets = self._risk_parity(selected)
        else:
            targets = self._conviction_weighted(selected)

        per_strategy = self._attribute_to_strategies(targets, blackboard)

        return TradeProposal(
            asof=ctx.now,
            targets=targets,
            per_strategy=per_strategy,
            notes=(
                f"Method: {self.allocation_method}, "
                f"{len(selected)}/{len(debate_results)} positions selected"
            ),
        )

    # ------------------------------------------------------------------
    # allocation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _conviction_weighted(results: list[DebateResult]) -> dict[str, float]:
        total = sum(abs(r.net_conviction) for r in results)
        if total == 0:
            return {}
        return {r.symbol: r.net_conviction / total for r in results}

    @staticmethod
    def _equal_weight(results: list[DebateResult]) -> dict[str, float]:
        if not results:
            return {}
        w = 1.0 / len(results)
        return {r.symbol: w if r.net_conviction >= 0 else -w for r in results}

    @staticmethod
    def _risk_parity(results: list[DebateResult]) -> dict[str, float]:
        """Simplified risk-parity: equal weight with conviction sign.

        A full implementation would use per-asset vol estimates from
        PitView; fall back to equal-weight when those are unavailable.
        """
        if not results:
            return {}
        w = 1.0 / len(results)
        return {r.symbol: w if r.net_conviction >= 0 else -w for r in results}

    def _attribute_to_strategies(
        self,
        targets: dict[str, float],
        blackboard: Any,
    ) -> dict[str, dict[str, float]]:
        """Build a per-strategy weight attribution from blackboard signals."""
        if blackboard is None:
            return {}

        per_strategy: dict[str, dict[str, float]] = {}

        for sym, weight in targets.items():
            signals = blackboard.get_signals_by_symbol(sym)
            if not signals:
                per_strategy.setdefault("unattributed", {})[sym] = weight
                continue

            total_score = sum(abs(s.score) for s in signals)
            for sig in signals:
                frac = abs(sig.score) / total_score if total_score > 0 else 1.0 / len(signals)
                bucket = per_strategy.setdefault(sig.strategy, {})
                bucket[sym] = bucket.get(sym, 0.0) + weight * frac

        budget_total = sum(self.strategy_budgets.values()) if self.strategy_budgets else 0.0
        if budget_total > 0:
            for strat, budget_pct in self.strategy_budgets.items():
                if strat in per_strategy:
                    strat_gross = sum(abs(v) for v in per_strategy[strat].values())
                    if strat_gross > budget_pct:
                        scale = budget_pct / strat_gross
                        per_strategy[strat] = {s: w * scale for s, w in per_strategy[strat].items()}

        return per_strategy
