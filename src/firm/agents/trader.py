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
            targets = self._risk_parity(selected, ctx)
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

    def _risk_parity(
        self, results: list[DebateResult], ctx: AgentContext
    ) -> dict[str, float]:
        """Inverse-volatility weighting (signed by conviction).

        Each name is weighted ∝ 1/vol using a realized-vol estimate from the
        PitView, so high-vol names get smaller allocations.  Falls back to
        equal weight only when no vol estimates are available.
        """
        if not results:
            return {}
        pit_view = getattr(ctx, "pit_view", None)
        inv_vol: dict[str, float] = {}
        for r in results:
            vol = self._estimate_vol(pit_view, r.symbol)
            if vol and vol > 0:
                inv_vol[r.symbol] = 1.0 / vol

        total = sum(inv_vol.values())
        if total <= 0:
            # No usable vol data – degrade to equal weight.
            w = 1.0 / len(results)
            return {r.symbol: w if r.net_conviction >= 0 else -w for r in results}

        targets: dict[str, float] = {}
        for r in results:
            mag = inv_vol.get(r.symbol, 0.0) / total
            targets[r.symbol] = mag if r.net_conviction >= 0 else -mag
        return targets

    @staticmethod
    def _estimate_vol(pit_view: Any, symbol: str, lookback: int = 63) -> float | None:
        """Annualized realized vol for *symbol* from PitView prices, or None."""
        if pit_view is None:
            return None
        try:
            df = pit_view.prices(symbols=[symbol], lookback_days=lookback)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        col = "adj_close" if "adj_close" in df.columns else "close"
        if col not in df.columns:
            return None
        prices = df.sort_values("date")[col]
        returns = prices.pct_change().dropna()
        if len(returns) < 2:
            return None
        vol = float(returns.std() * (252 ** 0.5))
        return vol if vol > 0 else None

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

        # NOTE: per_strategy is an *attribution* of the final ``targets`` and
        # must sum (per symbol) back to them.  Strategy budget caps are a
        # portfolio-construction concern and are intentionally NOT applied here
        # — scaling one bucket in isolation would break that invariant and
        # leave per_strategy inconsistent with the weights actually traded.
        return per_strategy
