"""Bear researcher agent.

Takes SignalSets from the blackboard and, for each symbol with net-negative
signal mass, builds a ``Thesis(side="bear", ...)`` focusing on negative
signals and risk factors.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import Agent, AgentContext
from firm.contracts.models import Thesis

log = logging.getLogger(__name__)

_ZSCORE_CAP = 3.0


class BearResearcher(Agent):
    """Builds bearish theses from negative signals on the blackboard."""

    role = "bear_researcher"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="bear_researcher", config=config)

    def run(self, ctx: AgentContext, **inputs: Any) -> list[Thesis]:
        from firm.agents.blackboard import Blackboard
        from firm.agents.research._combine import net_scores_for_blackboard

        blackboard: Blackboard = inputs["blackboard"]
        theses: list[Thesis] = []

        # Symmetric with the bull side: net (signed) score per symbol, using the
        # confidence-weighted mean by default or the optimal inverse-covariance
        # combination when configured and return history is available.
        net_scores = net_scores_for_blackboard(blackboard, ctx, self.config)

        for symbol in sorted(blackboard.get_all_symbols()):
            signals = blackboard.get_signals_by_symbol(symbol)
            if not signals:
                continue

            net = net_scores.get(symbol, 0.0)

            if net >= 0:  # not net-bearish – leave to the bull researcher
                continue

            conviction = min(1.0, -net / _ZSCORE_CAP)

            negative = [s for s in signals if s.score < 0]
            supporting_strategies = sorted({s.strategy for s in negative})
            supporting_domains = sorted(
                {ss.domain for ss in blackboard.signal_sets for sig in ss.signals if sig.symbol == symbol and sig.score < 0}
            )

            rationale = (
                f"Bearish on {symbol}: {len(negative)} negative signal(s) "
                f"from {', '.join(supporting_domains) or 'unknown'} domain(s), "
                f"net z-score {net:.2f}"
            )

            theses.append(
                Thesis(
                    side="bear",
                    symbol=symbol,
                    conviction=conviction,
                    rationale=rationale,
                    supporting=supporting_strategies,
                )
            )

        return theses
