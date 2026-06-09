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

        blackboard: Blackboard = inputs["blackboard"]
        theses: list[Thesis] = []

        for symbol in sorted(blackboard.get_all_symbols()):
            signals = blackboard.get_signals_by_symbol(symbol)
            negative = [s for s in signals if s.score < 0]
            if not negative:
                continue

            total_conf = sum(s.confidence for s in negative)
            if total_conf > 0:
                raw = sum(abs(s.score) * s.confidence for s in negative) / total_conf
            else:
                raw = sum(abs(s.score) for s in negative) / len(negative)

            conviction = min(1.0, max(0.0, raw / _ZSCORE_CAP))

            supporting_strategies = sorted({s.strategy for s in negative})
            supporting_domains = sorted(
                {ss.domain for ss in blackboard.signal_sets for sig in ss.signals if sig.symbol == symbol and sig.score < 0}
            )

            rationale = (
                f"Bearish on {symbol}: {len(negative)} negative signal(s) "
                f"from {', '.join(supporting_domains) or 'unknown'} domain(s), "
                f"avg |z-score| {raw:.2f}"
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
