"""Bull researcher agent.

Takes SignalSets from the blackboard and, for each symbol with net-positive
signal mass, builds a ``Thesis(side="bull", ...)`` aggregating supporting
evidence from multiple domains.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import Agent, AgentContext
from firm.contracts.models import Thesis

log = logging.getLogger(__name__)

_ZSCORE_CAP = 3.0  # z-score at which conviction saturates to 1.0


class BullResearcher(Agent):
    """Builds bullish theses from positive signals on the blackboard."""

    role = "bull_researcher"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="bull_researcher", config=config)

    def run(self, ctx: AgentContext, **inputs: Any) -> list[Thesis]:
        from firm.agents.blackboard import Blackboard

        blackboard: Blackboard = inputs["blackboard"]
        theses: list[Thesis] = []

        for symbol in sorted(blackboard.get_all_symbols()):
            signals = blackboard.get_signals_by_symbol(symbol)
            if not signals:
                continue

            # Conviction reflects the NET (signed) confidence-weighted mean of
            # ALL signals for the symbol, not just the positive subset.  This
            # is symmetric with the bear side and consistent with the debate
            # netting, so a lone loud signal can't dominate a quiet opposing one.
            total_conf = sum(s.confidence for s in signals)
            if total_conf > 0:
                net = sum(s.score * s.confidence for s in signals) / total_conf
            else:
                net = sum(s.score for s in signals) / len(signals)

            if net <= 0:  # not net-bullish – leave to the bear researcher
                continue

            conviction = min(1.0, net / _ZSCORE_CAP)

            positive = [s for s in signals if s.score > 0]
            supporting_strategies = sorted({s.strategy for s in positive})
            supporting_domains = sorted(
                {ss.domain for ss in blackboard.signal_sets for sig in ss.signals if sig.symbol == symbol and sig.score > 0}
            )

            rationale = (
                f"Bullish on {symbol}: {len(positive)} positive signal(s) "
                f"from {', '.join(supporting_domains) or 'unknown'} domain(s), "
                f"net z-score {net:.2f}"
            )

            theses.append(
                Thesis(
                    side="bull",
                    symbol=symbol,
                    conviction=conviction,
                    rationale=rationale,
                    supporting=supporting_strategies,
                )
            )

        return theses
