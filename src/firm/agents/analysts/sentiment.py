"""Sentiment analyst agent.

Receives strategies: sentiment, event_driven.  Runs each; analysts apply
the sole cross-sectional :func:`zscore_signals` pass before producing a
``SignalSet(domain="sentiment", ...)``.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.analysts import zscore_signals
from firm.agents.base import Agent, AgentContext
from firm.contracts.models import SignalSet
from firm.strategies.base import BaseStrategy

log = logging.getLogger(__name__)


class SentimentAnalyst(Agent):
    """Aggregates sentiment/event-driven strategy signals into a SignalSet."""

    role = "sentiment_analyst"

    def __init__(
        self,
        strategies: list[BaseStrategy] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name="sentiment_analyst", config=config)
        self.strategies = strategies or []

    def run(self, ctx: AgentContext, **inputs: Any) -> SignalSet:
        strategies = inputs.get("strategies", self.strategies)
        pit_view = ctx.pit_view
        if pit_view is None:
            return SignalSet(domain="sentiment", asof=ctx.now, signals=[])

        all_signals = []
        self._last_errors: list[dict] = []
        for strat in strategies:
            try:
                signals = strat.generate(pit_view)
                all_signals.extend(signals)
            except Exception as exc:
                log.warning("Strategy %s failed in sentiment analyst", strat.name, exc_info=True)
                self._last_errors.append({"strategy": strat.name, "error": str(exc)})

        zscored = zscore_signals(all_signals)
        return SignalSet(domain="sentiment", asof=ctx.now, signals=zscored)
