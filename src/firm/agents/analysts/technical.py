"""Technical analyst agent.

Receives strategies: momentum, trend, mean_reversion, volatility_breakout,
seasonality, gann.  Runs each; :func:`zscore_signals` is the **sole**
cross-sectional normalisation step for technical strategy outputs.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.analysts import zscore_signals
from firm.agents.base import Agent, AgentContext
from firm.contracts.models import SignalSet
from firm.strategies.base import BaseStrategy

log = logging.getLogger(__name__)


class TechnicalAnalyst(Agent):
    """Aggregates technical strategy signals into a single SignalSet."""

    role = "technical_analyst"

    def __init__(
        self,
        strategies: list[BaseStrategy] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name="technical_analyst", config=config)
        self.strategies = strategies or []
        self._ic_weights: dict[str, float] = (config or {}).get("ic_weights", {})

    def run(self, ctx: AgentContext, **inputs: Any) -> SignalSet:
        strategies = inputs.get("strategies", self.strategies)
        pit_view = ctx.pit_view
        if pit_view is None:
            return SignalSet(domain="technical", asof=ctx.now, signals=[])

        all_signals = []
        self._last_errors: list[dict] = []
        for strat in strategies:
            try:
                signals = strat.generate(pit_view)
                ic = self._ic_weights.get(strat.name, 1.0)
                if ic != 1.0:
                    from firm.contracts.models import Signal

                    signals = [
                        Signal(
                            symbol=s.symbol,
                            strategy=s.strategy,
                            score=s.score * ic,
                            confidence=s.confidence,
                            horizon=s.horizon,
                            asof=s.asof,
                            meta=s.meta,
                        )
                        for s in signals
                    ]
                all_signals.extend(signals)
            except Exception as exc:
                log.warning("Strategy %s failed in technical analyst", strat.name, exc_info=True)
                self._last_errors.append({"strategy": strat.name, "error": str(exc)})

        zscored = zscore_signals(all_signals)
        return SignalSet(domain="technical", asof=ctx.now, signals=zscored)
