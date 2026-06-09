"""Debate / synthesis agent.

Takes bull and bear theses for each symbol, produces a ``DebateResult``
with ``net_conviction = bull.conviction - bear.conviction``, and ranks
symbols by net conviction.
"""

from __future__ import annotations

from typing import Any

from firm.agents.base import Agent, AgentContext
from firm.contracts.models import DebateResult, Thesis


class DebateAgent(Agent):
    """Synthesises opposing theses into ranked conviction scores."""

    role = "debate"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="debate", config=config)

    def run(self, ctx: AgentContext, **inputs: Any) -> list[DebateResult]:
        bull_theses: list[Thesis] = inputs.get("bull_theses", [])
        bear_theses: list[Thesis] = inputs.get("bear_theses", [])

        bull_by_sym = {t.symbol: t for t in bull_theses}
        bear_by_sym = {t.symbol: t for t in bear_theses}

        all_symbols = sorted(set(bull_by_sym) | set(bear_by_sym))
        results: list[DebateResult] = []

        for sym in all_symbols:
            bull = bull_by_sym.get(sym)
            bear = bear_by_sym.get(sym)
            bull_conv = bull.conviction if bull else 0.0
            bear_conv = bear.conviction if bear else 0.0
            net = bull_conv - bear_conv

            results.append(
                DebateResult(
                    symbol=sym,
                    net_conviction=net,
                    bull_thesis=bull,
                    bear_thesis=bear,
                )
            )

        results.sort(key=lambda r: r.net_conviction, reverse=True)
        return results
