"""LLM-enhanced debate / synthesis agent."""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import AgentContext
from firm.agents.llm.base_llm_agent import LLMAgentMixin
from firm.agents.research.debate import DebateAgent
from firm.contracts.models import DebateResult

log = logging.getLogger(__name__)


class LLMDebateAgent(DebateAgent, LLMAgentMixin):
    """Runs quant debate then asks LLM to weigh arguments and adjust conviction."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        DebateAgent.__init__(self, config=config)
        LLMAgentMixin.__init__(self, llm_config=llm_config)

    def run(self, ctx: AgentContext, **inputs: Any) -> list[DebateResult]:
        quant_results = DebateAgent.run(self, ctx, **inputs)

        enhanced: list[DebateResult] = []
        for dr in quant_results:
            bull_text = dr.bull_thesis.rationale if dr.bull_thesis else "No bull thesis"
            bear_text = dr.bear_thesis.rationale if dr.bear_thesis else "No bear thesis"

            rag_context = self._retrieve_context(
                dr.symbol,
                f"investment debate analysis for {dr.symbol}",
                collections=["research", "system_docs"],
                asof=ctx.now,
            )
            context = self._compress(rag_context) if rag_context else ""
            prompt = (
                f"Symbol: {dr.symbol}\n"
                f"Bull thesis: {bull_text}\n"
                f"Bear thesis: {bear_text}\n"
                f"Quant net conviction: {dr.net_conviction:.2f}\n"
            )
            if context:
                prompt += f"Additional context:\n{context}\n\n"
            prompt += (
                "Weigh both sides. Return JSON: "
                '{"net_conviction": float (-1 to 1), "reasoning": "..."}'
            )
            try:
                result = self._call_llm(
                    "You are an investment committee synthesising opposing views.",
                    prompt,
                    json_mode=True,
                )
                enhanced.append(DebateResult(
                    symbol=dr.symbol,
                    net_conviction=float(result.get("net_conviction", dr.net_conviction)),
                    bull_thesis=dr.bull_thesis,
                    bear_thesis=dr.bear_thesis,
                ))
            except Exception:
                log.warning("LLM debate failed for %s", dr.symbol, exc_info=True)
                enhanced.append(dr)

        enhanced.sort(key=lambda r: r.net_conviction, reverse=True)
        return enhanced
