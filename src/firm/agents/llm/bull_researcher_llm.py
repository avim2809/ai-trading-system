"""LLM-enhanced bull researcher."""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import AgentContext
from firm.agents.llm.base_llm_agent import LLMAgentMixin
from firm.agents.research.bull import BullResearcher
from firm.contracts.models import Thesis

log = logging.getLogger(__name__)


class LLMBullResearcher(BullResearcher, LLMAgentMixin):
    """Builds quant bull theses then enriches with LLM-written rationale."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        BullResearcher.__init__(self, config=config)
        LLMAgentMixin.__init__(self, llm_config=llm_config)

    def run(self, ctx: AgentContext, **inputs: Any) -> list[Thesis]:
        quant_theses = BullResearcher.run(self, ctx, **inputs)

        enhanced: list[Thesis] = []
        for thesis in quant_theses:
            rag_context = self._retrieve_context(
                thesis.symbol,
                f"bullish investment thesis news filings for {thesis.symbol}",
                collections=["news", "sec_filings", "earnings"],
                asof=ctx.now,
            )
            if not rag_context:
                enhanced.append(thesis)
                continue
            context = self._compress(rag_context)
            prompt = (
                f"Symbol: {thesis.symbol}\n"
                f"Quant conviction: {thesis.conviction:.2f}\n"
                f"Quant rationale: {thesis.rationale}\n"
                f"Supporting data:\n{context}\n\n"
                "Write a richer bullish thesis with evidence. Return JSON: "
                '{"conviction": float (0 to 1), "rationale": "..."}'
            )
            try:
                result = self._call_llm(
                    "You are a buy-side equity research analyst building a bull case.",
                    prompt,
                    json_mode=True,
                )
                enhanced.append(Thesis(
                    side="bull",
                    symbol=thesis.symbol,
                    conviction=float(result.get("conviction", thesis.conviction)),
                    rationale=result.get("rationale", thesis.rationale),
                    supporting=list(thesis.supporting),
                ))
            except Exception:
                log.warning("LLM enhancement failed for bull %s", thesis.symbol, exc_info=True)
                enhanced.append(thesis)

        return enhanced
