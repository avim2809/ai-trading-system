"""LLM-enhanced bear researcher."""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import AgentContext
from firm.agents.llm.base_llm_agent import LLMAgentMixin
from firm.agents.research.bear import BearResearcher
from firm.contracts.models import Thesis

log = logging.getLogger(__name__)


class LLMBearResearcher(BearResearcher, LLMAgentMixin):
    """Builds quant bear theses then enriches with LLM risk analysis."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        BearResearcher.__init__(self, config=config)
        LLMAgentMixin.__init__(self, llm_config=llm_config)

    def run(self, ctx: AgentContext, **inputs: Any) -> list[Thesis]:
        quant_theses = BearResearcher.run(self, ctx, **inputs)

        enhanced: list[Thesis] = []
        for thesis in quant_theses:
            rag_context = self._retrieve_context(
                thesis.symbol,
                f"bearish risk factors regulatory litigation for {thesis.symbol}",
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
                f"Risk data:\n{context}\n\n"
                "Write a bearish thesis citing specific risks. Return JSON: "
                '{"conviction": float (0 to 1), "rationale": "..."}'
            )
            try:
                result = self._call_llm(
                    "You are a risk-focused analyst building a bear case.",
                    prompt,
                    json_mode=True,
                )
                enhanced.append(Thesis(
                    side="bear",
                    symbol=thesis.symbol,
                    conviction=float(result.get("conviction", thesis.conviction)),
                    rationale=result.get("rationale", thesis.rationale),
                    supporting=list(thesis.supporting),
                ))
            except Exception:
                log.warning("LLM enhancement failed for bear %s", thesis.symbol, exc_info=True)
                enhanced.append(thesis)

        return enhanced
