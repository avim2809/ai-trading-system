"""LLM-enhanced technical analyst."""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.analysts.technical import TechnicalAnalyst
from firm.agents.base import AgentContext
from firm.agents.llm.base_llm_agent import LLMAgentMixin
from firm.contracts.models import Signal, SignalSet

log = logging.getLogger(__name__)


class LLMTechnicalAnalyst(TechnicalAnalyst, LLMAgentMixin):
    """Runs quant technical analysis then validates signals against research via LLM."""

    def __init__(
        self,
        strategies: list | None = None,
        config: dict[str, Any] | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        TechnicalAnalyst.__init__(self, strategies=strategies, config=config)
        LLMAgentMixin.__init__(self, llm_config=llm_config)

    def run(self, ctx: AgentContext, **inputs: Any) -> SignalSet:
        quant_result = TechnicalAnalyst.run(self, ctx, **inputs)

        enhanced_signals: list[Signal] = []
        for sig in quant_result.signals:
            rag_context = self._retrieve_context(
                sig.symbol,
                f"academic research on {sig.strategy} strategy patterns for {sig.symbol}",
                collections=["research", "system_docs"],
                asof=ctx.now,
            )
            if not rag_context:
                enhanced_signals.append(sig)
                continue
            context = self._compress(rag_context)
            prompt = (
                f"Symbol: {sig.symbol}\nStrategy: {sig.strategy}\n"
                f"Quant signal score: {sig.score:.2f}\n"
                f"Research context:\n{context}\n\n"
                "Validate this signal against academic evidence. Return JSON: "
                '{"score": float (-1 to 1), "confidence": float (0 to 1), "rationale": "..."}'
            )
            try:
                result = self._call_llm(
                    "You are a quantitative researcher validating technical signals.", prompt, json_mode=True,
                )
                enhanced_signals.append(Signal(
                    symbol=sig.symbol,
                    strategy=sig.strategy,
                    score=float(result.get("score", sig.score)),
                    confidence=float(result.get("confidence", sig.confidence)),
                    horizon=sig.horizon,
                    asof=sig.asof,
                    meta={**sig.meta, "llm_rationale": result.get("rationale", ""), "llm_enhanced": True},
                ))
            except Exception:
                log.debug("LLM enhancement failed for %s/%s", sig.symbol, sig.strategy, exc_info=True)
                enhanced_signals.append(sig)

        return SignalSet(domain=quant_result.domain, asof=quant_result.asof, signals=enhanced_signals)
