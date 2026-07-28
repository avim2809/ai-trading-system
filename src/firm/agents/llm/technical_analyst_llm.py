"""LLM-enhanced technical analyst."""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.analysts import zscore_signals
from firm.agents.analysts.technical import TechnicalAnalyst
from firm.agents.base import AgentContext
from firm.agents.llm.base_llm_agent import LLMAgentMixin
from firm.contracts.models import Signal, SignalSet
from firm.llm.schemas import AnalystEnhancementResponse, parse_llm_response

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
        enhance_keys = self._signal_keys_to_enhance(quant_result.signals)
        for sig in quant_result.signals:
            if (sig.symbol, sig.strategy) not in enhance_keys:
                enhanced_signals.append(sig)
                continue
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
                raw = self._call_llm(
                    "You are a quantitative researcher validating technical signals.", prompt, json_mode=True,
                )
                parsed = parse_llm_response(
                    AnalystEnhancementResponse, raw, context=f"{sig.symbol}/{sig.strategy}",
                )
                if parsed is None:
                    enhanced_signals.append(sig)
                    continue
                score, confidence = self._bounded_override(
                    sig.symbol, sig.strategy,
                    parsed.score, parsed.confidence,
                    fallback_score=sig.score, fallback_confidence=sig.confidence,
                )
                enhanced_signals.append(Signal(
                    symbol=sig.symbol,
                    strategy=sig.strategy,
                    score=score,
                    confidence=confidence,
                    horizon=sig.horizon,
                    asof=sig.asof,
                    meta={**sig.meta, "llm_rationale": parsed.rationale, "llm_enhanced": True},
                ))
            except Exception:
                log.warning("LLM enhancement failed for %s/%s", sig.symbol, sig.strategy, exc_info=True)
                enhanced_signals.append(sig)

        # zscore_signals is the sole cross-sectional normalisation step for
        # this domain (see TechnicalAnalyst docstring); LLM overrides above
        # replace some signals' scores with a differently-scaled value
        # (clamped to [-1, 1], not z-scored), so the whole group must be
        # re-normalised here or the LLM-enhanced and pass-through signals
        # would sit on two different, incomparable scales.
        return SignalSet(
            domain=quant_result.domain, asof=quant_result.asof,
            signals=zscore_signals(enhanced_signals),
        )
