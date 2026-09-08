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
            rag_query = (
                self._pattern_rag_query(sig)
                if sig.strategy == "pattern_recognition"
                else f"academic research on {sig.strategy} strategy patterns for {sig.symbol}"
            )
            rag_context = self._retrieve_context(
                sig.symbol,
                rag_query,
                collections=["research", "system_docs"],
                asof=ctx.now,
            )
            if not rag_context:
                enhanced_signals.append(sig)
                continue
            context = self._compress(rag_context)
            if sig.strategy == "pattern_recognition":
                prompt = self._pattern_prompt(sig, context)
            else:
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

    @staticmethod
    def _pattern_rag_query(sig: Signal) -> str:
        """RAG query for a ``pattern_recognition`` signal.

        The generic query (``f"academic research on {sig.strategy} strategy
        patterns for {sig.symbol}"``) degenerates to "academic research on
        pattern_recognition strategy patterns" for every symbol, since
        ``pattern_recognition`` is the *strategy* name, not the detected
        chart pattern. Querying on the actual geometric pattern (e.g.
        ``"cup_handle"``, carried in ``sig.meta["pattern"]`` — see
        ``firm.strategies.pattern_recognition``) retrieves research
        relevant to that specific setup instead.
        """
        pattern = sig.meta.get("pattern", sig.strategy)
        return f"{pattern} chart pattern reliability breakout confirmation"

    @staticmethod
    def _pattern_prompt(sig: Signal, context: str) -> str:
        """Pattern-specific validation prompt for a ``pattern_recognition`` signal.

        Surfaces the rich structured detail ``pattern_recognition.py``
        attaches to ``Signal.meta`` (pattern/direction/entry/stop/target/
        risk:reward/quality score/breakout volume) that the generic prompt
        above ignores entirely, and asks a pattern-specific question instead
        of a strategy-agnostic one: is the geometric setup textbook-valid,
        how much conviction should the measured-move target get, and what
        regime/volume caveats apply. Answered against the exact same
        ``AnalystEnhancementResponse`` contract as every other enhancement
        prompt (score/confidence/rationale, parsed via ``parse_llm_response``
        and clamped via ``_bounded_override``) — only the prompt content
        differs, never the response shape or override mechanics.
        """
        meta = sig.meta
        pattern = meta.get("pattern", sig.strategy)
        direction = meta.get("direction", "unknown")
        entry = meta.get("entry", 0.0)
        stop = meta.get("stop", 0.0)
        target = meta.get("target", 0.0)
        risk_reward = meta.get("risk_reward", 0.0)
        quality_score = meta.get("quality_score", 0.0)
        volume_ratio = meta.get("volume_ratio", 1.0)
        return (
            f"Symbol: {sig.symbol}\n"
            f"Strategy: {sig.strategy}\n"
            f"Detected pattern: {pattern} ({direction})\n"
            f"Entry: {entry:.2f}  Stop: {stop:.2f}  Target: {target:.2f}\n"
            f"Risk:Reward: {risk_reward:.2f}  Quant quality score: {quality_score:.1f}/100\n"
            f"Breakout volume vs. average: {volume_ratio:.2f}x\n"
            f"Research context:\n{context}\n\n"
            f"Validate this {pattern} pattern against academic/technical-"
            "analysis evidence. Specifically assess: (1) whether this "
            f"geometric setup meets textbook criteria for a {pattern} "
            "pattern, (2) how much conviction the measured-move target "
            "above should receive, and (3) any market-regime or volume "
            "caveats that should lower conviction. Return JSON: "
            '{"score": float (-1 to 1), "confidence": float (0 to 1), "rationale": "..."}'
        )
