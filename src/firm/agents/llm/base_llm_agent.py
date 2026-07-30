"""Shared mixin providing LLM + RAG utilities for enhanced agents."""

from __future__ import annotations

import logging
from typing import Any

from firm.contracts.models import Signal

log = logging.getLogger(__name__)


class LLMAgentMixin:
    """Mixin providing LLM + RAG utilities for enhanced agents.

    Lazy-initialises the LLM service, RAG retriever, and token compressor
    on first use so the system starts even when the ``llm`` extra is not
    installed.

    Cost controls (``config/llm.yaml`` → ``enhancement``) gate which
    symbols/signals receive LLM+RAG treatment each cycle.
    """

    def __init__(self, llm_config: dict[str, Any] | None = None) -> None:
        self._llm: Any = None
        self._retriever: Any = None
        self._compressor: Any = None
        self._llm_config: dict[str, Any] = llm_config or {}
        self._llm_log: list[dict[str, Any]] = []

    # ── lazy accessors ──────────────────────────────────────────────

    def _get_llm(self) -> Any:
        if self._llm is None:
            try:
                from firm.llm.provider import LLMService
                self._llm = LLMService(self._llm_config)
            except Exception as exc:
                log.warning(
                    "LLMService unavailable — LLM enhancement disabled for this agent",
                    exc_info=True,
                )
                raise ImportError("firm.llm.provider.LLMService unavailable") from exc
        return self._llm

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            try:
                from firm.llm.config import rag_config
                from firm.rag.retriever import RAGRetriever
                from firm.rag.store import VectorStore

                rag = rag_config()
                store = VectorStore()
                self._retriever = RAGRetriever(
                    store,
                    reranker=bool(rag.get("reranking", True)),
                    hybrid=bool(rag.get("hybrid", False)),
                    reranker_provider=rag.get("reranker_provider"),
                    reranker_model=rag.get("reranker_model"),
                )
            except Exception as exc:
                log.warning(
                    "RAG retriever unavailable — RAG context disabled for this agent",
                    exc_info=True,
                )
                raise ImportError("RAG retriever unavailable") from exc
        return self._retriever

    def _enhancement_cfg(self) -> dict[str, Any]:
        from firm.llm.config import enhancement_config

        return enhancement_config(overrides=self._llm_config.get("enhancement"))

    # ── cost gating ─────────────────────────────────────────────────

    def _signal_keys_to_enhance(self, signals: list[Signal]) -> set[tuple[str, str]]:
        """Return (symbol, strategy) pairs that merit an LLM call this cycle."""
        cfg = self._enhancement_cfg()
        min_score = float(cfg.get("min_abs_score", 0.0))
        max_n = int(cfg.get("max_signals_per_agent", 0))

        candidates = [s for s in signals if abs(s.score) >= min_score]
        candidates.sort(key=lambda s: abs(s.score), reverse=True)
        if max_n > 0:
            candidates = candidates[:max_n]
        keys = {(s.symbol, s.strategy) for s in candidates}
        skipped = len(signals) - len(keys)
        if skipped:
            log.debug(
                "LLM enhancement gated: %d/%d signals (min_abs_score=%s max=%s)",
                len(keys), len(signals), min_score, max_n or "∞",
            )
        return keys

    def _thesis_symbols_to_enhance(self, theses: list[Any]) -> set[str]:
        cfg = self._enhancement_cfg()
        min_conv = float(cfg.get("min_conviction", 0.0))
        max_n = int(cfg.get("max_theses_per_agent", 0))

        candidates = [t for t in theses if float(t.conviction) >= min_conv]
        candidates.sort(key=lambda t: float(t.conviction), reverse=True)
        if max_n > 0:
            candidates = candidates[:max_n]
        return {t.symbol for t in candidates}

    def _debate_symbols_to_enhance(self, results: list[Any]) -> set[str]:
        cfg = self._enhancement_cfg()
        min_conv = float(cfg.get("min_conviction", 0.0))
        max_n = int(cfg.get("max_debate_symbols", 0))

        candidates = [r for r in results if abs(float(r.net_conviction)) >= min_conv]
        candidates.sort(key=lambda r: abs(float(r.net_conviction)), reverse=True)
        if max_n > 0:
            candidates = candidates[:max_n]
        return {r.symbol for r in candidates}

    def _allow_portfolio_llm(self) -> bool:
        return bool(self._enhancement_cfg().get("enhance_portfolio_review", False))

    def _allow_risk_llm(self) -> bool:
        return bool(self._enhancement_cfg().get("enhance_risk_review", False))

    # ── utilities ───────────────────────────────────────────────────

    def _retrieve_context(
        self,
        symbol: str,
        query: str,
        collections: list[str] | None = None,
        n: int | None = None,
        asof: Any = None,
    ) -> str:
        """Retrieve and format RAG context for *symbol*.

        *asof* (the decision timestamp, e.g. ``ctx.now``) must be passed so
        that only documents available at-or-before that time are retrieved;
        omitting it would allow future-dated filings/news to leak into the
        decision (look-ahead).
        """
        if n is None:
            n = int(self._enhancement_cfg().get("rag_n_results", 2))
        try:
            retriever = self._get_retriever()
            docs = retriever.retrieve_for_symbol(
                symbol, query, n_results=n, collections=collections, asof=asof
            )
            return "\n\n".join(
                f"[{d.metadata.get('source', '?')}] {d.text}" for d in docs
            )
        except ImportError:
            return ""  # already logged by _get_retriever()
        except Exception:
            # A retrieval-time failure (Chroma query, embedding API, etc.) —
            # distinct from retriever *construction* failing above — was
            # previously completely silent, with no log call at all.
            log.warning(
                "RAG context retrieval failed for %s — proceeding without "
                "RAG context this cycle",
                symbol, exc_info=True,
            )
            return ""

    def _compress(self, text: str) -> str:
        """Compress *text* per config/llm.yaml's ``optimization`` section."""
        from firm.llm.config import optimization_config

        opt = optimization_config()
        if not opt.get("compression_enabled", True):
            return text
        try:
            if self._compressor is None:
                from firm.llm.compression import TokenCompressor
                self._compressor = TokenCompressor(
                    use_llmlingua=opt.get("use_llmlingua", False)
                )
            return self._compressor.compress(
                text, target_ratio=opt.get("compression_ratio", 0.5)
            )
        except Exception:
            log.warning(
                "Compression failed — falling back to a hard 3000-char "
                "truncation for this call", exc_info=True,
            )
            return text[:3000]

    def _bounded_override(
        self,
        symbol: str,
        strategy: str,
        raw_score: Any,
        raw_confidence: Any,
        fallback_score: float,
        fallback_confidence: float,
    ) -> tuple[float, float]:
        """Coerce+clamp an LLM-proposed (score, confidence) override to its
        documented range, falling back to the quant value on bad input.

        The enhancement prompts all ask for ``score`` in [-1, 1] and
        ``confidence`` in [0, 1], but nothing enforces that an LLM actually
        honours it — a hallucinated or malformed value (e.g. score=4.0,
        confidence=-0.2, or a non-numeric string) would otherwise flow
        straight into the z-scored signal the rest of the pipeline assumes
        is bounded, silently distorting cross-sectional ranking and
        downstream position sizing. Out-of-range/invalid values are clamped
        (not just logged) so a single bad LLM response can never dominate
        the cross-section; callers should still re-run
        :func:`firm.agents.analysts.zscore_signals` afterwards so the
        clamp doesn't leave the sole z-score invariant violated for the
        rest of the group.
        """
        try:
            score = float(raw_score)
            if score != score:  # NaN
                raise ValueError("NaN score")
        except (TypeError, ValueError):
            log.warning(
                "LLM returned non-numeric score for %s/%s (%r) — using quant score",
                symbol, strategy, raw_score,
            )
            score = fallback_score
        try:
            confidence = float(raw_confidence)
            if confidence != confidence:  # NaN
                raise ValueError("NaN confidence")
        except (TypeError, ValueError):
            log.warning(
                "LLM returned non-numeric confidence for %s/%s (%r) — using quant confidence",
                symbol, strategy, raw_confidence,
            )
            confidence = fallback_confidence

        clamped_score = max(-1.0, min(1.0, score))
        clamped_confidence = max(0.0, min(1.0, confidence))
        if clamped_score != score or clamped_confidence != confidence:
            log.warning(
                "LLM override out of bounds for %s/%s: score=%s->%s confidence=%s->%s "
                "(clamped to documented [-1,1]/[0,1] ranges)",
                symbol, strategy, score, clamped_score, confidence, clamped_confidence,
            )
        return clamped_score, clamped_confidence

    def _bounded_conviction(
        self, symbol: str, side: str, raw_conviction: Any, fallback: float,
    ) -> float:
        """Coerce+clamp an LLM-proposed thesis conviction to [0, 1].

        Same rationale as :meth:`_bounded_override`: the prompt documents
        the range but nothing enforces it, and an out-of-range conviction
        feeds straight into the bull/bear debate's net-conviction math.
        """
        try:
            conviction = float(raw_conviction)
            if conviction != conviction:  # NaN
                raise ValueError("NaN conviction")
        except (TypeError, ValueError):
            log.warning(
                "LLM returned non-numeric conviction for %s thesis on %s (%r) — "
                "using quant conviction", side, symbol, raw_conviction,
            )
            conviction = fallback
        clamped = max(0.0, min(1.0, conviction))
        if clamped != conviction:
            log.warning(
                "LLM conviction out of bounds for %s thesis on %s: %s->%s "
                "(clamped to documented [0,1] range)",
                side, symbol, conviction, clamped,
            )
        return clamped

    def _bounded_net_conviction(
        self, symbol: str, raw_value: Any, fallback: float,
    ) -> float:
        """Coerce+clamp an LLM-proposed debate net_conviction to [-1, 1].

        Same rationale as :meth:`_bounded_override`/:meth:`_bounded_conviction`.
        """
        try:
            value = float(raw_value)
            if value != value:  # NaN
                raise ValueError("NaN net_conviction")
        except (TypeError, ValueError):
            log.warning(
                "LLM returned non-numeric net_conviction for %s (%r) — using quant value",
                symbol, raw_value,
            )
            value = fallback
        clamped = max(-1.0, min(1.0, value))
        if clamped != value:
            log.warning(
                "LLM net_conviction out of bounds for %s: %s->%s "
                "(clamped to documented [-1,1] range)",
                symbol, value, clamped,
            )
        return clamped

    def _call_llm(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
    ) -> str | dict:
        """Call LLM and log usage, honouring enhancement cost policy."""
        from firm.llm.exceptions import LLMEnhancementSkipped

        llm = self._get_llm()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        policy = self._enhancement_cfg().get("policy", "live_calls")
        if policy == "cache_only":
            cached = llm.get_cached(messages, json_mode=json_mode)
            if cached is None:
                log.debug("enhancement policy=cache_only: cache miss, skipping LLM")
                raise LLMEnhancementSkipped("cache_only miss")
            if json_mode:
                import json
                return json.loads(cached)
            return cached

        if json_mode:
            result = llm.chat_json(messages)
        else:
            result = llm.chat(messages)
        self._llm_log.append({
            "system_preview": system[:100],
            "tokens": getattr(llm, "usage_stats", {}).get("last_tokens", 0),
        })
        return result
