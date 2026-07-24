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
            except Exception:
                raise ImportError("firm.llm.provider.LLMService unavailable")
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
            except Exception:
                raise ImportError("RAG retriever unavailable")
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
        except Exception:
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
