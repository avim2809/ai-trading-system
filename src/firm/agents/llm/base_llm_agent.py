"""Shared mixin providing LLM + RAG utilities for enhanced agents."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class LLMAgentMixin:
    """Mixin providing LLM + RAG utilities for enhanced agents.

    Lazy-initialises the LLM service, RAG retriever, and token compressor
    on first use so the system starts even when the ``llm`` extra is not
    installed.
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
                from firm.rag.store import VectorStore
                from firm.rag.retriever import RAGRetriever
                store = VectorStore()
                self._retriever = RAGRetriever(store)
            except Exception:
                raise ImportError("RAG retriever unavailable")
        return self._retriever

    # ── utilities ───────────────────────────────────────────────────

    def _retrieve_context(
        self,
        symbol: str,
        query: str,
        collections: list[str] | None = None,
        n: int = 3,
        asof: Any = None,
    ) -> str:
        """Retrieve and format RAG context for *symbol*.

        *asof* (the decision timestamp, e.g. ``ctx.now``) must be passed so
        that only documents available at-or-before that time are retrieved;
        omitting it would allow future-dated filings/news to leak into the
        decision (look-ahead).
        """
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
        """Compress *text* using TokenCompressor, falling back to truncation."""
        try:
            if self._compressor is None:
                from firm.llm.compression import TokenCompressor
                self._compressor = TokenCompressor()
            return self._compressor.compress(text)
        except Exception:
            return text[:3000]

    def _call_llm(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
    ) -> str | dict:
        """Call LLM and log usage."""
        llm = self._get_llm()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if json_mode:
            result = llm.chat_json(messages)
        else:
            result = llm.chat(messages)
        self._llm_log.append({
            "system_preview": system[:100],
            "tokens": getattr(llm, "usage_stats", {}).get("last_tokens", 0),
        })
        return result
