"""Grounded question-answering assistant over backtest run artifacts.

Combines the two retrieval modes the research recommends:

* **Numeric questions → SQL.** Figures are computed deterministically by
  DuckDB (:class:`firm.rag.structured.RunStore`); the LLM writes the query and
  narrates the result but never does the arithmetic itself (frontier models
  hallucinate 10–20% on multi-step financial math).
* **Narrative questions → vector retrieval.** ``run_notes`` and other
  collections supply grounding context via :class:`firm.rag.retriever.RAGRetriever`.

The final answer is synthesised from the SQL result table plus retrieved
chunks, and the structured result + source chunks are returned alongside it so
callers can verify every claim. When the configured provider is Anthropic
Claude, the large static context is sent with ``cache_control`` so repeated
questions over the same runs hit cached input (~90% cheaper).

The LLM is injectable (``llm=`` / a callable) so the routing and SQL paths can
be tested without any network access.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from firm.llm.config import (
    assistant_config,
    is_anthropic,
    load_llm_config,
    provider_config,
    rag_config,
)
from firm.rag.structured import ReadOnlyQueryError, RunStore

log = logging.getLogger(__name__)

# Heuristic: does the question want a number/aggregate (→ try SQL first)?
_NUMERIC_HINTS = re.compile(
    r"\b(sharpe|sortino|drawdown|return|returns|cagr|calmar|pnl|p&l|profit|loss|"
    r"trade|trades|win|hit rate|alpha|beta|volatility|how many|count|number of|"
    r"average|avg|mean|median|total|sum|best|worst|top|bottom|highest|lowest|"
    r"most|least|rank|compare|nav)\b",
    re.IGNORECASE,
)


@dataclass
class AssistantAnswer:
    """Result of :meth:`TradingAssistant.ask`."""

    answer: str
    used_sql: bool = False
    sql: str | None = None
    rows: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    error: str | None = None


class TradingAssistant:
    """Answers questions over backtest runs: numeric→SQL, narrative→retrieval."""

    def __init__(
        self,
        run_store: RunStore | None = None,
        retriever: Any = None,
        llm: Any = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config if config is not None else load_llm_config()
        self._rag_cfg = rag_config(cfg)
        self._provider_cfg = provider_config(cfg)
        self._assistant_cfg = assistant_config(cfg)

        self._run_store = run_store or RunStore(self._rag_cfg["runs_dir"])
        self._retriever = retriever          # may be None → lazy-built on demand
        self._retriever_built = retriever is not None
        self._llm = llm                      # injectable; lazy-built otherwise
        self._llm_built = llm is not None

    # ── lazy dependencies ───────────────────────────────────────────

    def _get_llm(self) -> Any:
        if not self._llm_built:
            from firm.llm.provider import LLMService
            self._llm = LLMService(self._provider_cfg)
            self._llm_built = True
        if self._llm is None:
            raise RuntimeError("No LLM available for the assistant.")
        return self._llm

    def _get_retriever(self) -> Any:
        if not self._retriever_built:
            try:
                from firm.rag.retriever import RAGRetriever
                from firm.rag.store import VectorStore
                store = VectorStore(
                    persist_dir=self._rag_cfg["persist_dir"],
                    embedding_model=self._rag_cfg["embedding_model"],
                )
                self._retriever = RAGRetriever(store, reranker=self._rag_cfg["reranking"])
            except Exception:
                log.warning("RAG retriever unavailable; narrative grounding disabled",
                            exc_info=True)
                self._retriever = None
            self._retriever_built = True
        return self._retriever

    @property
    def _model(self) -> str:
        return self._provider_cfg["default_model"]

    # ── message construction (with optional Anthropic prompt caching) ─

    def _messages(self, system_static: str, instructions: str, user: str) -> list[dict]:
        """Build chat messages, marking *system_static* cacheable on Claude.

        On Anthropic with caching enabled, the large static block (schema +
        retrieved context) is sent as a cached content block so repeat
        questions over the same runs reuse it at ~0.1x input cost.
        """
        use_cache = self._assistant_cfg["prompt_caching"] and is_anthropic(self._model)
        if use_cache:
            system_content = [
                {"type": "text", "text": system_static,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": instructions},
            ]
        else:
            system_content = f"{system_static}\n\n{instructions}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user},
        ]

    # ── pipeline steps ──────────────────────────────────────────────

    @staticmethod
    def _wants_numbers(question: str) -> bool:
        return bool(_NUMERIC_HINTS.search(question))

    @staticmethod
    def _clean_sql(raw: str) -> str:
        """Strip code fences / prose around a generated SQL statement."""
        text = raw.strip()
        fence = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        # Keep from the first SELECT/WITH onward.
        m = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
        if m:
            text = text[m.start():]
        return text.strip().rstrip(";").strip()

    def _generate_sql(self, question: str) -> str:
        schema = self._run_store.schema()
        instructions = (
            "Write a single DuckDB SQL SELECT that answers the user's question "
            "using only the tables above. Return ONLY the SQL — no prose, no "
            "code fences. Use LIMIT for 'top/worst N' questions."
        )
        messages = self._messages(schema, instructions, question)
        raw = self._get_llm().chat(messages, temperature=0.0)
        return self._clean_sql(raw)

    def _retrieve(self, question: str, asof: Any) -> list:
        retriever = self._get_retriever()
        if retriever is None:
            return []
        try:
            return retriever.retrieve(
                question,
                collection="all",
                n_results=self._assistant_cfg["n_results"],
                asof=asof,
            )
        except Exception:
            log.warning("Retrieval failed", exc_info=True)
            return []

    @staticmethod
    def _rows_to_text(rows: list[dict]) -> str:
        if not rows:
            return "(no rows)"
        cols = list(rows[0].keys())
        lines = [" | ".join(cols)]
        for r in rows[:50]:
            lines.append(" | ".join(str(r.get(c)) for c in cols))
        return "\n".join(lines)

    def _synthesize(
        self, question: str, sql: str | None, rows: list[dict], docs: list
    ) -> str:
        context_parts: list[str] = []
        if sql is not None:
            context_parts.append(
                f"SQL query (authoritative — these numbers are exact):\n{sql}\n\n"
                f"Result:\n{self._rows_to_text(rows)}"
            )
        if docs:
            ctx = "\n\n".join(
                f"[{d.metadata.get('source', '?')}] {d.text}" for d in docs
            )
            context_parts.append(f"Retrieved context:\n{ctx}")
        static = "\n\n---\n\n".join(context_parts) or "(no data available)"

        instructions = (
            "Answer the user's question using ONLY the data above. The SQL "
            "result figures are authoritative — quote them exactly and never "
            "recompute or estimate numbers yourself. Cite sources in brackets "
            "(e.g. the run_id or [source]). If the data does not answer the "
            "question, say so plainly."
        )
        messages = self._messages(static, instructions, question)
        return self._get_llm().chat(messages)

    # ── public API ──────────────────────────────────────────────────

    def ask(self, question: str, asof: Any = None) -> AssistantAnswer:
        """Answer *question* grounded in run artifacts.

        Numeric questions are answered from SQL over the structured views;
        narrative grounding is always attached. *asof*, when given, restricts
        retrieved context to documents available at-or-before that timestamp.
        """
        sql: str | None = None
        rows: list[dict] = []
        used_sql = False
        error: str | None = None

        docs = self._retrieve(question, asof)

        if self._wants_numbers(question):
            try:
                sql = self._generate_sql(question)
                df = self._run_store.query(sql)
                rows = df.to_dict(orient="records")
                used_sql = True
            except ReadOnlyQueryError as exc:
                error = f"Rejected non-read-only SQL: {exc}"
                sql = None
            except Exception as exc:
                error = f"SQL step failed: {exc}"
                log.warning("Assistant SQL step failed", exc_info=True)
                sql = None

        sources = [
            {
                "doc_id": d.doc_id,
                "source": d.metadata.get("source"),
                "date": d.metadata.get("date"),
                "text": d.text[:300],
                "score": round(float(d.score), 4),
            }
            for d in docs
        ]

        try:
            answer = self._synthesize(question, sql, rows, docs)
        except Exception as exc:
            log.warning("assistant_synthesis_failed question=%r", question, exc_info=True)
            # Keep the structured data usable even when the LLM is unavailable.
            answer = (
                "(LLM unavailable — returning retrieved data only.) "
                f"{self._rows_to_text(rows) if rows else ''}"
            ).strip()
            error = error or f"Synthesis failed: {exc}"

        return AssistantAnswer(
            answer=answer,
            used_sql=used_sql,
            sql=sql,
            rows=rows,
            sources=sources,
            error=error,
        )
