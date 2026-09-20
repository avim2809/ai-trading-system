"""Scheduled RAG "news" collection ingestion (in-app, daily).

:class:`~firm.rag.ingestors.news_ingestor.NewsIngestor` already knows how to
fetch real article headline+body text for the live universe — Yahoo Finance
RSS (no key required) plus Tiingo/Massive/AlphaVantage when their keys are
configured, see that module's own fallback-chain comments — and write it
into the Chroma "news" collection that every LLM-enhanced agent's
``_retrieve_context(..., collections=["news", ...])`` call already queries
(see ``firm.agents.llm.base_llm_agent.LLMAgentMixin._retrieve_context``).
Until this module, nothing ever invoked it on a recurring schedule — only
the manual ``scripts/ingest_docs.py --news`` CLI and
``POST /api/llm/rag/ingest`` did — so in production the "news" collection
sat at zero documents and every one of those ``_retrieve_context`` calls
silently fell back to quant-only despite ``config/llm.yaml`` already
enabling ``llm_enhanced`` mode for the sentiment/fundamental analysts.

Runs once/day (see ``news_ingestion: {enabled, hour, days}`` in
config/live*.yaml) — this feeds a multi-day drift signal per the research
motivating this feature (Glasserman & Lin, arXiv 2309.17322), not an
intraday reaction, so a daily cadence is sufficient. Off by default,
gated behind ``news_ingestion.enabled`` (added to the systemd auto-start
allowlist in ``firm.live.provider_utils`` — same "ships disabled unless
explicitly wired through" convention as every other knob added this
session, to avoid the silent-drop bug class documented there).
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_DAYS = 3

# Serializes ingestion the same way firm.live.fundamentals_refresh serializes
# its own cache refresh — concurrent callers (e.g. a manual API-triggered
# ingest racing the scheduled job) block rather than racing the same Chroma
# collection.
_ingest_lock = threading.Lock()


def ingest_recent_news(symbols: list[str], *, days: int = _DEFAULT_LOOKBACK_DAYS) -> int:
    """Fetch and store recent news for *symbols* into the "news" RAG collection.

    Returns the number of newly-added document chunks (0 on an empty
    symbol list or a hard failure — never raises, since this is meant to
    run unattended off a scheduler).

    Idempotent by design: :meth:`~firm.rag.store.VectorStore.add_documents`
    skips any document whose content-hash id already exists, so re-running
    this daily with an overlapping lookback window (the default 3 days)
    never duplicates already-ingested articles or re-pays their embedding
    cost — it only ever adds what's genuinely new since the last run.
    """
    if not symbols:
        log.warning("News ingestion skipped — empty symbol list")
        return 0

    with _ingest_lock:
        try:
            from firm.llm.config import rag_config
            from firm.rag.chunker import DocumentChunker
            from firm.rag.ingestors.news_ingestor import NewsIngestor
            from firm.rag.store import VectorStore

            rag = rag_config()
            store = VectorStore()
            chunker = DocumentChunker(
                chunk_size=int(rag.get("chunk_size", 500)),
                overlap=int(rag.get("chunk_overlap", 50)),
            )
            ingestor = NewsIngestor(store, chunker)
            count = ingestor.ingest(symbols=list(symbols), days=days)
            log.info(
                "News ingestion done: %d new chunks across %d symbols (lookback=%dd)",
                count, len(symbols), days,
            )
            return count
        except Exception:
            log.error("News ingestion failed", exc_info=True)
            return 0


def _ingest_in_background(
    symbols: list[str], *, days: int, reason: str,
) -> None:
    def _run() -> None:
        try:
            ingest_recent_news(symbols, days=days)
        except Exception:
            log.error("Background news ingestion failed (%s)", reason, exc_info=True)

    thread = threading.Thread(
        target=_run,
        name=f"news-ingestion-{reason}",
        daemon=True,
    )
    thread.start()
    log.info("Started background news ingestion (%s)", reason)


def run_scheduled_news_ingestion(
    symbols: list[str], *, days: int = _DEFAULT_LOOKBACK_DAYS,
) -> None:
    """APScheduler entrypoint — daily "news" RAG collection refresh.

    Mirrors ``firm.live.fundamentals_refresh.run_scheduled_fundamentals_refresh``:
    fires the actual work on a background thread so a slow/failing ingest
    (network calls to several providers) never blocks the scheduler's own
    thread or delays the next job tick.
    """
    log.info("Scheduled news ingestion")
    _ingest_in_background(symbols, days=days, reason="scheduled")
