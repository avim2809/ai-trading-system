"""RAG retriever with optional hybrid (BM25 + dense) recall and reranking.

The research-recommended retrieval recipe is hybrid lexical+dense recall
followed by a cross-encoder reranker. Dense recall comes from the Chroma
vector store; the optional lexical channel is an in-memory BM25 index over the
collection, fused with the dense list via Reciprocal Rank Fusion (RRF) before
reranking. The lexical channel recovers exact-token matches (tickers, run-ids)
that pure dense retrieval can miss.

Both channels honour the same point-in-time (*asof*) and symbol/doc_type
filters, so hybrid mode never leaks future-dated documents.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from firm.rag.models import RetrievedDoc
from firm.rag.store import VectorStore, _asof_str, _doc_available_by

log = logging.getLogger(__name__)

_RRF_K = 60  # Reciprocal Rank Fusion constant (standard default).


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class RAGRetriever:
    """Retrieves, optionally hybrid-fuses, and reranks documents."""

    def __init__(
        self,
        store: VectorStore,
        reranker: bool = True,
        hybrid: bool = False,
        reranker_provider: str | None = None,
        reranker_model: str | None = None,
    ) -> None:
        self._store = store
        self._hybrid = hybrid
        self._reranker = None
        # Lazy BM25 index cache per collection: name -> (doc_count, bm25, docs).
        self._bm25_cache: dict[str, tuple[int, Any, list[RetrievedDoc]]] = {}

        if reranker:
            self._reranker = self._build_reranker(reranker_provider, reranker_model)

    @staticmethod
    def _build_reranker(provider: str | None = None, model: str | None = None):
        """Pick the reranker: hosted Voyage (default), local cross-encoder, or none."""
        from firm.llm.config import rag_config

        cfg = rag_config()
        provider = provider or cfg.get("reranker_provider", "voyage")
        model = model or cfg.get("reranker_model")
        if provider == "none":
            return None
        if provider == "voyage":
            try:
                from firm.rag.voyage import VoyageReranker

                return VoyageReranker(model=model or "rerank-2.5")
            except Exception:
                log.warning("Voyage reranker unavailable; continuing without rerank",
                            exc_info=True)
                return None
        # Local fallback — requires the optional ``firm[local]`` extra (torch).
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder(model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            log.warning("Local reranker unavailable; continuing without rerank",
                        exc_info=True)
            return None

    def retrieve(
        self,
        query: str,
        collection: str = "all",
        symbols: list[str] | None = None,
        doc_types: list[str] | None = None,
        n_results: int = 5,
        collections: list[str] | None = None,
        asof: Any = None,
    ) -> list[RetrievedDoc]:
        """Retrieve relevant documents, optionally filtering by symbol/doc_type.

        When *asof* is supplied, only documents available at-or-before that
        timestamp are returned (point-in-time safety). *collections* restricts
        the search to a specific set of collection names.
        """
        where_filters = self._build_where(symbols, doc_types)

        if collections:
            known = set(self._store.list_collections())
            search_collections = [c for c in collections if c in known]
        elif collection == "all":
            search_collections = self._store.list_collections()
        else:
            search_collections = [collection]

        pool: list[RetrievedDoc] = []
        for coll_name in search_collections:
            pool.extend(self._candidates(
                coll_name, query, n_results * 2, where_filters,
                symbols, doc_types, asof,
            ))

        pool.sort(key=lambda d: d.score, reverse=True)
        candidates = pool[: n_results * 2]

        if self._reranker and candidates:
            return self._rerank(query, candidates, n_results)
        return candidates[:n_results]

    # ── recall ──────────────────────────────────────────────────────

    def _candidates(
        self,
        collection: str,
        query: str,
        n: int,
        where_filters: dict[str, Any] | None,
        symbols: list[str] | None,
        doc_types: list[str] | None,
        asof: Any,
    ) -> list[RetrievedDoc]:
        """Dense candidates for one collection, fused with BM25 when hybrid."""
        dense = self._store.query(
            collection, query, n_results=n, where_filters=where_filters, asof=asof
        )
        if not self._hybrid:
            return dense
        lexical = self._bm25_query(collection, query, n, symbols, doc_types, asof)
        if not lexical:
            return dense
        return self._rrf(dense, lexical)

    def _bm25_query(
        self,
        collection: str,
        query: str,
        n: int,
        symbols: list[str] | None,
        doc_types: list[str] | None,
        asof: Any,
    ) -> list[RetrievedDoc]:
        """Top-n BM25 hits for *query*, filtered like the dense channel."""
        index = self._get_bm25(collection)
        if index is None:
            return []
        bm25, docs = index
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        out: list[RetrievedDoc] = []
        for score, doc in ranked:
            if score <= 0:
                break
            if not self._passes_filters(doc.metadata, symbols, doc_types, asof):
                continue
            out.append(RetrievedDoc(
                doc_id=doc.doc_id, text=doc.text, metadata=doc.metadata,
                score=float(score),
            ))
            if len(out) >= n:
                break
        return out

    def _get_bm25(self, collection: str) -> tuple[Any, list[RetrievedDoc]] | None:
        """Build/cache a BM25 index over *collection*; None if unavailable."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return None

        docs = self._store.get_all(collection)
        cached = self._bm25_cache.get(collection)
        if cached is not None and cached[0] == len(docs):
            return cached[1], cached[2]
        if not docs:
            return None

        bm25 = BM25Okapi([_tokenize(d.text) for d in docs])
        self._bm25_cache[collection] = (len(docs), bm25, docs)
        return bm25, docs

    @staticmethod
    def _passes_filters(
        metadata: dict[str, Any],
        symbols: list[str] | None,
        doc_types: list[str] | None,
        asof: Any,
    ) -> bool:
        """Replicate the dense channel's where/asof filtering for BM25 hits.

        Shares :func:`firm.rag.store._doc_available_by` with the dense
        channel's own asof check, so both channels fail closed identically
        on a missing/``None``/malformed ``date`` instead of maintaining two
        subtly different implementations.
        """
        if symbols and metadata.get("symbol") not in symbols:
            return False
        if doc_types and metadata.get("doc_type") not in doc_types:
            return False
        if asof is not None and not _doc_available_by(metadata, _asof_str(asof)):
            return False
        return True

    @staticmethod
    def _rrf(
        dense: list[RetrievedDoc], lexical: list[RetrievedDoc]
    ) -> list[RetrievedDoc]:
        """Fuse two ranked lists via Reciprocal Rank Fusion."""
        fused: dict[str, RetrievedDoc] = {}
        rrf: dict[str, float] = {}
        for ranked in (dense, lexical):
            for rank, doc in enumerate(ranked):
                rrf[doc.doc_id] = rrf.get(doc.doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
                fused.setdefault(doc.doc_id, doc)
        for doc_id, score in rrf.items():
            fused[doc_id].score = score
        return sorted(fused.values(), key=lambda d: d.score, reverse=True)

    # ── rerank ──────────────────────────────────────────────────────

    def _rerank(
        self, query: str, docs: list[RetrievedDoc], n_results: int
    ) -> list[RetrievedDoc]:
        """Rerank documents using a cross-encoder model."""
        pairs = [(query, d.text) for d in docs]
        scores = self._reranker.predict(pairs)
        for doc, score in zip(docs, scores):
            doc.score = float(score)
        docs.sort(key=lambda d: d.score, reverse=True)
        return docs[:n_results]

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _build_where(
        symbols: list[str] | None, doc_types: list[str] | None
    ) -> dict[str, Any] | None:
        """Build a Chroma where-clause from symbol/doc_type filters."""
        symbol_filter: dict[str, Any] | None = None
        if symbols and len(symbols) == 1:
            symbol_filter = {"symbol": symbols[0]}
        elif symbols and len(symbols) > 1:
            symbol_filter = {"symbol": {"$in": symbols}}

        type_filter: dict[str, Any] | None = None
        if doc_types:
            type_filter = (
                {"doc_type": doc_types[0]} if len(doc_types) == 1
                else {"doc_type": {"$in": doc_types}}
            )
        return RAGRetriever._and(symbol_filter, type_filter)

    @staticmethod
    def _and(
        a: dict[str, Any] | None, b: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Combine two where-clauses with ``$and`` (single clause unwrapped)."""
        if a and b:
            return {"$and": [a, b]}
        return a or b

    def retrieve_for_symbol(
        self,
        symbol: str,
        query: str,
        n_results: int = 3,
        collections: list[str] | None = None,
        asof: Any = None,
    ) -> list[RetrievedDoc]:
        """Convenience method to retrieve docs filtered to a single symbol."""
        return self.retrieve(
            query,
            collection="all",
            symbols=[symbol],
            n_results=n_results,
            collections=collections,
            asof=asof,
        )
