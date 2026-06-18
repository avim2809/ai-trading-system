"""RAG retriever with optional cross-encoder reranking."""

from __future__ import annotations

from typing import Any

from firm.rag.models import RetrievedDoc
from firm.rag.store import VectorStore


class RAGRetriever:
    """Retrieves and optionally reranks documents from the vector store."""

    def __init__(self, store: VectorStore, reranker: bool = True) -> None:
        self._store = store
        self._reranker = None

        if reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except (ImportError, Exception):
                self._reranker = None

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
        timestamp are returned (point-in-time safety; see
        :meth:`VectorStore.query`).  *collections* restricts the search to a
        specific set of collection names.
        """
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

        where_filters = self._and(symbol_filter, type_filter)

        # Determine which collections to search.
        if collections:
            search_collections = [
                c for c in collections if c in set(self._store.list_collections())
            ]
        elif collection == "all":
            search_collections = self._store.list_collections()
        else:
            search_collections = [collection]

        if len(search_collections) == 1:
            docs_to_rerank = self._store.query(
                search_collections[0], query, n_results=n_results * 2,
                where_filters=where_filters, asof=asof,
            )
        else:
            all_docs: list[RetrievedDoc] = []
            for coll_name in search_collections:
                docs = self._store.query(coll_name, query, n_results=n_results,
                                         where_filters=where_filters, asof=asof)
                all_docs.extend(docs)
            all_docs.sort(key=lambda d: d.score, reverse=True)
            docs_to_rerank = all_docs[:n_results * 2]

        if self._reranker and docs_to_rerank:
            return self._rerank(query, docs_to_rerank, n_results)

        return docs_to_rerank[:n_results]

    def _rerank(self, query: str, docs: list[RetrievedDoc], n_results: int) -> list[RetrievedDoc]:
        """Rerank documents using a cross-encoder model."""
        pairs = [(query, d.text) for d in docs]
        scores = self._reranker.predict(pairs)

        for doc, score in zip(docs, scores):
            doc.score = float(score)

        docs.sort(key=lambda d: d.score, reverse=True)
        return docs[:n_results]

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
