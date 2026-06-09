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
    ) -> list[RetrievedDoc]:
        """Retrieve relevant documents, optionally filtering by symbol/doc_type."""
        where_filters: dict[str, Any] | None = None
        if symbols and len(symbols) == 1:
            where_filters = {"symbol": symbols[0]}
        elif symbols and len(symbols) > 1:
            where_filters = {"symbol": {"$in": symbols}}

        if doc_types and not where_filters:
            if len(doc_types) == 1:
                where_filters = {"doc_type": doc_types[0]}
            else:
                where_filters = {"doc_type": {"$in": doc_types}}
        elif doc_types and where_filters:
            # Combine with $and
            type_filter = (
                {"doc_type": doc_types[0]} if len(doc_types) == 1
                else {"doc_type": {"$in": doc_types}}
            )
            where_filters = {"$and": [where_filters, type_filter]}

        # Query multiple collections or a specific one
        if collection == "all":
            all_docs: list[RetrievedDoc] = []
            for coll_name in self._store.list_collections():
                docs = self._store.query(coll_name, query, n_results=n_results,
                                         where_filters=where_filters)
                all_docs.extend(docs)
            # Sort by score, keep top n
            all_docs.sort(key=lambda d: d.score, reverse=True)
            docs_to_rerank = all_docs[:n_results * 2]
        else:
            docs_to_rerank = self._store.query(
                collection, query, n_results=n_results * 2,
                where_filters=where_filters,
            )

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

    def retrieve_for_symbol(
        self, symbol: str, query: str, n_results: int = 3
    ) -> list[RetrievedDoc]:
        """Convenience method to retrieve docs filtered to a single symbol."""
        return self.retrieve(query, collection="all", symbols=[symbol], n_results=n_results)
