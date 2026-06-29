"""ChromaDB-backed vector store for document embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from firm.rag.embeddings import get_model_info
from firm.rag.models import Document, RetrievedDoc

log = logging.getLogger(__name__)


def _asof_str(asof: Any) -> str:
    """Coerce a datetime/date/ISO-string *asof* to an ISO ``YYYY-MM-DD``."""
    if hasattr(asof, "strftime"):
        return asof.strftime("%Y-%m-%d")
    return str(asof)[:10]


def _and_filters(
    a: dict[str, Any] | None, b: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Combine two ChromaDB where-clauses with ``$and`` (single clause unwrapped)."""
    if a and b:
        return {"$and": [a, b]}
    return a or b


class VectorStore:
    """Persistent vector store wrapping ChromaDB with sentence-transformer embeddings."""

    def __init__(
        self,
        persist_dir: str = "data/vectordb",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError:
            raise ImportError(
                "chromadb and sentence-transformers are required for VectorStore. "
                "Install with: pip install 'firm[llm]'"
            )

        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_model = embedding_model

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

    @property
    def current_model(self) -> str:
        """Return the active embedding model identifier."""
        return self._embedding_model

    def requires_reindex(self, new_model: str) -> bool:
        """Return True if switching to *new_model* would need a full re-index.

        A re-index is required whenever the dimensionality changes, which makes
        existing vectors incompatible with the new embedding function.
        """
        if new_model == self._embedding_model:
            return False
        current_info = get_model_info(self._embedding_model)
        new_info = get_model_info(new_model)
        if current_info is None or new_info is None:
            return True
        return current_info.dimensions != new_info.dimensions

    def get_or_create_collection(self, name: str):
        """Get or create a named ChromaDB collection."""
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_fn,
        )

    def add_documents(self, collection_name: str, docs: list[Document]) -> int:
        """Add documents to a collection. Returns count of documents added."""
        if not docs:
            return 0

        collection = self.get_or_create_collection(collection_name)
        ids = [d.doc_id for d in docs]
        texts = [d.text for d in docs]
        metadatas = [d.metadata for d in docs]

        # ChromaDB silently skips duplicates by ID
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(docs)

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where_filters: dict[str, Any] | None = None,
        asof: Any = None,
    ) -> list[RetrievedDoc]:
        """Query a collection for similar documents.

        When *asof* (a ``datetime``/date or ISO string) is given, only
        documents whose ``date`` metadata is ``<= asof`` are returned, so
        future-dated filings/news can never leak into a point-in-time
        decision.
        """
        if asof is not None:
            where_filters = _and_filters(where_filters, {"date": {"$lte": _asof_str(asof)}})
        collection = self.get_or_create_collection(collection_name)

        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, collection.count() or n_results),
        }
        if where_filters:
            kwargs["where"] = where_filters

        if collection.count() == 0:
            return []

        results = collection.query(**kwargs)

        docs: list[RetrievedDoc] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append(RetrievedDoc(
                    doc_id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=1.0 - (results["distances"][0][i] if results["distances"] else 0.0),
                ))
        return docs

    def delete_collection(self, name: str) -> None:
        """Delete an entire collection.

        A missing collection is benign (idempotent delete); any other failure
        is logged with a traceback rather than silently swallowed.
        """
        try:
            self._client.delete_collection(name)
        except Exception:
            log.warning("Failed to delete collection %r", name, exc_info=True)

    def list_collections(self) -> list[str]:
        """List all collection names."""
        return [c.name for c in self._client.list_collections()]

    def stats(self) -> dict[str, Any]:
        """Return per-collection document counts."""
        result: dict[str, Any] = {}
        for coll in self._client.list_collections():
            result[coll.name] = coll.count()
        result["_total"] = sum(result.values())
        return result
