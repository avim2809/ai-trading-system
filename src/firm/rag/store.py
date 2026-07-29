"""ChromaDB-backed vector store for document embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from firm.rag.dates import UNKNOWN_DATE
from firm.rag.embeddings import get_model_info
from firm.rag.models import Document, RetrievedDoc

log = logging.getLogger(__name__)


def _asof_str(asof: Any) -> str:
    """Coerce a datetime/date/ISO-string *asof* to an ISO ``YYYY-MM-DD``."""
    if hasattr(asof, "strftime"):
        return asof.strftime("%Y-%m-%d")
    return str(asof)[:10]


def _doc_available_by(metadata: dict[str, Any], asof_str: str) -> bool:
    """True iff *metadata*'s ``date`` is at-or-before ``asof_str``.

    Fails closed on anything that isn't a real ISO date. Plain
    ``metadata.get("date", UNKNOWN_DATE)`` only substitutes the sentinel
    when the key is entirely *missing* — a doc whose metadata explicitly
    carries ``"date": None`` (a malformed/legacy record that bypassed
    :func:`firm.rag.dates.normalize_date`) still slips through as ``None``
    and crashes the ``>`` comparison instead of being excluded. Treating a
    missing/``None``/malformed date the same as :data:`UNKNOWN_DATE` (always
    in the future, so always excluded) is the correct fail-closed behaviour:
    a doc of unknown vintage must never be silently treated as available,
    since that is exactly the look-ahead leak this filter exists to prevent.
    """
    date = metadata.get("date") or UNKNOWN_DATE
    try:
        return str(date)[:10] <= asof_str
    except Exception:
        log.warning(
            "RAG asof filter: malformed date metadata %r treated as unavailable", date
        )
        return False


class VectorStore:
    """Persistent ChromaDB vector store with pluggable embeddings (Voyage / local ST)."""

    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
    ) -> None:
        # Resolve unset args from config/llm.yaml so bare ``VectorStore()`` calls
        # pick up the configured (external by default) embedding provider.
        from firm.llm.config import rag_config

        cfg = rag_config()
        persist_dir = persist_dir or cfg.get("persist_dir", "data/vectordb")
        embedding_provider = embedding_provider or cfg.get("embedding_provider", "voyage")
        embedding_model = embedding_model or cfg.get("embedding_model")

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for VectorStore. "
                "Install with: pip install 'firm[llm]'"
            )

        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._embedding_fn = self._build_embedding_fn(embedding_provider, embedding_model)

    @staticmethod
    def _build_embedding_fn(provider: str, model: str | None):
        """Pick the embedding function: hosted Voyage (default) or local ST."""
        if provider == "voyage":
            from firm.rag.voyage import VoyageEmbeddingFunction

            return VoyageEmbeddingFunction(model=model or "voyage-finance-2")
        # Local fallback — requires the optional ``firm[local]`` extra (torch).
        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
        except ImportError:
            raise ImportError(
                "Local embeddings need sentence-transformers. Install 'firm[local]', "
                "or set rag.embedding_provider to 'voyage' in config/llm.yaml."
            )
        return SentenceTransformerEmbeddingFunction(model_name=model or "all-MiniLM-L6-v2")

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
        """Add documents to a collection. Returns count of *new* documents added.

        Skips documents whose id already exists in the collection first —
        ``upsert()`` does not skip duplicates, it re-embeds and overwrites
        them, so re-running ingestion over already-stored content would
        otherwise re-pay the (Voyage) embedding cost for no change in the
        stored result. Chunk ids are a content+identity hash (see
        ``DocumentChunker._make_chunk``), so this only skips genuinely
        unchanged documents — different content hashes to a different id.
        """
        if not docs:
            return 0

        collection = self.get_or_create_collection(collection_name)
        ids = [d.doc_id for d in docs]

        existing_ids = set(collection.get(ids=ids, include=[])["ids"])
        new_docs = [d for d in docs if d.doc_id not in existing_ids]
        if not new_docs:
            return 0

        collection.upsert(
            ids=[d.doc_id for d in new_docs],
            documents=[d.text for d in new_docs],
            metadatas=[d.metadata for d in new_docs],
        )
        return len(new_docs)

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

        Chroma's ``where`` filter only supports ``$lte``/``$gte`` on numeric
        operands, not the sortable ISO date *strings* this system stores (see
        ``firm.rag.dates``) — passing a string there raises a ``ValueError``.
        So the as-of bound is applied as a plain string comparison in Python
        instead: over-fetch the dense-similarity candidate pool (the whole
        collection, when *asof* is set, since we don't know in advance how
        many will be filtered out) and stop once *n_results* pass the cutoff.
        Fine at this system's targeted thousands-scale collections — the same
        assumption :meth:`get_all` already makes for hybrid/BM25.
        """
        collection = self.get_or_create_collection(collection_name)
        count = collection.count()
        if count == 0:
            return []

        fetch_n = count if asof is not None else n_results
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(fetch_n, count),
        }
        if where_filters:
            kwargs["where"] = where_filters

        results = collection.query(**kwargs)

        asof_str = _asof_str(asof) if asof is not None else None
        docs: list[RetrievedDoc] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                if asof_str is not None and not _doc_available_by(metadata, asof_str):
                    continue
                docs.append(RetrievedDoc(
                    doc_id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    metadata=metadata,
                    score=1.0 - (results["distances"][0][i] if results["distances"] else 0.0),
                ))
                if len(docs) >= n_results:
                    break
        return docs

    def get_all(self, collection_name: str) -> list[RetrievedDoc]:
        """Return every document in a collection (id, text, metadata).

        Used to build an in-memory BM25 lexical index for hybrid retrieval.
        Suitable at the thousands–low-millions scale this system targets;
        callers should not invoke it on collections beyond that.
        """
        collection = self.get_or_create_collection(collection_name)
        if collection.count() == 0:
            return []
        res = collection.get()
        docs: list[RetrievedDoc] = []
        ids = res.get("ids") or []
        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        for i, doc_id in enumerate(ids):
            docs.append(RetrievedDoc(
                doc_id=doc_id,
                text=documents[i] if i < len(documents) else "",
                metadata=metadatas[i] if i < len(metadatas) else {},
                score=0.0,
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
