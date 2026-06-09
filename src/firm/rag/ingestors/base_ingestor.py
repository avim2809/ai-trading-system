"""Abstract base class for all RAG document ingestors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from firm.rag.chunker import DocumentChunker
from firm.rag.store import VectorStore


class BaseIngestor(ABC):
    """Base class for document ingestion pipelines.

    Subclasses implement :meth:`ingest` to fetch, chunk, and store
    documents into the vector store.
    """

    def __init__(self, store: VectorStore, chunker: DocumentChunker) -> None:
        self.store = store
        self.chunker = chunker

    @abstractmethod
    def ingest(self, **kwargs) -> int:
        """Run the ingestion pipeline. Returns count of documents ingested."""
        ...
