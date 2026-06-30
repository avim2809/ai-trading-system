"""Document chunking utilities for RAG ingestion."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from firm.rag.models import Document


class DocumentChunker:
    """Splits text into overlapping chunks suitable for embedding.

    When *contextual* is True, a short metadata header (doc_type/source/
    symbol/date/section/...) is prepended to each chunk's text *before*
    embedding ("contextual retrieval"): research shows metadata-enriched
    chunks improve retrieval precision. It is opt-in and defaults off — the
    "metadata is the single largest gain" claim was not robustly supported,
    and the gains were measured on code/financial-text, so it should be
    piloted on local data (and a re-index is needed when toggling it, since
    the embedded text — and thus the chunk id — changes).
    """

    # Metadata keys worth surfacing into the embedded text, in header order.
    _CONTEXT_KEYS = ("doc_type", "source", "symbol", "strategy", "date", "section")

    def __init__(
        self, chunk_size: int = 500, overlap: int = 50, contextual: bool = False
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.contextual = contextual

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _contextualize(self, text: str, metadata: dict[str, Any]) -> str:
        """Prepend a metadata header to *text* when contextual mode is on."""
        if not self.contextual:
            return text
        parts = [f"{k}={metadata[k]}" for k in self._CONTEXT_KEYS if metadata.get(k)]
        if not parts:
            return text
        return "[" + " | ".join(parts) + "]\n" + text

    def _make_chunk(
        self, chunk_text: str, metadata: dict[str, Any], index: int
    ) -> Document:
        """Build a Document, applying contextual header + stable id."""
        embed_text = self._contextualize(chunk_text, metadata)
        doc_id = hashlib.sha256(embed_text.encode()).hexdigest()[:16]
        return Document(
            doc_id=f"{metadata.get('source', 'doc')}_{doc_id}",
            text=embed_text,
            metadata={**metadata, "chunk_index": index},
        )

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """Split text into overlapping chunks by sentence boundaries."""
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[Document] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(self._make_chunk(chunk_text, metadata, len(chunks)))

                # Keep overlap sentences
                overlap_tokens = 0
                overlap_start = len(current_sentences)
                for j in range(len(current_sentences) - 1, -1, -1):
                    overlap_tokens += self._estimate_tokens(current_sentences[j])
                    if overlap_tokens >= self.overlap:
                        overlap_start = j
                        break
                current_sentences = current_sentences[overlap_start:]
                current_tokens = sum(self._estimate_tokens(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(self._make_chunk(chunk_text, metadata, len(chunks)))

        return chunks

    def chunk_by_sections(
        self,
        text: str,
        section_headers: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Split text at section headers, then chunk each section."""
        metadata = metadata or {}
        pattern = "|".join(re.escape(h) for h in section_headers)
        parts = re.split(f"({pattern})", text, flags=re.IGNORECASE)

        all_docs: list[Document] = []
        current_header = "preamble"

        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if any(stripped.lower() == h.lower() for h in section_headers):
                current_header = stripped
                continue

            section_meta = {**metadata, "section": current_header}
            section_chunks = self.chunk(stripped, section_meta)
            all_docs.extend(section_chunks)

        return all_docs
