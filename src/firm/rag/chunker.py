"""Document chunking utilities for RAG ingestion."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from firm.rag.models import Document


class DocumentChunker:
    """Splits text into overlapping chunks suitable for embedding."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

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
                doc_id = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
                chunks.append(Document(
                    doc_id=f"{metadata.get('source', 'doc')}_{doc_id}",
                    text=chunk_text,
                    metadata={**metadata, "chunk_index": len(chunks)},
                ))

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
            doc_id = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
            chunks.append(Document(
                doc_id=f"{metadata.get('source', 'doc')}_{doc_id}",
                text=chunk_text,
                metadata={**metadata, "chunk_index": len(chunks)},
            ))

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
