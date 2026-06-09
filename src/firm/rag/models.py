"""Data models for the RAG document pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document chunk ready for embedding and storage."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDoc:
    """A document returned from a vector similarity search."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
