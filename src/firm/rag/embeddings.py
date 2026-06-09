"""Registry of supported embedding models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmbeddingModelInfo:
    model_id: str
    name: str
    dimensions: int
    size_mb: int
    quality: str  # "good", "better", "excellent"
    speed: str  # "very_fast", "fast", "medium"
    description: str


EMBEDDING_MODELS: dict[str, EmbeddingModelInfo] = {
    "all-MiniLM-L6-v2": EmbeddingModelInfo(
        model_id="all-MiniLM-L6-v2",
        name="MiniLM L6 v2",
        dimensions=384,
        size_mb=80,
        quality="good",
        speed="very_fast",
        description="Lightweight general-purpose model. Best for quick iteration.",
    ),
    "all-mpnet-base-v2": EmbeddingModelInfo(
        model_id="all-mpnet-base-v2",
        name="MPNet Base v2",
        dimensions=768,
        size_mb=420,
        quality="better",
        speed="fast",
        description="Higher quality than MiniLM with moderate size increase.",
    ),
    "BAAI/bge-small-en-v1.5": EmbeddingModelInfo(
        model_id="BAAI/bge-small-en-v1.5",
        name="BGE Small v1.5",
        dimensions=384,
        size_mb=130,
        quality="good",
        speed="very_fast",
        description="Beijing Academy of AI compact model. Strong for its size.",
    ),
    "BAAI/bge-large-en-v1.5": EmbeddingModelInfo(
        model_id="BAAI/bge-large-en-v1.5",
        name="BGE Large v1.5",
        dimensions=1024,
        size_mb=1300,
        quality="excellent",
        speed="medium",
        description="Top-tier open-source embedding model. Best quality for English.",
    ),
    "nomic-ai/nomic-embed-text-v1.5": EmbeddingModelInfo(
        model_id="nomic-ai/nomic-embed-text-v1.5",
        name="Nomic Embed Text v1.5",
        dimensions=768,
        size_mb=550,
        quality="excellent",
        speed="fast",
        description="Nomic AI's latest. Excellent quality with Matryoshka dimension support.",
    ),
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": EmbeddingModelInfo(
        model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        name="Qwen2 GTE 1.5B",
        dimensions=1536,
        size_mb=3000,
        quality="excellent",
        speed="medium",
        description=(
            "Alibaba's Qwen-based embedding. Highest dimensions, "
            "best for nuanced financial text."
        ),
    ),
    "intfloat/e5-large-v2": EmbeddingModelInfo(
        model_id="intfloat/e5-large-v2",
        name="E5 Large v2",
        dimensions=1024,
        size_mb=1300,
        quality="excellent",
        speed="medium",
        description="Microsoft's E5 model. Strong retrieval performance.",
    ),
    "intfloat/e5-small-v2": EmbeddingModelInfo(
        model_id="intfloat/e5-small-v2",
        name="E5 Small v2",
        dimensions=384,
        size_mb=130,
        quality="good",
        speed="very_fast",
        description="Compact E5 variant. Good balance for resource-constrained setups.",
    ),
}


def get_model_info(model_id: str) -> EmbeddingModelInfo | None:
    """Return metadata for a supported embedding model, or None if unknown."""
    return EMBEDDING_MODELS.get(model_id)


def list_models() -> list[EmbeddingModelInfo]:
    """Return all supported embedding models."""
    return list(EMBEDDING_MODELS.values())
