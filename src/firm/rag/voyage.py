"""Voyage AI embeddings + reranking — external (hosted) RAG models.

Lets the RAG stack run with no local torch / sentence-transformers, which keeps
the deployed image small and the RAM footprint low. Embeddings default to
Voyage's finance-tuned model; reranking uses Voyage ``rerank-2.5``.

Requires ``VOYAGE_API_KEY`` and the ``voyageai`` package (``firm[llm]`` extra).
Both classes are lazy: they don't touch the network (or require the key) until
the first embed/rerank call, so construction never fails just because the key
is absent — callers degrade gracefully instead.
"""

from __future__ import annotations

import os

_DEFAULT_EMBED_MODEL = "voyage-finance-2"
_DEFAULT_RERANK_MODEL = "rerank-2.5"
# Voyage allows up to 1,000 texts per embed request, but also caps total tokens
# per request (120K for voyage-finance-2) and per text (32K context). With the
# corpus chunked to ~500 tokens, a 128-text batch (~64K) stays well under both,
# so we batch by count rather than pulling in Voyage's tokenizer. If large,
# un-chunked documents are ever embedded, switch to token-aware batching via
# voyageai.Client.count_tokens() — see docs.voyageai.com/docs/tokenization.
_EMBED_BATCH = 128


def _client():
    """Return a configured Voyage client, or raise a clear error if unconfigured."""
    try:
        import voyageai
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "voyageai is required for the Voyage RAG provider. "
            "Install with: pip install 'firm[llm]'"
        ) from exc
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set — required for the Voyage embedding/rerank "
            "provider. Set it, or switch rag.embedding_provider/reranker_provider "
            "to 'local' in config/llm.yaml."
        )
    return voyageai.Client(api_key=api_key)


class VoyageEmbeddingFunction:
    """ChromaDB-compatible embedding function backed by the Voyage API."""

    def __init__(self, model: str = _DEFAULT_EMBED_MODEL) -> None:
        self._model = model or _DEFAULT_EMBED_MODEL
        self._client = None  # lazy — don't require the key until first use

    def __call__(self, input: list[str]) -> list[list[float]]:
        if self._client is None:
            self._client = _client()
        out: list[list[float]] = []
        for i in range(0, len(input), _EMBED_BATCH):
            batch = input[i : i + _EMBED_BATCH]
            # truncation=True (Voyage's default, made explicit): an over-long text
            # is truncated to the model's context limit rather than erroring.
            out.extend(
                self._client.embed(batch, model=self._model, truncation=True).embeddings
            )
        return out

    @staticmethod
    def name() -> str:
        """Stable identifier so Chroma can persist the collection's EF config."""
        return "voyage"


class VoyageReranker:
    """Cross-encoder-style reranker via the Voyage rerank API.

    Exposes ``predict(pairs)`` to mirror sentence-transformers' ``CrossEncoder``
    so it drops straight into ``RAGRetriever._rerank`` with no caller changes.
    """

    def __init__(self, model: str = _DEFAULT_RERANK_MODEL) -> None:
        self._model = model or _DEFAULT_RERANK_MODEL
        self._client = None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, doc) pairs; returns scores in the input order.

        The retriever always passes pairs that share one query, so this issues a
        single rerank request and maps the ranked results back by index.
        """
        if not pairs:
            return []
        if self._client is None:
            self._client = _client()
        query = pairs[0][0]
        documents = [doc for _, doc in pairs]
        result = self._client.rerank(query, documents, model=self._model)
        scores = [0.0] * len(documents)
        for item in result.results:
            scores[item.index] = float(item.relevance_score)
        return scores
