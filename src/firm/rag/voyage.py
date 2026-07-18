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
# per request (120K for voyage-finance-2) and per text (32K context). Chunk
# *count* alone isn't a safe batching signal — DocumentChunker's ~500-token
# target is a soft heuristic (char-count/4, sentence-boundary splitting), and
# real-world text (SEC filing HTML-to-text extraction especially) can produce
# chunks several times that size, blowing a 128-text batch well past 120K
# tokens. So batching is token-aware: accumulate texts until either the count
# or the token budget would be exceeded, using Voyage's own (local, no extra
# API call) tokenizer via ``Client.tokenize`` to measure each text.
_EMBED_BATCH = 128
_MAX_TOKENS_PER_BATCH = 100_000  # safety margin under the 120K/request cap


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
        return self._embed(input, input_type="document")

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed a query, called by Chroma separately from ``__call__``.

        Voyage supports asymmetric embedding via ``input_type`` — "document"
        for corpus text, "query" for search text — which measurably improves
        retrieval relevance for short queries against longer documents.
        Without this method Chroma's newer client (which distinguishes
        query-time from document-time embedding) raises an ``AttributeError``
        rather than falling back to ``__call__``.
        """
        return self._embed(input, input_type="query")

    def _embed(self, input: list[str], input_type: str) -> list[list[float]]:
        if self._client is None:
            self._client = _client()
        if not input:
            return []

        # One local tokenize call for the whole input, not one per batch —
        # Client.tokenize doesn't hit the network (no auth needed beyond
        # client construction), so this is cheap even for large corpora.
        token_counts = [len(t) for t in self._client.tokenize(input, model=self._model)]

        out: list[list[float]] = []
        batch: list[str] = []
        batch_tokens = 0
        for text, n_tokens in zip(input, token_counts):
            if batch and (
                len(batch) >= _EMBED_BATCH
                or batch_tokens + n_tokens > _MAX_TOKENS_PER_BATCH
            ):
                out.extend(
                    self._client.embed(
                        batch, model=self._model, input_type=input_type, truncation=True
                    ).embeddings
                )
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += n_tokens

        if batch:
            # truncation=True (Voyage's default, made explicit): an over-long text
            # is truncated to the model's context limit rather than erroring.
            out.extend(
                self._client.embed(
                    batch, model=self._model, input_type=input_type, truncation=True
                ).embeddings
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
