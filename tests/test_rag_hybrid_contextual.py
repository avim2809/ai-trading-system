"""Phase 3: contextual-embedding chunking and hybrid (BM25+dense) retrieval."""

from __future__ import annotations

from datetime import datetime

from firm.rag.chunker import DocumentChunker
from firm.rag.models import RetrievedDoc
from firm.rag.retriever import RAGRetriever


# ── Phase 3a: contextual embeddings ─────────────────────────────────

class TestContextualChunker:
    def test_default_off_no_header(self):
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        docs = chunker.chunk("A short sentence about Apple.", {"symbol": "AAPL"})
        assert docs
        assert not docs[0].text.startswith("[")

    def test_contextual_prepends_metadata_header(self):
        chunker = DocumentChunker(contextual=True)
        meta = {"doc_type": "run_note", "symbol": "AAPL", "date": "2021-06-30"}
        docs = chunker.chunk("Some narrative text here.", meta)
        assert docs[0].text.startswith("[")
        assert "symbol=AAPL" in docs[0].text
        assert "date=2021-06-30" in docs[0].text
        # Original content is preserved after the header.
        assert "Some narrative text here." in docs[0].text

    def test_contextual_changes_doc_id(self):
        meta = {"symbol": "AAPL", "source": "x"}
        plain = DocumentChunker().chunk("Same text.", meta)[0]
        ctx = DocumentChunker(contextual=True).chunk("Same text.", meta)[0]
        assert plain.doc_id != ctx.doc_id

    def test_contextual_no_metadata_is_noop(self):
        docs = DocumentChunker(contextual=True).chunk("Plain text.", {})
        assert not docs[0].text.startswith("[")


# ── Phase 3b: hybrid retrieval ──────────────────────────────────────

class _FakeHybridStore:
    """Dense channel deliberately misses the rare-keyword doc; BM25 finds it."""

    def __init__(self, docs):
        self._docs = docs

    def list_collections(self):
        return ["c"]

    def get_all(self, name):
        return [RetrievedDoc(d.doc_id, d.text, dict(d.metadata)) for d in self._docs]

    def query(self, collection_name, query_text, n_results=5,
              where_filters=None, asof=None):
        # Simulate a semantic retriever that returns only the "alpha" doc.
        hits = [d for d in self._docs if "alpha" in d.text.lower()]
        return [RetrievedDoc(d.doc_id, d.text, dict(d.metadata), 0.9) for d in hits][:n_results]


def _corpus():
    return [
        RetrievedDoc("d1", "The alpha strategy performed well this period.", {}),
        RetrievedDoc("d2", "Momentum returns were strong overall.", {}),
        RetrievedDoc("d3", "Ticker ZZZZ had a unique idiosyncratic event.", {}),
    ]


class TestHybridRetrieval:
    def test_bm25_recovers_exact_keyword_dense_missed(self):
        store = _FakeHybridStore(_corpus())
        hybrid = RAGRetriever(store, reranker=False, hybrid=True)
        ids = {d.doc_id for d in hybrid.retrieve("ZZZZ ticker event", collection="c")}
        assert "d3" in ids  # lexical channel surfaced it

    def test_dense_only_misses_exact_keyword(self):
        store = _FakeHybridStore(_corpus())
        dense = RAGRetriever(store, reranker=False, hybrid=False)
        ids = {d.doc_id for d in dense.retrieve("ZZZZ ticker event", collection="c")}
        assert "d3" not in ids

    def test_rrf_fuses_and_dedupes(self):
        dense = [RetrievedDoc("a", "x", {}, 0.9), RetrievedDoc("b", "y", {}, 0.8)]
        lexical = [RetrievedDoc("b", "y", {}, 5.0), RetrievedDoc("c", "z", {}, 4.0)]
        fused = RAGRetriever._rrf(dense, lexical)
        ids = [d.doc_id for d in fused]
        assert set(ids) == {"a", "b", "c"}
        # 'b' appears in both lists → highest fused score → ranked first.
        assert ids[0] == "b"

    def test_passes_filters_excludes_future_dates(self):
        meta = {"date": "2025-01-01"}
        asof = datetime(2023, 6, 1)
        assert RAGRetriever._passes_filters(meta, None, None, asof) is False
        assert RAGRetriever._passes_filters({"date": "2023-01-01"}, None, None, asof) is True

    def test_passes_filters_symbol_and_type(self):
        meta = {"symbol": "AAPL", "doc_type": "news"}
        assert RAGRetriever._passes_filters(meta, ["AAPL"], ["news"], None) is True
        assert RAGRetriever._passes_filters(meta, ["MSFT"], None, None) is False
        assert RAGRetriever._passes_filters(meta, None, ["sec"], None) is False
