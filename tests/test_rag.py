"""RAG point-in-time safety and date-normalization tests.

These guard the look-ahead fix: the retriever must thread an ``asof`` filter
down to the vector store so future-dated documents can never be retrieved
into a past decision.
"""

from __future__ import annotations

from datetime import datetime

from firm.rag.dates import (
    ALWAYS_AVAILABLE_DATE,
    UNKNOWN_DATE,
    normalize_date,
)
from firm.rag.models import RetrievedDoc
from firm.rag.retriever import RAGRetriever


class _FakeStore:
    """Records the kwargs the retriever passes to ``query``."""

    def __init__(self, collections):
        self._collections = collections
        self.calls: list[dict] = []

    def list_collections(self):
        return list(self._collections)

    def query(self, collection_name, query_text, n_results=5,
              where_filters=None, asof=None):
        self.calls.append({
            "collection": collection_name,
            "where": where_filters,
            "asof": asof,
        })
        return [RetrievedDoc(doc_id=f"{collection_name}-1", text="t", score=0.9)]


class TestDateNormalization:
    def test_iso_date_passthrough(self):
        assert normalize_date("2023-01-02") == "2023-01-02"

    def test_iso_datetime_with_z(self):
        assert normalize_date("2023-01-02T14:30:00Z") == "2023-01-02"

    def test_rfc822_rss_pubdate(self):
        assert normalize_date("Mon, 02 Jan 2023 14:30:00 GMT") == "2023-01-02"

    def test_alphavantage_compact(self):
        assert normalize_date("20230102T143000") == "2023-01-02"

    def test_fiscal_quarter_maps_to_availability_after_quarter_end(self):
        # Q1 2023 ends 2023-03-31; availability must be strictly after it.
        result = normalize_date("2023-Q1")
        assert result > "2023-03-31"

    def test_empty_is_unknown_far_future(self):
        assert normalize_date("") == UNKNOWN_DATE
        assert normalize_date(None) == UNKNOWN_DATE

    def test_datetime_object(self):
        assert normalize_date(datetime(2022, 7, 4, 9, 0)) == "2022-07-04"


class TestRetrieverAsOf:
    def test_asof_is_passed_to_store(self):
        store = _FakeStore(["news"])
        retriever = RAGRetriever(store, reranker=False)
        asof = datetime(2023, 6, 1)
        retriever.retrieve("q", symbols=["AAPL"], collections=["news"], asof=asof)
        assert store.calls, "store.query was not called"
        assert all(c["asof"] == asof for c in store.calls)

    def test_collections_restrict_search(self):
        store = _FakeStore(["news", "sec_filings", "research"])
        retriever = RAGRetriever(store, reranker=False)
        retriever.retrieve("q", collections=["news", "research"])
        queried = {c["collection"] for c in store.calls}
        assert queried == {"news", "research"}
        assert "sec_filings" not in queried

    def test_retrieve_for_symbol_threads_asof(self):
        store = _FakeStore(["news"])
        retriever = RAGRetriever(store, reranker=False)
        asof = datetime(2024, 1, 1)
        retriever.retrieve_for_symbol("AAPL", "q", collections=["news"], asof=asof)
        assert store.calls[0]["asof"] == asof


class _FakeChromaCollection:
    """Mimics the subset of Chroma's Collection API that VectorStore.query uses."""

    def __init__(self, ids, documents, metadatas, distances):
        self._ids = ids
        self._documents = documents
        self._metadatas = metadatas
        self._distances = distances

    def count(self):
        return len(self._ids)

    def query(self, query_texts, n_results, where=None):
        n = min(n_results, len(self._ids))
        return {
            "ids": [self._ids[:n]],
            "documents": [self._documents[:n]],
            "metadatas": [self._metadatas[:n]],
            "distances": [self._distances[:n]],
        }


class _FakeChromaStore:
    """Exposes just enough of VectorStore's API for the real ``query`` method
    to run unmodified against a fake collection instead of a live Chroma client."""

    def __init__(self, collection):
        self._collection = collection

    def get_or_create_collection(self, name):
        return self._collection


class TestStoreDateFilter:
    def test_asof_excludes_future_and_undated_docs(self):
        # Regression test: Chroma's $lte/$gte only accept numeric operands, so
        # pushing the ISO date *string* into a `where` clause raises a
        # ValueError at query time — silently swallowed by the caller's broad
        # except, which made every point-in-time-filtered retrieval return
        # nothing. VectorStore.query must filter by date in Python instead.
        from firm.rag.store import VectorStore

        collection = _FakeChromaCollection(
            ids=["past", "future", "no-date"],
            documents=["past doc", "future doc", "undated doc"],
            metadatas=[{"date": "2023-01-01"}, {"date": "2099-01-01"}, {}],
            distances=[0.1, 0.1, 0.1],
        )
        store = _FakeChromaStore(collection)
        asof = datetime(2023, 6, 1)

        results = VectorStore.query(store, "news", "q", n_results=5, asof=asof)

        ids = {d.doc_id for d in results}
        assert ids == {"past"}, "future-dated and undated docs must be excluded"

    def test_asof_excludes_explicit_none_date_without_crashing(self):
        """A doc whose metadata explicitly carries ``"date": None`` (a
        malformed/legacy record that bypassed normalize_date) must be
        excluded like any other unknown-vintage doc, not crash the whole
        query. ``metadata.get("date", UNKNOWN_DATE)`` only substitutes the
        sentinel when the key is *missing* — an explicit ``None`` value
        used to slip through and raise ``TypeError`` on the ``>`` compare."""
        from firm.rag.store import VectorStore

        collection = _FakeChromaCollection(
            ids=["past", "none-date"],
            documents=["past doc", "malformed doc"],
            metadatas=[{"date": "2023-01-01"}, {"date": None}],
            distances=[0.1, 0.1],
        )
        store = _FakeChromaStore(collection)
        asof = datetime(2023, 6, 1)

        results = VectorStore.query(store, "news", "q", n_results=5, asof=asof)

        ids = {d.doc_id for d in results}
        assert ids == {"past"}, "explicit-None-date doc must be excluded, not crash"

    def test_no_asof_returns_everything(self):
        from firm.rag.store import VectorStore

        collection = _FakeChromaCollection(
            ids=["a", "b"], documents=["x", "y"],
            metadatas=[{"date": "2023-01-01"}, {"date": "2099-01-01"}],
            distances=[0.1, 0.2],
        )
        store = _FakeChromaStore(collection)

        results = VectorStore.query(store, "news", "q", n_results=5, asof=None)

        assert {d.doc_id for d in results} == {"a", "b"}

    def test_system_sentinel_passes_any_asof(self):
        # Timeless system docs use a min-date so a $lte asof always admits them.
        assert ALWAYS_AVAILABLE_DATE <= "2020-01-01"
        # Unknown docs use a far-future date so a $lte asof always excludes them.
        assert UNKNOWN_DATE > "2099-01-01"


class _FakeMetadataCollection:
    """Mimics just the ``get``/``update`` subset of Chroma's Collection API
    that ``VectorStore.update_metadata`` uses."""

    def __init__(self, ids, metadatas):
        self._ids = ids
        self._metadatas = metadatas
        self.update_calls: list[dict] = []

    def get(self, ids, include=None):
        found = [(i, m) for i, m in zip(self._ids, self._metadatas) if i in ids]
        return {
            "ids": [i for i, _ in found],
            "metadatas": [m for _, m in found],
        }

    def update(self, ids, metadatas):
        self.update_calls.append({"ids": ids, "metadatas": metadatas})


class TestVectorStoreUpdateMetadata:
    """update_metadata's job: patch an existing doc's metadata in place,
    merged onto what's already stored, without touching embeddings/text.

    Distinct from add_documents' dedup-by-id, which would silently no-op a
    metadata-only change to an id that's already in the collection (see
    firm.agents.memory's recommendation-applied loop, the caller this
    method was added for)."""

    def test_merges_new_metadata_onto_existing_and_updates_in_place(self):
        from firm.rag.store import VectorStore

        collection = _FakeMetadataCollection(
            ids=["recommendation:2026-01-01"],
            metadatas=[{"date": "2026-01-01", "action": "flag_strategy_for_review", "applied": False}],
        )
        store = _FakeChromaStore(collection)

        result = VectorStore.update_metadata(
            store, "recommendations", "recommendation:2026-01-01", {"applied": True},
        )

        assert result is True
        assert len(collection.update_calls) == 1
        call = collection.update_calls[0]
        assert call["ids"] == ["recommendation:2026-01-01"]
        # Unrelated existing fields survive; only "applied" changed.
        assert call["metadatas"] == [
            {"date": "2026-01-01", "action": "flag_strategy_for_review", "applied": True},
        ]

    def test_returns_false_and_does_not_update_when_doc_id_missing(self):
        from firm.rag.store import VectorStore

        collection = _FakeMetadataCollection(ids=[], metadatas=[])
        store = _FakeChromaStore(collection)

        result = VectorStore.update_metadata(store, "recommendations", "missing-id", {"applied": True})

        assert result is False
        assert collection.update_calls == []
