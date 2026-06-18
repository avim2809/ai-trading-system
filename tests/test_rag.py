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


class TestStoreDateFilter:
    def test_asof_adds_lte_date_clause(self):
        from firm.rag.store import _and_filters, _asof_str

        asof = datetime(2023, 6, 1, 15, 0)
        assert _asof_str(asof) == "2023-06-01"
        combined = _and_filters({"symbol": "AAPL"}, {"date": {"$lte": _asof_str(asof)}})
        assert combined == {
            "$and": [{"symbol": "AAPL"}, {"date": {"$lte": "2023-06-01"}}]
        }

    def test_and_filters_unwraps_single_clause(self):
        from firm.rag.store import _and_filters

        assert _and_filters(None, {"date": {"$lte": "x"}}) == {"date": {"$lte": "x"}}
        assert _and_filters({"symbol": "AAPL"}, None) == {"symbol": "AAPL"}

    def test_system_sentinel_passes_any_asof(self):
        # Timeless system docs use a min-date so a $lte asof always admits them.
        assert ALWAYS_AVAILABLE_DATE <= "2020-01-01"
        # Unknown docs use a far-future date so a $lte asof always excludes them.
        assert UNKNOWN_DATE > "2099-01-01"
