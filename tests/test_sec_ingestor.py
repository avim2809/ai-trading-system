"""SEC filing ingestor: high-value section filtering.

No real network calls — requests.get is monkeypatched with canned search-hit
and filing-HTML responses so these run offline/fast.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from firm.rag.chunker import DocumentChunker
from firm.rag.ingestors.sec_ingestor import SECIngestor

_TENK_BODY = (
    "<html><body>"
    "Item 1 Business overview text describing the company. "
    "Item 1A Risk factors: supply chain and competition risks. "
    "Item 3 Legal proceedings: ongoing litigation matters. "
    "Item 7 Management discussion and analysis of results. "
    "Item 7A Quantitative disclosures about market risk. "
    "Item 8 Financial statements: balance sheet and income statement tables. "
    "Item 10 Directors and executive officers biographical information. "
    "Item 15 Exhibit index and financial statement schedules."
    "</body></html>"
)

_EIGHTK_BODY = (
    "<html><body>"
    "Item 2.02 Results of Operations and Financial Condition: "
    "the company announced quarterly earnings results today. "
    "Item 9.01 Financial Statements and Exhibits: see attached press release."
    "</body></html>"
)


def _search_response(form: str):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "0000320193-24-000005:aapl-20240101.htm",
                    "_source": {
                        "ciks": ["0000320193"],
                        "form": form,
                        "file_date": "2024-01-01",
                    },
                }
            ]
        }
    }
    return resp


def _filing_response(body: str):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = body
    return resp


class TestSECHighValueSectionFilter:
    def test_10k_keeps_only_high_value_sections(self):
        ingestor = SECIngestor(store=None, chunker=DocumentChunker(chunk_size=500))

        with patch("firm.rag.ingestors.sec_ingestor.requests.get") as mock_get:
            mock_get.side_effect = [
                _search_response("10-K"),
                _filing_response(_TENK_BODY),
            ]
            docs = ingestor._fetch_filings("AAPL", 2024, 2024, "10-K", max_results=1)

        sections = {d.metadata["section"] for d in docs}
        assert sections == {"Item 1", "Item 1A", "Item 3", "Item 7", "Item 7A"}
        # Item 8 (financial tables) and Item 10/15 (governance/exhibits)
        # must be excluded.
        joined = " ".join(d.text for d in docs)
        assert "balance sheet" not in joined
        assert "executive officers" not in joined
        assert "Exhibit index" not in joined

    def test_8k_is_kept_whole_not_filtered(self):
        ingestor = SECIngestor(store=None, chunker=DocumentChunker(chunk_size=500))

        with patch("firm.rag.ingestors.sec_ingestor.requests.get") as mock_get:
            mock_get.side_effect = [
                _search_response("8-K"),
                _filing_response(_EIGHTK_BODY),
            ]
            docs = ingestor._fetch_filings("AAPL", 2024, 2024, "8-K", max_results=1)

        assert docs, "8-K content must not be filtered away to nothing"
        joined = " ".join(d.text for d in docs)
        assert "quarterly earnings results" in joined
        assert "press release" in joined
