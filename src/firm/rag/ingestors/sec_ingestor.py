"""SEC EDGAR filing ingestor.

Uses the free SEC EDGAR full-text search API to find and ingest
10-K, 10-Q, and 8-K filings.
"""

from __future__ import annotations

import re
import time
from html.parser import HTMLParser
from typing import Any

import requests

from firm.rag.chunker import DocumentChunker
from firm.rag.dates import normalize_date
from firm.rag.ingestors.base_ingestor import BaseIngestor
from firm.rag.models import Document
from firm.rag.store import VectorStore

COLLECTION = "sec_filings"
_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_SECTION_HEADERS = [
    "Item 1", "Item 1A", "Item 1B", "Item 2", "Item 3", "Item 4",
    "Item 5", "Item 6", "Item 7", "Item 7A", "Item 8", "Item 9",
    "Item 9A", "Item 9B", "Item 10", "Item 11", "Item 12", "Item 13",
    "Item 14", "Item 15",
]
_HEADERS = {"User-Agent": "FirmBot/1.0 research@example.com"}


class _HTMLTextExtractor(HTMLParser):
    """Simple HTML to plain-text converter."""

    def __init__(self):
        super().__init__()
        self._text: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._text.append(data)

    def get_text(self) -> str:
        return " ".join(self._text)


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    text = parser.get_text()
    return re.sub(r'\s+', ' ', text).strip()


class SECIngestor(BaseIngestor):
    """Ingests SEC EDGAR filings for given symbols."""

    def __init__(self, store: VectorStore, chunker: DocumentChunker) -> None:
        super().__init__(store, chunker)

    def ingest(
        self,
        symbols: list[str] | None = None,
        start_year: int = 2020,
        end_year: int = 2024,
        forms: str = "10-K,10-Q,8-K",
        max_per_symbol: int = 10,
        **kwargs: Any,
    ) -> int:
        symbols = symbols or []
        total = 0

        for symbol in symbols:
            docs = self._fetch_filings(symbol, start_year, end_year, forms, max_per_symbol)
            if docs:
                added = self.store.add_documents(COLLECTION, docs)
                total += added

        return total

    def _fetch_filings(
        self,
        symbol: str,
        start_year: int,
        end_year: int,
        forms: str,
        max_results: int,
    ) -> list[Document]:
        try:
            params = {
                "q": symbol,
                "dateRange": "custom",
                "startdt": f"{start_year}-01-01",
                "enddt": f"{end_year}-12-31",
                "forms": forms,
            }
            resp = requests.get(_SEARCH_URL, params=params, headers=_HEADERS, timeout=30)
            if resp.status_code != 200:
                return []

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])[:max_results]
        except Exception:
            return []

        all_docs: list[Document] = []
        for hit in hits:
            source = hit.get("_source", {})
            filing_url = source.get("file_url", "")
            form_type = source.get("form_type", "")
            filed_date = source.get("file_date", "")

            if not filing_url:
                continue

            text = self._fetch_filing_text(filing_url)
            if not text or len(text) < 100:
                continue

            metadata = {
                "source": "sec_edgar",
                "symbol": symbol,
                "doc_type": form_type,
                "date": normalize_date(filed_date),
                "url": filing_url,
            }

            chunks = self.chunker.chunk_by_sections(
                text, _SECTION_HEADERS, metadata
            )
            if not chunks:
                chunks = self.chunker.chunk(text, metadata)

            all_docs.extend(chunks)
            time.sleep(0.2)  # Rate-limit courtesy

        return all_docs

    def _fetch_filing_text(self, url: str) -> str:
        try:
            if not url.startswith("http"):
                url = f"https://www.sec.gov/Archives/{url}"
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            if resp.status_code != 200:
                return ""
            return _html_to_text(resp.text)[:200_000]  # Cap at ~200k chars
        except Exception:
            return ""
