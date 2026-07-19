"""SEC EDGAR filing ingestor.

Uses the free SEC EDGAR full-text search API to find and ingest
10-K, 10-Q, and 8-K filings.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger("firm.rag.ingestors.sec")

COLLECTION = "sec_filings"
_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_SECTION_HEADERS = [
    "Item 1", "Item 1A", "Item 1B", "Item 2", "Item 3", "Item 4",
    "Item 5", "Item 6", "Item 7", "Item 7A", "Item 8", "Item 9",
    "Item 9A", "Item 9B", "Item 10", "Item 11", "Item 12", "Item 13",
    "Item 14", "Item 15",
]
# The analytically valuable narrative sections for sentiment/research
# grounding: Business overview, Risk Factors, Legal Proceedings, MD&A, and
# Market Risk. Deliberately excludes Item 8 (raw financial-statement tables —
# typically the single largest section by far, and structured fundamentals
# already come from FMP, not text embedding) and the governance/exhibit
# boilerplate (Items 2, 4-6, 9-15), which carry little sentiment value but
# would otherwise dominate embedding volume if the whole filing were ingested.
_HIGH_VALUE_SECTIONS = {"item 1", "item 1a", "item 3", "item 7", "item 7a"}
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
                log.warning(
                    "sec_search_failed symbol=%s status=%d body=%.200s",
                    symbol, resp.status_code, resp.text,
                )
                return []

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])[:max_results]
        except Exception:
            log.warning("sec_search_error symbol=%s", symbol, exc_info=True)
            return []

        all_docs: list[Document] = []
        for hit in hits:
            source = hit.get("_source", {})
            # The full-text-search API's hit has no "file_url"/"form_type"
            # fields (those don't exist in its response schema — every hit
            # was previously skipped as a result). The document URL must be
            # built from the top-level "_id" ("<accession>:<primary-file>")
            # plus the source's CIK, per SEC EDGAR's standard Archives layout.
            form_type = source.get("form", "")
            filed_date = source.get("file_date", "")
            ciks = source.get("ciks") or []
            hit_id = hit.get("_id", "")

            if not ciks or ":" not in hit_id:
                continue
            accession, _, filename = hit_id.partition(":")
            cik = str(int(ciks[0]))
            accession_nodash = accession.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"

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

            if form_type in ("10-K", "10-Q"):
                # The classic Item 1-15 numbering only applies to these two
                # forms. 8-K uses its own "Item 2.02"/"Item 9.01"-style event
                # codes, which happen to share literal prefixes with several
                # of _SECTION_HEADERS ("Item 2", "Item 9", ...) — matching
                # against them here would misdetect an 8-K as having real
                # Item-N structure and then filter its actual content away.
                section_chunks = self.chunker.chunk_by_sections(
                    text, _SECTION_HEADERS, metadata
                )
                if section_chunks:
                    # Keep only the narrative sections with sentiment/
                    # research value (drops Item 8 financial-statement
                    # tables and governance/exhibit boilerplate).
                    chunks = [
                        c for c in section_chunks
                        if c.metadata.get("section", "").strip().lower()
                        in _HIGH_VALUE_SECTIONS
                    ]
                else:
                    chunks = self.chunker.chunk(text, metadata)
            else:
                # 8-K (and anything else): no Item-N structure to filter
                # against, and these are event disclosures (earnings
                # releases, executive changes) that are typically short and
                # often *more* sentiment-relevant than 10-K boilerplate —
                # ingest the whole thing.
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
                log.warning("sec_filing_fetch_failed url=%s status=%d", url, resp.status_code)
                return ""
            return _html_to_text(resp.text)[:200_000]  # Cap at ~200k chars
        except Exception:
            log.warning("sec_filing_fetch_error url=%s", url, exc_info=True)
            return ""
