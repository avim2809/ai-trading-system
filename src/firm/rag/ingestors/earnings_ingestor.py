"""Earnings call transcript ingestor via FMP API."""

from __future__ import annotations

import os
from typing import Any

import requests

from firm.rag.chunker import DocumentChunker
from firm.rag.dates import normalize_date
from firm.rag.ingestors.base_ingestor import BaseIngestor
from firm.rag.models import Document
from firm.rag.store import VectorStore

COLLECTION = "earnings"
_FMP_BASE = "https://financialmodelingprep.com"


class EarningsIngestor(BaseIngestor):
    """Ingests earnings call transcripts from FMP."""

    def __init__(self, store: VectorStore, chunker: DocumentChunker) -> None:
        super().__init__(store, chunker)
        self._api_key = os.environ.get("FMP_API_KEY", "")

    def ingest(
        self,
        symbols: list[str] | None = None,
        start_year: int = 2020,
        end_year: int = 2024,
        **kwargs: Any,
    ) -> int:
        symbols = symbols or []
        total = 0

        for symbol in symbols:
            docs = self._fetch_transcripts(symbol, start_year, end_year)
            if docs:
                added = self.store.add_documents(COLLECTION, docs)
                total += added

        return total

    def _fetch_transcripts(
        self, symbol: str, start_year: int, end_year: int
    ) -> list[Document]:
        all_docs: list[Document] = []

        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                text = self._get_transcript(symbol, year, quarter)
                if not text:
                    continue

                metadata = {
                    "source": "fmp_earnings",
                    "symbol": symbol,
                    "doc_type": "earnings_transcript",
                    # Fiscal-period label normalized to a conservative
                    # availability date (quarter end + reporting lag) so it
                    # cannot be retrieved before results were public.
                    "date": normalize_date(f"{year}-Q{quarter}"),
                    "fiscal_period": f"{year}-Q{quarter}",
                    "year": year,
                    "quarter": quarter,
                }
                chunks = self.chunker.chunk(text, metadata)
                all_docs.extend(chunks)

        return all_docs

    def _get_transcript(self, symbol: str, year: int, quarter: int) -> str:
        """Fetch a single earnings call transcript from FMP."""
        if not self._api_key:
            return ""
        try:
            url = f"{_FMP_BASE}/api/v3/earning_call_transcript/{symbol}"
            params = {"quarter": quarter, "year": year, "apikey": self._api_key}
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                return ""
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("content", "")
            return ""
        except Exception:
            return ""
