"""Academic paper ingestor from arXiv quantitative finance."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

import requests

from firm.rag.chunker import DocumentChunker
from firm.rag.dates import normalize_date
from firm.rag.ingestors.base_ingestor import BaseIngestor
from firm.rag.models import Document
from firm.rag.store import VectorStore

log = logging.getLogger("firm.rag.ingestors.research")

COLLECTION = "research"
_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ResearchIngestor(BaseIngestor):
    """Ingests academic papers from arXiv quantitative finance category."""

    def __init__(self, store: VectorStore, chunker: DocumentChunker) -> None:
        super().__init__(store, chunker)

    def ingest(
        self,
        topics: list[str] | None = None,
        max_results: int = 100,
        **kwargs: Any,
    ) -> int:
        if topics:
            query = " OR ".join(f"all:{t}" for t in topics)
            search_query = f"cat:q-fin.* AND ({query})"
        else:
            search_query = "cat:q-fin.*"

        docs = self._fetch_arxiv(search_query, max_results)
        if docs:
            return self.store.add_documents(COLLECTION, docs)
        return 0

    def _fetch_arxiv(self, search_query: str, max_results: int) -> list[Document]:
        """Fetch papers from arXiv API and extract title + abstract."""
        try:
            params = {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            resp = requests.get(_ARXIV_API, params=params, timeout=30)
            if resp.status_code != 200:
                log.warning(
                    "arxiv_search_failed query=%r status=%d", search_query, resp.status_code
                )
                return []

            return self._parse_atom(resp.text)
        except Exception:
            log.warning("arxiv_search_error query=%r", search_query, exc_info=True)
            return []

    def _parse_atom(self, xml_text: str) -> list[Document]:
        """Parse arXiv Atom XML feed into documents."""
        docs: list[Document] = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", _ATOM_NS):
                title = (entry.findtext("atom:title", "", _ATOM_NS) or "").strip()
                abstract = (entry.findtext("atom:summary", "", _ATOM_NS) or "").strip()
                published = (entry.findtext("atom:published", "", _ATOM_NS) or "").strip()
                arxiv_id = (entry.findtext("atom:id", "", _ATOM_NS) or "").strip()

                authors = [
                    a.findtext("atom:name", "", _ATOM_NS)
                    for a in entry.findall("atom:author", _ATOM_NS)
                ]

                text = f"Title: {title}\nAuthors: {', '.join(authors)}\nAbstract: {abstract}"
                if len(text) < 50:
                    continue

                metadata = {
                    "source": "arxiv",
                    "doc_type": "research_paper",
                    "date": normalize_date(published),
                    "url": arxiv_id,
                    "title": title,
                    "authors": ", ".join(authors[:5]),
                }

                chunks = self.chunker.chunk(text, metadata)
                docs.extend(chunks)
        except ET.ParseError:
            log.warning("arxiv_parse_failed", exc_info=True)
        return docs
