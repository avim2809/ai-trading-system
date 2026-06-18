"""Financial news ingestor via RSS feeds and data providers."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any

import requests

from firm.rag.chunker import DocumentChunker
from firm.rag.dates import normalize_date
from firm.rag.ingestors.base_ingestor import BaseIngestor
from firm.rag.models import Document
from firm.rag.store import VectorStore

COLLECTION = "news"

_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
_TIINGO_NEWS = "https://api.tiingo.com/tiingo/news"
_AV_NEWS = "https://www.alphavantage.co/query"


class NewsIngestor(BaseIngestor):
    """Ingests financial news from RSS feeds and provider APIs."""

    def __init__(self, store: VectorStore, chunker: DocumentChunker) -> None:
        super().__init__(store, chunker)
        self._tiingo_key = os.environ.get("TIINGO_API_KEY", "")
        self._av_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")

    def ingest(
        self,
        symbols: list[str] | None = None,
        days: int = 30,
        **kwargs: Any,
    ) -> int:
        symbols = symbols or []
        total = 0

        for symbol in symbols:
            docs: list[Document] = []

            rss_docs = self._fetch_yahoo_rss(symbol)
            docs.extend(rss_docs)

            if self._tiingo_key:
                tiingo_docs = self._fetch_tiingo_news(symbol, days)
                docs.extend(tiingo_docs)

            if self._av_key:
                av_docs = self._fetch_alphavantage_news(symbol)
                docs.extend(av_docs)

            if docs:
                added = self.store.add_documents(COLLECTION, docs)
                total += added

        return total

    def _fetch_yahoo_rss(self, symbol: str) -> list[Document]:
        """Parse Yahoo Finance RSS feed for a symbol."""
        try:
            url = _YAHOO_RSS.format(symbol=symbol)
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                return []
            return self._parse_rss(resp.text, symbol, "yahoo_finance")
        except Exception:
            return []

    def _parse_rss(self, xml_text: str, symbol: str, source: str) -> list[Document]:
        """Parse RSS XML into Document chunks."""
        docs: list[Document] = []
        try:
            root = ET.fromstring(xml_text)
            for item in root.iter("item"):
                title = item.findtext("title", "")
                description = item.findtext("description", "")
                pub_date = item.findtext("pubDate", "")
                link = item.findtext("link", "")

                text = f"{title}. {description}".strip()
                if len(text) < 20:
                    continue

                metadata = {
                    "source": source,
                    "symbol": symbol,
                    "doc_type": "news",
                    "date": normalize_date(pub_date),
                    "url": link,
                }
                chunks = self.chunker.chunk(text, metadata)
                docs.extend(chunks)
        except ET.ParseError:
            pass
        return docs

    def _fetch_tiingo_news(self, symbol: str, days: int) -> list[Document]:
        """Fetch news from Tiingo API."""
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            params = {
                "tickers": symbol,
                "startDate": start_date,
                "token": self._tiingo_key,
            }
            resp = requests.get(_TIINGO_NEWS, params=params, timeout=15)
            if resp.status_code != 200:
                return []

            docs: list[Document] = []
            for article in resp.json()[:50]:
                title = article.get("title", "")
                desc = article.get("description", "")
                text = f"{title}. {desc}".strip()
                if len(text) < 20:
                    continue

                metadata = {
                    "source": "tiingo",
                    "symbol": symbol,
                    "doc_type": "news",
                    "date": normalize_date(article.get("publishedDate", "")),
                    "url": article.get("url", ""),
                }
                chunks = self.chunker.chunk(text, metadata)
                docs.extend(chunks)
            return docs
        except Exception:
            return []

    def _fetch_alphavantage_news(self, symbol: str) -> list[Document]:
        """Fetch news from Alpha Vantage."""
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self._av_key,
                "limit": 50,
            }
            resp = requests.get(_AV_NEWS, params=params, timeout=15)
            if resp.status_code != 200:
                return []

            data = resp.json()
            feed = data.get("feed", [])

            docs: list[Document] = []
            for article in feed:
                title = article.get("title", "")
                summary = article.get("summary", "")
                text = f"{title}. {summary}".strip()
                if len(text) < 20:
                    continue

                metadata = {
                    "source": "alphavantage",
                    "symbol": symbol,
                    "doc_type": "news",
                    "date": normalize_date(article.get("time_published", "")),
                    "url": article.get("url", ""),
                }
                chunks = self.chunker.chunk(text, metadata)
                docs.extend(chunks)
            return docs
        except Exception:
            return []
