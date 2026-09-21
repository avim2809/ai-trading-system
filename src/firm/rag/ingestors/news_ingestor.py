"""Financial news ingestor via RSS feeds and data providers."""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from firm.agents.llm.news_anonymizer import mentions_symbol
from firm.rag.chunker import DocumentChunker
from firm.rag.dates import normalize_date
from firm.rag.ingestors.base_ingestor import BaseIngestor
from firm.rag.models import Document
from firm.rag.store import VectorStore

log = logging.getLogger("firm.rag.ingestors.news")

COLLECTION = "news"

_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
_TIINGO_NEWS = "https://api.tiingo.com/tiingo/news"
_AV_NEWS = "https://www.alphavantage.co/query"
_MASSIVE_NEWS = "https://api.massive.com/v2/reference/news"
_FINNHUB_NEWS = "https://finnhub.io/api/v1/company-news"
# Massive's free tier is 5 requests/minute with (as far as documented) no
# separate daily cap — unlike AlphaVantage's hard 25/day, so it's the primary
# source now. Paced conservatively (13s > 60/5s) to stay clear of the limit
# across a full-universe ingestion pass rather than bursting and hitting 429s.
_MASSIVE_PACING_SECONDS = 13


class NewsIngestor(BaseIngestor):
    """Ingests financial news from RSS feeds and provider APIs."""

    def __init__(self, store: VectorStore, chunker: DocumentChunker) -> None:
        super().__init__(store, chunker)
        self._tiingo_key = os.environ.get("TIINGO_API_KEY", "")
        self._av_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self._massive_key = os.environ.get("MASSIVE_API_KEY", "")
        self._finnhub_key = os.environ.get("FINNHUB_API_KEY", "")

    def ingest(
        self,
        symbols: list[str] | None = None,
        days: int = 30,
        **kwargs: Any,
    ) -> int:
        symbols = symbols or []
        total = 0

        for i, symbol in enumerate(symbols):
            docs: list[Document] = []

            rss_docs = self._fetch_yahoo_rss(symbol)
            docs.extend(rss_docs)

            if self._tiingo_key:
                tiingo_docs = self._fetch_tiingo_news(symbol, days)
                docs.extend(tiingo_docs)

            massive_docs: list[Document] = []
            if self._massive_key:
                massive_docs = self._fetch_massive_news(symbol, days)
                docs.extend(massive_docs)
                if i < len(symbols) - 1:
                    time.sleep(_MASSIVE_PACING_SECONDS)

            # AlphaVantage's key is shared with price/fundamentals data
            # elsewhere in the system (firm.data.providers.fallback documents
            # Massive -> AlphaVantage -> Tiingo as the intended priority for
            # both prices and news) and capped at a hard 25 requests/day
            # total. Spend it here only when Massive didn't cover this symbol
            # at all, rather than on duplicate coverage every ingestion run —
            # otherwise RAG ingestion could exhaust the quota a live trading
            # cycle needs for its own price-fallback duty.
            if self._av_key and not massive_docs:
                docs.extend(self._fetch_alphavantage_news(symbol))

            if self._finnhub_key:
                docs.extend(self._fetch_finnhub_news(symbol, days))

            if docs:
                added = self.store.add_documents(COLLECTION, docs)
                total += added

        return total

    def _fetch_yahoo_rss(self, symbol: str) -> list[Document]:
        """Parse Yahoo Finance RSS feed for a symbol.

        Yahoo's edge blocks requests with no User-Agent, or with the
        default `requests`/`curl` UA strings (empty/no UA -> 429; curl's
        default UA -> 404 "sad panda" HTML page instead of the feed) —
        this is generic bot-signature filtering, not a sign the endpoint
        itself is gone. Any honestly-identifying custom UA (no browser
        impersonation needed) gets a normal 200 with real RSS content, so
        we send one purely to avoid being lumped in with the default
        library signatures.
        """
        try:
            url = _YAHOO_RSS.format(symbol=symbol)
            headers = {"User-Agent": "ai-trading-system-news-ingestor/1.0"}
            resp = requests.get(url, timeout=15, headers=headers)
            if resp.status_code != 200:
                log.warning(
                    "yahoo_rss_failed symbol=%s status=%d", symbol, resp.status_code
                )
                return []
            return self._parse_rss(resp.text, symbol, "yahoo_finance")
        except Exception:
            log.warning("yahoo_rss_error symbol=%s", symbol, exc_info=True)
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
                if not mentions_symbol(text, symbol):
                    # A ticker-search endpoint can return an article that
                    # never actually mentions the queried company (broad
                    # keyword/related-ticker matching on the provider's
                    # side) -- don't store it tagged with the wrong symbol.
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
            log.warning("rss_parse_failed symbol=%s source=%s", symbol, source, exc_info=True)
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
                log.warning(
                    "tiingo_news_failed symbol=%s status=%d body=%.200s",
                    symbol, resp.status_code, resp.text,
                )
                return []

            docs: list[Document] = []
            for article in resp.json()[:50]:
                title = article.get("title", "")
                desc = article.get("description", "")
                text = f"{title}. {desc}".strip()
                if len(text) < 20:
                    continue
                if not mentions_symbol(text, symbol):
                    # A ticker-search endpoint can return an article that
                    # never actually mentions the queried company (broad
                    # keyword/related-ticker matching on the provider's
                    # side) -- don't store it tagged with the wrong symbol.
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
            log.warning("tiingo_news_error symbol=%s", symbol, exc_info=True)
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
                log.warning(
                    "alphavantage_news_failed symbol=%s status=%d body=%.200s",
                    symbol, resp.status_code, resp.text,
                )
                return []

            data = resp.json()
            feed = data.get("feed", [])
            if not feed and "feed" not in data:
                # AV returns HTTP 200 even for rate-limit/quota messages (e.g.
                # {"Information": "...25 requests per day..."}) — surface it,
                # since a silent [] here looks identical to "no news today".
                log.warning(
                    "alphavantage_news_no_feed symbol=%s body=%.200s", symbol, data
                )

            docs: list[Document] = []
            for article in feed:
                title = article.get("title", "")
                summary = article.get("summary", "")
                text = f"{title}. {summary}".strip()
                if len(text) < 20:
                    continue
                if not mentions_symbol(text, symbol):
                    # A ticker-search endpoint can return an article that
                    # never actually mentions the queried company (broad
                    # keyword/related-ticker matching on the provider's
                    # side) -- don't store it tagged with the wrong symbol.
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
            log.warning("alphavantage_news_error symbol=%s", symbol, exc_info=True)
            return []

    def _fetch_massive_news(self, symbol: str, days: int) -> list[Document]:
        """Fetch news from Massive (free tier: 5 req/min, no known daily cap)."""
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            params = {
                "ticker": symbol,
                "published_utc.gte": start_date,
                "limit": 50,
                "sort": "published_utc",
                "order": "desc",
                "apiKey": self._massive_key,
            }
            resp = requests.get(_MASSIVE_NEWS, params=params, timeout=15)
            if resp.status_code != 200:
                log.warning(
                    "massive_news_failed symbol=%s status=%d body=%.200s",
                    symbol, resp.status_code, resp.text,
                )
                return []

            docs: list[Document] = []
            for article in resp.json().get("results", []):
                title = article.get("title", "")
                desc = article.get("description", "")
                text = f"{title}. {desc}".strip()
                if len(text) < 20:
                    continue
                if not mentions_symbol(text, symbol):
                    # A ticker-search endpoint can return an article that
                    # never actually mentions the queried company (broad
                    # keyword/related-ticker matching on the provider's
                    # side) -- don't store it tagged with the wrong symbol.
                    continue

                metadata = {
                    "source": "massive",
                    "symbol": symbol,
                    "doc_type": "news",
                    "date": normalize_date(article.get("published_utc", "")),
                    "url": article.get("article_url", ""),
                }
                chunks = self.chunker.chunk(text, metadata)
                docs.extend(chunks)
            return docs
        except Exception:
            log.warning("massive_news_error symbol=%s", symbol, exc_info=True)
            return []

    def _fetch_finnhub_news(self, symbol: str, days: int) -> list[Document]:
        """Fetch news from Finnhub (free tier: 60 req/min, company-news
        endpoint)."""
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            params = {"symbol": symbol, "from": start_date, "to": end_date}
            resp = requests.get(
                _FINNHUB_NEWS, params=params, timeout=15,
                headers={"X-Finnhub-Token": self._finnhub_key},
            )
            if resp.status_code != 200:
                log.warning(
                    "finnhub_news_failed symbol=%s status=%d body=%.200s",
                    symbol, resp.status_code, resp.text,
                )
                return []

            docs: list[Document] = []
            for article in resp.json()[:50]:
                headline = article.get("headline", "")
                summary = article.get("summary", "")
                text = f"{headline}. {summary}".strip()
                if len(text) < 20:
                    continue
                if not mentions_symbol(text, symbol):
                    continue

                # Finnhub's "datetime" is Unix seconds (UTC), not an
                # ISO/RFC-822 string like the other providers -- normalize_date
                # doesn't parse raw epoch ints, so convert explicitly.
                epoch = article.get("datetime")
                pub_dt = datetime.fromtimestamp(epoch, tz=timezone.utc) if epoch else None
                metadata = {
                    "source": "finnhub",
                    "symbol": symbol,
                    "doc_type": "news",
                    "date": normalize_date(pub_dt),
                    "url": article.get("url", ""),
                }
                chunks = self.chunker.chunk(text, metadata)
                docs.extend(chunks)
            return docs
        except Exception:
            log.warning("finnhub_news_error symbol=%s", symbol, exc_info=True)
            return []
