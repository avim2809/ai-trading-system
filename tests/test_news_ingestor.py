"""News ingestor: Massive.com source (no daily cap, unlike AlphaVantage's
25/day) plus rate-limit pacing between symbols.

No real network calls — requests.get/time.sleep are monkeypatched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from firm.rag.chunker import DocumentChunker
from firm.rag.ingestors.news_ingestor import NewsIngestor


def _massive_response(articles):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"results": articles}
    return resp


class TestMassiveNewsSource:
    def test_fetch_massive_news_produces_documents(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
        ingestor = NewsIngestor(store=None, chunker=DocumentChunker(chunk_size=500))

        articles = [
            {
                "title": "Apple beats earnings estimates",
                "description": "Apple reported strong iPhone sales this quarter.",
                "published_utc": "2026-07-18T12:00:00Z",
                "article_url": "https://example.com/aapl-earnings",
            },
        ]
        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get:
            mock_get.return_value = _massive_response(articles)
            docs = ingestor._fetch_massive_news("AAPL", days=30)

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "massive"
        assert docs[0].metadata["symbol"] == "AAPL"
        assert docs[0].metadata["date"] == "2026-07-18"
        assert "strong iPhone sales" in docs[0].text

    def test_fetch_massive_news_handles_failure_gracefully(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
        ingestor = NewsIngestor(store=None, chunker=DocumentChunker(chunk_size=500))

        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get:
            mock_get.side_effect = Exception("network error")
            docs = ingestor._fetch_massive_news("AAPL", days=30)

        assert docs == []

    def test_ingest_paces_massive_calls_between_symbols_not_after_last(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
        monkeypatch.delenv("TIINGO_API_KEY", raising=False)
        monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)

        class _NoStore:
            def add_documents(self, collection, docs):
                return len(docs)

        ingestor = NewsIngestor(store=_NoStore(), chunker=DocumentChunker(chunk_size=500))

        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get, \
             patch("firm.rag.ingestors.news_ingestor.time.sleep") as mock_sleep:
            mock_get.side_effect = [
                MagicMock(status_code=404),  # yahoo rss for AAPL
                _massive_response([]),        # massive for AAPL
                MagicMock(status_code=404),  # yahoo rss for MSFT
                _massive_response([]),        # massive for MSFT
            ]
            ingestor.ingest(symbols=["AAPL", "MSFT"], days=30)

        # Paces after the first symbol, not after the last.
        assert mock_sleep.call_count == 1


class TestFinnhubNewsSource:
    def test_fetch_finnhub_news_produces_documents(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
        ingestor = NewsIngestor(store=None, chunker=DocumentChunker(chunk_size=500))

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [{
            "headline": "Apple unveils new iPhone lineup",
            "summary": "Apple announced its latest iPhone models today.",
            "datetime": 1752840000,  # 2025-07-18T12:00:00Z
            "url": "https://example.com/aapl-iphone",
        }]
        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get:
            mock_get.return_value = resp
            docs = ingestor._fetch_finnhub_news("AAPL", days=30)

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "finnhub"
        assert docs[0].metadata["symbol"] == "AAPL"
        assert docs[0].metadata["date"] == "2025-07-18"
        assert "iPhone" in docs[0].text

    def test_fetch_finnhub_news_filters_irrelevant_article(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
        ingestor = NewsIngestor(store=None, chunker=DocumentChunker(chunk_size=500))

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [{
            "headline": "Recorded Future named a leader in threat intelligence",
            "summary": "The vendor topped an industry analyst report.",
            "datetime": 1752840000,
            "url": "https://example.com/unrelated",
        }]
        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get:
            mock_get.return_value = resp
            docs = ingestor._fetch_finnhub_news("MA", days=30)

        assert docs == []

    def test_fetch_finnhub_news_handles_failure_gracefully(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
        ingestor = NewsIngestor(store=None, chunker=DocumentChunker(chunk_size=500))

        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get:
            mock_get.side_effect = Exception("network error")
            docs = ingestor._fetch_finnhub_news("AAPL", days=30)

        assert docs == []

    def test_finnhub_not_called_without_api_key(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
        monkeypatch.delenv("TIINGO_API_KEY", raising=False)

        class _NoStore:
            def add_documents(self, collection, docs):
                return len(docs)

        ingestor = NewsIngestor(store=_NoStore(), chunker=DocumentChunker(chunk_size=500))
        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404)
            ingestor.ingest(symbols=["AAPL"], days=30)

        # Only yahoo rss is attempted when no provider keys are configured.
        assert mock_get.call_count == 1


class TestRelevanceFilterAcrossProviders:
    """A ticker-search endpoint can return an article that never actually
    mentions the queried company -- must not be stored tagged with the
    wrong symbol, regardless of which provider it came from."""

    def test_massive_drops_article_not_mentioning_symbol(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
        ingestor = NewsIngestor(store=None, chunker=DocumentChunker(chunk_size=500))

        articles = [{
            "title": "Recorded Future named a leader in threat intelligence",
            "description": "The vendor topped an industry analyst report.",
            "published_utc": "2026-07-18T12:00:00Z",
            "article_url": "https://example.com/unrelated",
        }]
        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get:
            mock_get.return_value = _massive_response(articles)
            docs = ingestor._fetch_massive_news("MA", days=30)

        assert docs == []

    def test_alphavantage_drops_article_not_mentioning_symbol(self, monkeypatch):
        monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
        ingestor = NewsIngestor(store=None, chunker=DocumentChunker(chunk_size=500))

        resp = MagicMock(
            status_code=200,
            json=lambda: {"feed": [{
                "title": "Anthropic's IPO could benefit cloud partners",
                "summary": "Analysts see upside for major cloud providers.",
                "time_published": "20260718T120000", "url": "https://av",
            }]},
        )
        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get:
            mock_get.return_value = resp
            docs = ingestor._fetch_alphavantage_news("AVGO")

        assert docs == []


class TestAlphaVantageIsFallbackOnly:
    """AlphaVantage's key is shared with price/fundamentals elsewhere in the
    system and capped at 25 req/day total — it must only be spent on RAG
    news when Massive didn't cover the symbol, never unconditionally."""

    class _NoStore:
        def add_documents(self, collection, docs):
            return len(docs)

    def _ingestor(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "massive-key")
        monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-key")
        monkeypatch.delenv("TIINGO_API_KEY", raising=False)
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        return NewsIngestor(store=self._NoStore(), chunker=DocumentChunker(chunk_size=500))

    def test_av_not_called_when_massive_has_coverage(self, monkeypatch):
        ingestor = self._ingestor(monkeypatch)
        massive_articles = [{
            "title": "Apple reports record earnings",
            "description": "Apple beat analyst estimates this quarter.",
            "published_utc": "2026-07-18T12:00:00Z", "article_url": "https://x",
        }]

        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get, \
             patch("firm.rag.ingestors.news_ingestor.time.sleep"):
            mock_get.side_effect = [
                MagicMock(status_code=404),           # yahoo rss
                _massive_response(massive_articles),  # massive: has coverage
            ]
            ingestor.ingest(symbols=["AAPL"], days=30)

        # 1 yahoo + 1 massive = 2 calls; alphavantage must not be reached.
        assert mock_get.call_count == 2

    def test_av_called_when_massive_has_no_coverage(self, monkeypatch):
        ingestor = self._ingestor(monkeypatch)

        with patch("firm.rag.ingestors.news_ingestor.requests.get") as mock_get, \
             patch("firm.rag.ingestors.news_ingestor.time.sleep"):
            mock_get.side_effect = [
                MagicMock(status_code=404),      # yahoo rss
                _massive_response([]),           # massive: nothing for this symbol
                MagicMock(                        # alphavantage: fallback fires
                    status_code=200,
                    json=lambda: {"feed": [{
                        "title": "AV headline", "summary": "AV summary",
                        "time_published": "20260718T120000", "url": "https://av",
                    }]},
                ),
            ]
            ingestor.ingest(symbols=["AAPL"], days=30)

        # 1 yahoo + 1 massive + 1 alphavantage = 3 calls when massive is empty.
        assert mock_get.call_count == 3
