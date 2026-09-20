"""Tests for firm.live.news_ingestion_job — the scheduled entrypoint that
populates the RAG "news" collection (see NewsIngestor) on a recurring basis.

No real network calls: NewsIngestor itself is mocked out entirely, since
its own provider-fetch behavior is already covered by test_news_ingestor.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from firm.live.news_ingestion_job import (
    ingest_recent_news,
    run_scheduled_news_ingestion,
)


class TestIngestRecentNews:
    def test_empty_symbols_is_a_noop(self):
        count = ingest_recent_news([])
        assert count == 0

    def test_calls_news_ingestor_with_symbols_and_days(self):
        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = 7
        with patch(
            "firm.rag.ingestors.news_ingestor.NewsIngestor",
            return_value=mock_ingestor,
        ) as mock_cls, patch("firm.rag.store.VectorStore"), patch(
            "firm.rag.chunker.DocumentChunker"
        ):
            count = ingest_recent_news(["AAPL", "MSFT"], days=5)

        assert count == 7
        mock_cls.assert_called_once()
        mock_ingestor.ingest.assert_called_once_with(symbols=["AAPL", "MSFT"], days=5)

    def test_default_lookback_is_three_days(self):
        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = 0
        with patch(
            "firm.rag.ingestors.news_ingestor.NewsIngestor",
            return_value=mock_ingestor,
        ), patch("firm.rag.store.VectorStore"), patch("firm.rag.chunker.DocumentChunker"):
            ingest_recent_news(["AAPL"])

        mock_ingestor.ingest.assert_called_once_with(symbols=["AAPL"], days=3)

    def test_ingestor_construction_failure_is_caught_and_logged(self, caplog):
        with patch(
            "firm.rag.store.VectorStore", side_effect=RuntimeError("chroma unavailable"),
        ):
            count = ingest_recent_news(["AAPL"])
        assert count == 0

    def test_ingest_failure_is_caught_and_logged(self):
        mock_ingestor = MagicMock()
        mock_ingestor.ingest.side_effect = RuntimeError("provider exploded")
        with patch(
            "firm.rag.ingestors.news_ingestor.NewsIngestor",
            return_value=mock_ingestor,
        ), patch("firm.rag.store.VectorStore"), patch("firm.rag.chunker.DocumentChunker"):
            count = ingest_recent_news(["AAPL"])
        assert count == 0


class TestRunScheduledNewsIngestion:
    def test_runs_ingestion_in_background_thread(self):
        with patch(
            "firm.live.news_ingestion_job.ingest_recent_news"
        ) as mock_ingest:
            run_scheduled_news_ingestion(["AAPL", "MSFT"], days=4)
            # Background thread — give it a moment to actually invoke the
            # patched function rather than asserting immediately.
            import time
            for _ in range(50):
                if mock_ingest.called:
                    break
                time.sleep(0.02)

        mock_ingest.assert_called_once_with(["AAPL", "MSFT"], days=4)

    def test_background_failure_does_not_raise(self):
        with patch(
            "firm.live.news_ingestion_job.ingest_recent_news",
            side_effect=RuntimeError("boom"),
        ):
            run_scheduled_news_ingestion(["AAPL"])  # must not raise
