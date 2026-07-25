"""Tests for MassiveProvider's rate-limit handling.

Regression coverage for a real production incident: a 25-symbol live cycle
against the free-tier account-wide rate limit spent minutes retrying each
already-rate-limited symbol with exponential backoff before ever reaching
the Tiingo/AlphaVantage/FMP fallback chain.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.data.providers.base import ProviderError
from firm.data.providers.massive import MassiveProvider


@pytest.fixture()
def provider():
    MassiveProvider._fundamentals_plan_blocked = False
    return MassiveProvider(api_key="test-key")


class TestMassiveRateLimitHandling:
    @patch("firm.data.providers.massive.requests.get")
    @patch("firm.data.providers.massive.time.sleep")
    def test_429_fails_fast_without_retry(self, mock_sleep, mock_get, provider):
        resp = MagicMock(status_code=429)
        mock_get.return_value = resp

        with pytest.raises(ProviderError, match="429"):
            provider._get("/v2/aggs/ticker/AAPL/range/1/day/2026-01-01/2026-01-02")

        # Exactly one request — no retry loop, no backoff sleep.
        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    @patch("firm.data.providers.massive.requests.get")
    def test_get_prices_skips_rate_limited_symbols_quickly(self, mock_get, provider):
        mock_get.return_value = MagicMock(status_code=429)

        df = provider.get_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-02")

        assert df.empty
        # One request per symbol, no retries.
        assert mock_get.call_count == 2

    @patch("firm.data.providers.massive.requests.get")
    @patch("firm.data.providers.massive.time.sleep")
    def test_transient_5xx_still_retries(self, mock_sleep, mock_get, provider):
        # Unlike 429, a transient server error should still retry — only
        # the account-wide rate limit case was the problem.
        mock_get.side_effect = [
            MagicMock(status_code=503),
            MagicMock(status_code=503),
            MagicMock(status_code=200, ok=True, json=lambda: {"results": []}),
        ]
        result = provider._get("/v2/aggs/ticker/AAPL/range/1/day/2026-01-01/2026-01-02")
        assert result == {"results": []}
        assert mock_get.call_count == 3

    @patch("firm.data.providers.massive.requests.get")
    @patch("firm.data.providers.massive.time.sleep")
    def test_news_waits_between_symbols(self, mock_sleep, mock_get, provider, monkeypatch):
        monkeypatch.setenv("MASSIVE_NEWS_MIN_INTERVAL_SEC", "1.0")
        ok = MagicMock(status_code=200, ok=True, json=lambda: {"results": []})
        mock_get.return_value = ok

        provider.get_news_sentiment(["AAPL", "MSFT"], "2026-01-01", "2026-01-02")

        assert mock_get.call_count == 2
        assert mock_sleep.call_count >= 1

    @patch("firm.data.providers.massive.requests.get")
    @patch("firm.data.providers.massive.time.sleep")
    def test_news_429_stops_batch(self, mock_sleep, mock_get, provider):
        mock_get.return_value = MagicMock(status_code=429)

        df = provider.get_news_sentiment(["AAPL", "MSFT", "NVDA"], "2026-01-01", "2026-01-02")

        assert df.empty
        assert mock_get.call_count == 1


class TestMassiveFundamentalsPublicationLag:
    """Regression: the ratios endpoint's "date" is the fiscal period-end,
    not the actual filing date — a real look-ahead bug for any strategy
    trusting date <= asof (multi_factor's value/quality factors read this
    directly). A Q4 report would become visible the instant the quarter
    ends, weeks to months before the 10-K/10-Q is actually public.
    """

    @patch.object(MassiveProvider, "_get")
    def test_date_is_shifted_forward_by_the_publication_lag(self, mock_get, provider):
        from firm.data.providers.base import FUNDAMENTALS_PUBLICATION_LAG_DAYS

        period_end = "2025-12-31"
        mock_get.return_value = {
            "results": [{
                "date": period_end, "market_cap": 1e9, "price_to_earnings": 20.0,
                "price_to_book": 3.0, "return_on_equity": 0.15, "debt_to_equity": 0.5,
                "earnings_per_share": 2.0,
            }]
        }

        df = provider.get_fundamentals(["AAPL"], "2020-01-01", "2027-01-01")

        expected = pd.Timestamp(period_end) + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)
        assert len(df) == 1
        assert pd.Timestamp(df.iloc[0]["date"]) == expected
        assert pd.Timestamp(df.iloc[0]["date"]) > pd.Timestamp(period_end)


class TestMassiveFundamentalsPlanGate:
    @patch("firm.data.providers.massive.requests.get")
    def test_403_fails_fast_for_remaining_symbols(self, mock_get, provider):
        mock_get.return_value = MagicMock(
            status_code=403,
            ok=False,
            text='{"status":"NOT_AUTHORIZED","message":"upgrade"}',
        )
        df = provider.get_fundamentals(["AAPL", "MSFT", "GOOG"], "2020-01-01", "2027-01-01")
        assert df.empty
        assert mock_get.call_count == 1
