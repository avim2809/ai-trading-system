"""Tests for MassiveProvider's rate-limit handling.

Regression coverage for a real production incident: a 25-symbol live cycle
against the free-tier account-wide rate limit spent minutes retrying each
already-rate-limited symbol with exponential backoff before ever reaching
the Tiingo/AlphaVantage/FMP fallback chain.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from firm.data.providers.base import ProviderError
from firm.data.providers.massive import MassiveProvider


@pytest.fixture()
def provider():
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
