"""Tests for FMPProvider's fundamentals point-in-time correctness.

Regression coverage: the ratios endpoint's "date" field is the fiscal
period-end date, not the actual filing/announcement date — a real
look-ahead bug for any strategy trusting date <= asof (multi_factor's
value/quality factors read this directly). A Q4 report would become
visible the instant the quarter ends, weeks to months before the
10-K/10-Q is actually public.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.data.providers.base import FUNDAMENTALS_PUBLICATION_LAG_DAYS, ProviderError
from firm.data.providers.fmp import FMPProvider


@pytest.fixture()
def provider():
    with patch("firm.data.providers.fmp.get_settings"):
        return FMPProvider(api_key="test-key")


class TestFMPFundamentalsPublicationLag:
    def test_date_is_shifted_forward_by_the_publication_lag(self, provider):
        period_end = "2025-12-31"
        ratios_response = [{
            "date": period_end, "fiscalYear": 2025, "period": "Q4",
            "priceToEarningsRatio": 20.0, "priceToBookRatio": 3.0,
            "returnOnEquity": 0.15, "debtToEquityRatio": 0.5,
            "dividendYield": 0.01,
        }]
        income_response = [{"fiscalYear": 2025, "period": "Q4", "revenue": 1e9, "netIncome": 1e8, "eps": 2.0}]
        metrics_response = [{"fiscalYear": 2025, "period": "Q4", "marketCap": 1e10}]

        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            side_effect=lambda path, **_: {
                "/stable/income-statement": income_response,
                "/stable/key-metrics": metrics_response,
                "/stable/ratios": ratios_response,
            }[path]
        )

        df = provider.get_fundamentals(["AAPL"], "2020-01-01", "2027-01-01")

        expected = pd.Timestamp(period_end) + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)
        assert len(df) == 1
        assert pd.Timestamp(df.iloc[0]["date"]) == expected
        assert pd.Timestamp(df.iloc[0]["date"]) > pd.Timestamp(period_end)

    def test_402_subscription_limit_logs_warning_not_exception(self, provider, caplog):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            side_effect=ProviderError(
                "https://financialmodelingprep.com/stable/income-statement "
                "returned HTTP 402: Premium Query Parameter"
            )
        )
        with caplog.at_level("WARNING"):
            df = provider.get_fundamentals(["AVGO"], "2020-01-01", "2026-01-01")
        assert df.empty
        assert any("fmp_fundamentals_unavailable" in r.message for r in caplog.records)
        assert not any(r.levelname == "ERROR" for r in caplog.records)

    def test_etf_symbols_skipped(self, provider):
        provider._client = MagicMock()
        df = provider.get_fundamentals(["SPY", "QQQ"], "2020-01-01", "2026-01-01")
        assert df.empty
        provider._client.get_json.assert_not_called()
