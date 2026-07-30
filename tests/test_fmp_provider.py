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
    def test_date_is_shifted_forward_by_the_publication_lag_when_no_filling_date(self, provider):
        """No `fillingDate` on the matched income record (e.g. it didn't
        match any ratios record) must fall back to the heuristic."""
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

    def test_uses_real_filling_date_when_available(self, provider):
        """FMP's income-statement endpoint exposes the real SEC
        `fillingDate` — prefer it over the period-end+lag heuristic."""
        period_end = "2025-12-31"
        filling_date = "2026-01-20"  # well under the 45-day heuristic
        ratios_response = [{
            "date": period_end, "fiscalYear": 2025, "period": "Q4",
            "priceToEarningsRatio": 20.0, "priceToBookRatio": 3.0,
            "returnOnEquity": 0.15, "debtToEquityRatio": 0.5,
            "dividendYield": 0.01,
        }]
        income_response = [{
            "fiscalYear": 2025, "period": "Q4", "revenue": 1e9,
            "netIncome": 1e8, "eps": 2.0, "fillingDate": filling_date,
        }]
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

        assert len(df) == 1
        assert pd.Timestamp(df.iloc[0]["date"]) == pd.Timestamp(filling_date)

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


class TestFMPAnalystRatings:
    """get_analyst_ratings backed by /stable/grades-historical — fixture
    shape matches a real live response captured 2026-07-31 (AAPL)."""

    _REAL_SHAPE_RESPONSE = [
        {
            "symbol": "AAPL", "date": "2026-07-01",
            "analystRatingsStrongBuy": 6, "analystRatingsBuy": 23,
            "analystRatingsHold": 17, "analystRatingsSell": 2,
            "analystRatingsStrongSell": 2,
        },
        {
            "symbol": "AAPL", "date": "2018-12-01",
            "analystRatingsStrongBuy": 4, "analystRatingsBuy": 20,
            "analystRatingsHold": 10, "analystRatingsSell": 1,
            "analystRatingsStrongSell": 0,
        },
    ]

    def test_maps_real_response_shape_to_analyst_ratings_cols(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(return_value=self._REAL_SHAPE_RESPONSE)

        df = provider.get_analyst_ratings(["AAPL"], "2000-01-01", "2027-01-01")

        assert list(df.columns) == ["date", "symbol", "strong_buy", "buy", "hold", "sell", "strong_sell"]
        assert len(df) == 2
        row = df[df["date"] == pd.Timestamp("2026-07-01")].iloc[0]
        assert row["strong_buy"] == 6
        assert row["buy"] == 23
        assert row["hold"] == 17
        assert row["sell"] == 2
        assert row["strong_sell"] == 2

    def test_no_limit_param_passed_full_history_requested(self, provider):
        """A 'limit' query param is capped at 10 under this plan tier —
        omitting it returns the full available history (verified live: 91
        monthly rows for AAPL back to 2018-12), so it must never be passed."""
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(return_value=[])
        provider.get_analyst_ratings(["AAPL"], "2000-01-01", "2027-01-01")
        _, kwargs = provider._client.get_json.call_args
        assert "limit" not in kwargs.get("params", {})

    def test_date_range_filter_excludes_rows_outside_window(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(return_value=self._REAL_SHAPE_RESPONSE)

        df = provider.get_analyst_ratings(["AAPL"], "2020-01-01", "2027-01-01")

        assert len(df) == 1
        assert df.iloc[0]["date"] == pd.Timestamp("2026-07-01")

    def test_402_subscription_limit_logs_warning_not_exception(self, provider, caplog):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            side_effect=ProviderError(
                "https://financialmodelingprep.com/stable/grades-historical "
                "returned HTTP 402: Premium Query Parameter"
            )
        )
        with caplog.at_level("WARNING"):
            df = provider.get_analyst_ratings(["AAPL"], "2020-01-01", "2026-01-01")
        assert df.empty
        assert any("fmp_analyst_ratings_unavailable" in r.message for r in caplog.records)
        assert not any(r.levelname == "ERROR" for r in caplog.records)

    def test_empty_response_returns_typed_empty_frame(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(return_value=[])
        df = provider.get_analyst_ratings(["AAPL"], "2020-01-01", "2026-01-01")
        assert df.empty
        assert list(df.columns) == ["date", "symbol", "strong_buy", "buy", "hold", "sell", "strong_sell"]

    def test_one_bad_symbol_does_not_fail_whole_batch(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            side_effect=lambda path, **kwargs: (
                (_ for _ in ()).throw(ProviderError("boom"))
                if kwargs["params"]["symbol"] == "BADSYM"
                else self._REAL_SHAPE_RESPONSE
            )
        )
        df = provider.get_analyst_ratings(["BADSYM", "AAPL"], "2000-01-01", "2027-01-01")
        assert not df.empty
        assert set(df["symbol"]) == {"AAPL"}
