"""Tests for FallbackProvider's per-symbol chain merging.

Regression coverage for a real incident: fetching 25 symbols only cached
data for 5 of them, with a normal-looking "Done" log line and no error.
Root cause — _try_chain treated a multi-symbol batch as one unit ("first
non-empty result wins"), so once Massive's rate limit kicked in partway
through (it catches errors per-symbol internally and returns whatever
succeeded), the partial non-empty result was accepted as a full success
and Tiingo/AlphaVantage/FMP were never tried for the other 20 symbols.
This silently starves both backtests (data_source="cache") and live
trading (which uses the same FallbackProvider for market data) of most of
the requested universe with zero visibility.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.data.providers.base import ProviderError
from firm.data.providers.fallback import FallbackProvider, _load
from firm.data.schemas import ANALYST_RATINGS_COLS, PRICE_COLS


def _price_df(symbols: list[str], date=None) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [date if date is not None else "2026-01-01"] * len(symbols),
        "symbol": symbols,
        "open": [1.0] * len(symbols),
        "high": [1.0] * len(symbols),
        "low": [1.0] * len(symbols),
        "close": [1.0] * len(symbols),
        "volume": [1.0] * len(symbols),
        "adj_close": [1.0] * len(symbols),
    })[PRICE_COLS]


def _price_df_multi_date(symbol: str, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates,
        "symbol": [symbol] * len(dates),
        "open": [1.0] * len(dates),
        "high": [1.0] * len(dates),
        "low": [1.0] * len(dates),
        "close": [1.0] * len(dates),
        "volume": [1.0] * len(dates),
        "adj_close": [1.0] * len(dates),
    })[PRICE_COLS]


@pytest.fixture()
def provider():
    with patch("firm.data.providers.fallback.get_settings"):
        return FallbackProvider(settings=MagicMock())


def _mock_provider(get_prices_return):
    p = MagicMock()
    p.get_prices = MagicMock(side_effect=get_prices_return) if callable(get_prices_return) else MagicMock(return_value=get_prices_return)
    return p


class TestPartialFallbackMerging:
    def test_partial_primary_result_falls_through_for_missing_symbols(self, provider):
        massive = _mock_provider(_price_df(["AAPL", "MSFT"]))  # only 2 of 5 succeeded
        tiingo = _mock_provider(_price_df(["GOOG", "AMZN", "META"]))  # covers the rest

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: {"massive": massive, "tiingo": tiingo}.get(name)):
            result = provider.get_prices(["AAPL", "MSFT", "GOOG", "AMZN", "META"], "2026-01-01", "2026-01-02")

        assert set(result["symbol"]) == {"AAPL", "MSFT", "GOOG", "AMZN", "META"}
        # Tiingo must only have been asked for the symbols Massive missed.
        tiingo.get_prices.assert_called_once_with(["GOOG", "AMZN", "META"], "2026-01-01", "2026-01-02")

    def test_full_primary_success_skips_remaining_chain(self, provider):
        massive = _mock_provider(_price_df(["AAPL", "MSFT"]))
        tiingo = MagicMock()

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: {"massive": massive, "tiingo": tiingo}.get(name)):
            result = provider.get_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-02")

        assert set(result["symbol"]) == {"AAPL", "MSFT"}
        tiingo.get_prices.assert_not_called()

    def test_all_providers_fail_returns_empty_with_warning(self, provider, caplog):
        with patch("firm.data.providers.fallback._load", return_value=None):
            with caplog.at_level("WARNING"):
                result = provider.get_prices(["AAPL"], "2026-01-01", "2026-01-02")

        assert result.empty
        assert any("fallback_incomplete" in r.message for r in caplog.records)

    def test_provider_error_does_not_abort_the_chain(self, provider):
        massive = MagicMock()
        massive.get_prices = MagicMock(side_effect=ProviderError("rate limited"))
        tiingo = _mock_provider(_price_df(["AAPL"]))

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: {"massive": massive, "tiingo": tiingo}.get(name)):
            result = provider.get_prices(["AAPL"], "2026-01-01", "2026-01-02")

        assert set(result["symbol"]) == {"AAPL"}

    def test_still_missing_symbols_after_full_chain_are_logged_not_silent(self, provider, caplog):
        massive = _mock_provider(_price_df(["AAPL"]))

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: massive if name == "massive" else None):
            with caplog.at_level("WARNING"):
                result = provider.get_prices(["AAPL", "MSFT", "GOOG"], "2026-01-01", "2026-01-02")

        assert set(result["symbol"]) == {"AAPL"}
        assert any("MSFT" in r.message and "GOOG" in r.message for r in caplog.records)


class TestMergedResultHasUniformDateDtype:
    """Regression: Massive/FMP/AlphaVantage store "date" as a normalized
    Timestamp; Tiingo used to store it as a python datetime.date. Merging a
    partial Massive result with a Tiingo fallthrough for the rest produced a
    "date" column pyarrow couldn't write to parquet — this only ever
    surfaced once a real rate limit forced an actual multi-provider merge,
    the exact case this whole fallback chain exists to handle.
    """

    def test_mixed_date_dtypes_across_providers_are_normalized_after_merge(self, provider, tmp_path):
        import datetime as dt

        massive = _mock_provider(_price_df(["AAPL"], date=pd.Timestamp("2026-01-01")))
        tiingo = _mock_provider(_price_df(["MSFT"], date=dt.date(2026, 1, 1)))

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: {"massive": massive, "tiingo": tiingo}.get(name)):
            result = provider.get_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-02")

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        # Must survive the exact operation that crashed in production.
        result.to_parquet(tmp_path / "out.parquet", index=False)

    def test_mixed_tz_aware_and_naive_dates_across_providers_are_normalized(self, provider, tmp_path):
        """The actual production crash: pd.to_datetime() on a column mixing
        tz-naive and tz-aware Timestamps raises ValueError, not just a
        pyarrow write error — this must be handled before parquet is even
        attempted.
        """
        massive = _mock_provider(_price_df(["AAPL"], date=pd.Timestamp("2026-01-01")))  # naive
        tiingo = _mock_provider(_price_df(["MSFT"], date=pd.Timestamp("2026-01-01", tz="UTC")))  # aware

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: {"massive": massive, "tiingo": tiingo}.get(name)):
            result = provider.get_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-02")

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["date"].dt.tz is None
        result.to_parquet(tmp_path / "out.parquet", index=False)


class TestTruncatedRangeIsNotAcceptedAsFullyResolved:
    """Regression: a real incident where Massive's free tier silently
    truncated history to its own ~2-year rolling window instead of erroring
    — returning real, non-empty data for AAPL/MSFT/NVDA/GOOG/AMZN, just not
    covering the requested 2020-01-01 start. The old "got any non-empty
    rows for this symbol" check accepted that as fully resolved, so Tiingo
    (which had the full range for every *other* symbol in the same run)
    never got a chance to supply the missing years for these five.
    """

    def test_provider_with_truncated_history_does_not_block_a_fuller_fallback(self, provider):
        massive = _mock_provider(_price_df_multi_date("AAPL", ["2024-07-22", "2024-07-23"]))
        tiingo = _mock_provider(_price_df_multi_date("AAPL", ["2020-01-02", "2020-01-03", "2024-07-22", "2024-07-23"]))

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: {"massive": massive, "tiingo": tiingo}.get(name)):
            result = provider.get_prices(["AAPL"], "2020-01-01", "2026-07-20")

        # Tiingo's fuller answer wins outright — no splicing/mixing sources.
        tiingo.get_prices.assert_called_once_with(["AAPL"], "2020-01-01", "2026-07-20")
        assert sorted(result["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03", "2024-07-22", "2024-07-23"]

    def test_truncated_history_is_kept_as_last_resort_when_no_provider_has_more(self, provider, caplog):
        massive = _mock_provider(_price_df_multi_date("AAPL", ["2024-07-22", "2024-07-23"]))
        # Tiingo also can't reach back further — same truncated window.
        tiingo = _mock_provider(_price_df_multi_date("AAPL", ["2024-07-22", "2024-07-23"]))

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: {"massive": massive, "tiingo": tiingo}.get(name)):
            with caplog.at_level("WARNING"):
                result = provider.get_prices(["AAPL"], "2020-01-01", "2026-07-20")

        assert sorted(result["date"].dt.strftime("%Y-%m-%d")) == ["2024-07-22", "2024-07-23"]
        assert any("fallback_truncated_range" in r.message for r in caplog.records)

    def test_full_coverage_symbol_is_not_reclassified_as_truncated(self, provider):
        """A symbol whose provider genuinely covers the requested start must
        not be re-queried against later providers in the chain."""
        massive = _mock_provider(_price_df_multi_date("AAPL", ["2020-01-02", "2020-01-03"]))
        tiingo = MagicMock()

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: {"massive": massive, "tiingo": tiingo}.get(name)):
            result = provider.get_prices(["AAPL"], "2020-01-01", "2026-07-20")

        tiingo.get_prices.assert_not_called()
        assert sorted(result["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03"]


class TestLoadDoesNotCrashOnBrokenProvider:
    """Regression: a real incident where TiingoProvider had no __init__
    accepting `settings=`, so get_provider("tiingo", settings=cfg) raised
    TypeError. _load only caught KeyError/ProviderError/ValueError, so that
    TypeError propagated uncaught and crashed the entire fetch-data run (and
    would have crashed a live trading cycle the same way) the moment Massive
    rate-limited and the chain tried to fall through to Tiingo.
    """

    def test_unexpected_constructor_error_is_skipped_not_raised(self, caplog):
        with patch("firm.data.providers.get_provider", side_effect=TypeError("bad kwarg")):
            with caplog.at_level("WARNING"):
                result = _load("tiingo", MagicMock())

        assert result is None
        assert any("data_provider_init_failed" in r.message for r in caplog.records)


class TestAnalystRatingsChain:
    """analyst_ratings chain is FMP-only (no other provider implements it)."""

    def test_delegates_to_fmp(self, provider):
        ratings_df = pd.DataFrame(
            [{"date": "2026-07-01", "symbol": "AAPL", "strong_buy": 6, "buy": 23,
              "hold": 17, "sell": 2, "strong_sell": 2}],
            columns=ANALYST_RATINGS_COLS,
        )
        fmp = _mock_provider(None)
        fmp.get_analyst_ratings = MagicMock(return_value=ratings_df)

        with patch("firm.data.providers.fallback._load", side_effect=lambda name, cfg: {"fmp": fmp}.get(name)):
            result = provider.get_analyst_ratings(["AAPL"], "2020-01-01", "2026-08-01")

        fmp.get_analyst_ratings.assert_called_once()
        assert not result.empty
        assert result.iloc[0]["strong_buy"] == 6

    def test_fmp_unavailable_returns_empty_typed_frame(self, provider):
        with patch("firm.data.providers.fallback._load", return_value=None):
            result = provider.get_analyst_ratings(["AAPL"], "2020-01-01", "2026-08-01")
        assert result.empty
        assert list(result.columns) == ANALYST_RATINGS_COLS
