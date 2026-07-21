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
from firm.data.providers.fallback import FallbackProvider
from firm.data.schemas import PRICE_COLS


def _price_df(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-01-01"] * len(symbols),
        "symbol": symbols,
        "open": [1.0] * len(symbols),
        "high": [1.0] * len(symbols),
        "low": [1.0] * len(symbols),
        "close": [1.0] * len(symbols),
        "volume": [1.0] * len(symbols),
        "adj_close": [1.0] * len(symbols),
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
