"""Tests for TiingoProvider construction.

Regression coverage for a real production incident: TiingoProvider had no
__init__ override, so it inherited DataProvider.__init__(self, api_key).
FallbackProvider._load() always calls get_provider(name, settings=cfg) — the
exact call every other provider in the chain (Massive, AlphaVantage, FMP)
supports — and that raised TypeError for Tiingo specifically. This never
showed up in a quiet chain, only once Massive's rate limit forced a real
fallthrough to Tiingo, at which point it crashed the whole fetch (and would
have crashed a live trading cycle the same way).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.data.providers import get_provider
from firm.data.providers.tiingo import TiingoProvider


class TestTiingoConstruction:
    def test_accepts_settings_kwarg_like_the_rest_of_the_chain(self):
        settings = MagicMock()
        settings.require.return_value = "key-from-settings"

        provider = TiingoProvider(settings=settings)

        assert provider.api_key == "key-from-settings"
        settings.require.assert_called_once_with("tiingo_api_key")

    def test_explicit_api_key_bypasses_settings_lookup(self):
        settings = MagicMock()

        provider = TiingoProvider(api_key="explicit-key", settings=settings)

        assert provider.api_key == "explicit-key"
        settings.require.assert_not_called()

    def test_get_provider_with_settings_kwarg_does_not_raise(self):
        """This is the exact call FallbackProvider._load() makes."""
        settings = MagicMock()
        settings.require.return_value = "key"

        provider = get_provider("tiingo", settings=settings)

        assert isinstance(provider, TiingoProvider)


class TestTiingoDateDtype:
    """Regression: Tiingo used to emit "date" as python datetime.date
    objects while every other provider in the chain (Massive, FMP,
    AlphaVantage) emits a normalized Timestamp. Merging results from two
    providers with mismatched date dtypes broke pyarrow's parquet writer —
    only surfaced once a real rate limit forced an actual multi-provider
    merge in production.
    """

    def test_get_prices_date_column_matches_the_rest_of_the_chain(self):
        provider = TiingoProvider(api_key="test-key")
        raw = [{"date": "2026-01-01T00:00:00.000Z", "adjClose": 100.0, "open": 99.0,
                "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000.0}]

        with patch.object(provider, "_get", return_value=raw):
            df = provider.get_prices(["AAPL"], "2026-01-01", "2026-01-02")

        assert pd.api.types.is_datetime64_any_dtype(df["date"])
