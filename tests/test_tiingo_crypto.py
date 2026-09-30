"""Tests for TiingoProvider's crypto daily-bar support (BTC trend satellite
sleeve work) -- the fallback source for BTC/USD daily bars when
AlpacaProvider's own crypto data client is unavailable, via Tiingo's
``/tiingo/crypto/prices`` endpoint (TIINGO_API_KEY, same key as equities).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pandas as pd

from firm.data.providers.tiingo import TiingoProvider


def _crypto_payload(bars: list[dict]) -> list[dict]:
    return [
        {
            "ticker": "btcusd",
            "baseCurrency": "btc",
            "quoteCurrency": "usd",
            "priceData": bars,
        }
    ]


class TestTiingoCryptoPrices:
    def test_get_prices_routes_slash_symbol_to_crypto_endpoint(self):
        provider = TiingoProvider(api_key="test-key")
        payload = _crypto_payload(
            [
                {"date": "2026-08-02T00:00:00.000Z", "open": 60000.0, "high": 61000.0,
                 "low": 59500.0, "close": 60500.0, "volume": 1234.5},
                {"date": "2026-08-03T00:00:00.000Z", "open": 60500.0, "high": 62000.0,
                 "low": 60000.0, "close": 61500.0, "volume": 987.6},
            ]
        )
        with patch.object(provider, "_get", return_value=payload) as mock_get, \
             patch("firm.data.providers.tiingo.utcnow", return_value=datetime(2026, 9, 30)):
            df = provider.get_prices(["BTC/USD"], "2026-08-01", "2026-08-05")

        url = mock_get.call_args[0][0]
        assert url == "https://api.tiingo.com/tiingo/crypto/prices"
        params = mock_get.call_args[1]["params"]
        assert params["tickers"] == "btcusd"

        assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]
        assert len(df) == 2
        assert set(df["symbol"]) == {"BTC/USD"}
        row = df.iloc[0]
        assert row["date"] == pd.Timestamp("2026-08-02")
        assert row["close"] == 60500.0
        assert row["adj_close"] == row["close"]

    def test_todays_forming_bar_is_dropped(self):
        provider = TiingoProvider(api_key="test-key")
        payload = _crypto_payload(
            [
                {"date": "2026-09-29T00:00:00.000Z", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10},
                {"date": "2026-09-30T00:00:00.000Z", "open": 2, "high": 3, "low": 1, "close": 2.5, "volume": 20},
            ]
        )
        with patch.object(provider, "_get", return_value=payload), \
             patch("firm.data.providers.tiingo.utcnow", return_value=datetime(2026, 9, 30, 14, 0, 0)):
            df = provider.get_prices(["BTC/USD"], "2026-09-25", "2026-09-30")

        assert len(df) == 1
        assert df.iloc[0]["date"] == pd.Timestamp("2026-09-29")

    def test_empty_price_data_returns_empty_frame(self):
        provider = TiingoProvider(api_key="test-key")
        with patch.object(provider, "_get", return_value=_crypto_payload([])), \
             patch("firm.data.providers.tiingo.utcnow", return_value=datetime(2026, 9, 30)):
            df = provider.get_prices(["BTC/USD"], "2026-08-01", "2026-08-05")
        assert df.empty

    def test_mixed_equity_and_crypto_symbols_both_resolve(self):
        provider = TiingoProvider(api_key="test-key")
        equity_raw = [{"date": "2026-08-03T00:00:00.000Z", "adjClose": 211.5, "open": 210.0,
                       "high": 212.0, "low": 209.0, "close": 211.5, "volume": 1_000_000}]
        crypto_payload = _crypto_payload(
            [{"date": "2026-08-03T00:00:00.000Z", "open": 60000.0, "high": 61000.0,
              "low": 59500.0, "close": 60500.0, "volume": 1234.5}]
        )

        def fake_get(url, params=None):
            if "crypto" in url:
                return crypto_payload
            return equity_raw

        with patch.object(provider, "_get", side_effect=fake_get), \
             patch("firm.data.providers.tiingo.utcnow", return_value=datetime(2026, 9, 30)):
            df = provider.get_prices(["AAPL", "BTC/USD"], "2026-08-01", "2026-08-05")

        assert set(df["symbol"]) == {"AAPL", "BTC/USD"}
