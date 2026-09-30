"""Tests for AlpacaProvider's crypto daily-bar support (BTC trend satellite
sleeve work): historical BTC/USD daily bars via alpaca-py's
CryptoHistoricalDataClient, partitioned correctly alongside equity symbols,
with today's still-forming UTC bar always dropped. No network is hit --
alpaca-py's clients are mocked, per this file's sibling test_alpaca_provider.py.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from firm.data.providers.alpaca import AlpacaProvider


def _bar(timestamp, open_, high, low, close, volume):
    bar = MagicMock()
    bar.timestamp = timestamp
    bar.open = open_
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    return bar


@patch("firm.data.providers.alpaca.NewsClient")
@patch("firm.data.providers.alpaca.CryptoHistoricalDataClient")
@patch("firm.data.providers.alpaca.StockHistoricalDataClient")
def test_get_prices_routes_crypto_symbol_to_crypto_client(mock_stock_cls, mock_crypto_cls, mock_news_cls):
    mock_crypto_client = mock_crypto_cls.return_value
    barset = MagicMock()
    barset.data = {
        "BTC/USD": [
            _bar(datetime(2026, 8, 2), 60000.0, 61000.0, 59500.0, 60500.0, 1234.5),
            _bar(datetime(2026, 8, 3), 60500.0, 62000.0, 60000.0, 61500.0, 987.6),
        ],
    }
    mock_crypto_client.get_crypto_bars.return_value = barset

    with patch("firm.data.providers.alpaca.utcnow", return_value=datetime(2026, 9, 30)):
        provider = AlpacaProvider(api_key="k", secret_key="s")
        df = provider.get_prices(["BTC/USD"], "2026-08-01", "2026-08-05")

    mock_stock_cls.return_value.get_stock_bars.assert_not_called()
    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]
    assert len(df) == 2
    assert set(df["symbol"]) == {"BTC/USD"}
    row = df.iloc[0]
    assert row["close"] == 60500.0
    assert row["adj_close"] == row["close"]


@patch("firm.data.providers.alpaca.NewsClient")
@patch("firm.data.providers.alpaca.CryptoHistoricalDataClient")
@patch("firm.data.providers.alpaca.StockHistoricalDataClient")
def test_todays_forming_utc_bar_is_dropped(mock_stock_cls, mock_crypto_cls, mock_news_cls):
    mock_crypto_client = mock_crypto_cls.return_value
    barset = MagicMock()
    barset.data = {
        "BTC/USD": [
            _bar(datetime(2026, 9, 29), 1, 2, 1, 1.5, 10),
            _bar(datetime(2026, 9, 30), 2, 3, 1, 2.5, 20),  # today -- still forming
        ],
    }
    mock_crypto_client.get_crypto_bars.return_value = barset

    with patch("firm.data.providers.alpaca.utcnow", return_value=datetime(2026, 9, 30, 14, 0, 0)):
        provider = AlpacaProvider(api_key="k", secret_key="s")
        df = provider.get_prices(["BTC/USD"], "2026-09-25", "2026-09-30")

    assert len(df) == 1
    assert df.iloc[0]["date"] == datetime(2026, 9, 29).date()


@patch("firm.data.providers.alpaca.NewsClient")
@patch("firm.data.providers.alpaca.CryptoHistoricalDataClient")
@patch("firm.data.providers.alpaca.StockHistoricalDataClient")
def test_mixed_equity_and_crypto_batch_partitions_correctly(mock_stock_cls, mock_crypto_cls, mock_news_cls):
    mock_stock_client = mock_stock_cls.return_value
    stock_barset = MagicMock()
    stock_barset.data = {"AAPL": [_bar(datetime(2026, 8, 3), 210.0, 212.0, 209.0, 211.5, 1_000_000)]}
    mock_stock_client.get_stock_bars.return_value = stock_barset

    mock_crypto_client = mock_crypto_cls.return_value
    crypto_barset = MagicMock()
    crypto_barset.data = {"BTC/USD": [_bar(datetime(2026, 8, 3), 60000.0, 61000.0, 59500.0, 60500.0, 1234.5)]}
    mock_crypto_client.get_crypto_bars.return_value = crypto_barset

    with patch("firm.data.providers.alpaca.utcnow", return_value=datetime(2026, 9, 30)):
        provider = AlpacaProvider(api_key="k", secret_key="s")
        df = provider.get_prices(["AAPL", "BTC/USD"], "2026-08-01", "2026-08-05")

    assert set(df["symbol"]) == {"AAPL", "BTC/USD"}
    # Each client only ever sees its own asset class's symbols.
    stock_request = mock_stock_client.get_stock_bars.call_args[0][0]
    assert stock_request.symbol_or_symbols == ["AAPL"]
    crypto_request = mock_crypto_client.get_crypto_bars.call_args[0][0]
    assert crypto_request.symbol_or_symbols == ["BTC/USD"]


@patch("firm.data.providers.alpaca.NewsClient")
@patch("firm.data.providers.alpaca.CryptoHistoricalDataClient")
@patch("firm.data.providers.alpaca.StockHistoricalDataClient")
def test_missing_crypto_symbol_returns_empty_for_it(mock_stock_cls, mock_crypto_cls, mock_news_cls):
    mock_crypto_client = mock_crypto_cls.return_value
    barset = MagicMock()
    barset.data = {}
    mock_crypto_client.get_crypto_bars.return_value = barset

    with patch("firm.data.providers.alpaca.utcnow", return_value=datetime(2026, 9, 30)):
        provider = AlpacaProvider(api_key="k", secret_key="s")
        df = provider.get_prices(["BTC/USD"], "2026-08-01", "2026-08-05")

    assert df.empty
