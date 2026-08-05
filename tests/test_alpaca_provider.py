"""Tests for AlpacaProvider (prices via Alpaca's Market Data API, news via
Alpaca's News API)."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from firm.data.providers.alpaca import AlpacaProvider
from firm.data.providers.base import ProviderError


def _bar(timestamp, open_, high, low, close, volume):
    bar = MagicMock()
    bar.timestamp = timestamp
    bar.open = open_
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    return bar


def _news(headline, created_at, symbols, source="Benzinga"):
    article = MagicMock()
    article.headline = headline
    article.created_at = created_at
    article.symbols = symbols
    article.source = source
    return article


@patch("firm.data.providers.alpaca.NewsClient")
@patch("firm.data.providers.alpaca.StockHistoricalDataClient")
def test_get_prices_maps_bars_to_price_cols(mock_stock_cls, mock_news_cls):
    mock_stock_client = mock_stock_cls.return_value
    barset = MagicMock()
    barset.data = {
        "AAPL": [
            _bar(datetime(2026, 8, 3), 210.0, 212.0, 209.0, 211.5, 1_000_000),
            _bar(datetime(2026, 8, 4), 211.5, 213.0, 210.5, 212.0, 900_000),
        ],
    }
    mock_stock_client.get_stock_bars.return_value = barset

    provider = AlpacaProvider(api_key="k", secret_key="s")
    df = provider.get_prices(["AAPL"], "2026-08-01", "2026-08-05")

    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]
    assert len(df) == 2
    row = df.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["date"] == date(2026, 8, 3)
    assert row["close"] == 211.5
    assert row["adj_close"] == row["close"]


@patch("firm.data.providers.alpaca.NewsClient")
@patch("firm.data.providers.alpaca.StockHistoricalDataClient")
def test_get_prices_missing_symbol_returns_empty_for_it(mock_stock_cls, mock_news_cls):
    mock_stock_client = mock_stock_cls.return_value
    barset = MagicMock()
    barset.data = {"AAPL": [_bar(datetime(2026, 8, 3), 1, 2, 1, 1.5, 100)]}
    mock_stock_client.get_stock_bars.return_value = barset

    provider = AlpacaProvider(api_key="k", secret_key="s")
    df = provider.get_prices(["AAPL", "ZZZZ"], "2026-08-01", "2026-08-05")

    assert set(df["symbol"]) == {"AAPL"}


@patch("firm.data.providers.alpaca.NewsClient")
@patch("firm.data.providers.alpaca.StockHistoricalDataClient")
def test_get_prices_wraps_client_errors_as_provider_error(mock_stock_cls, mock_news_cls):
    mock_stock_client = mock_stock_cls.return_value
    mock_stock_client.get_stock_bars.side_effect = RuntimeError("boom")

    provider = AlpacaProvider(api_key="k", secret_key="s")
    with pytest.raises(ProviderError):
        provider.get_prices(["AAPL"], "2026-08-01", "2026-08-05")


@patch("firm.data.providers.alpaca.NewsClient")
@patch("firm.data.providers.alpaca.StockHistoricalDataClient")
def test_get_news_sentiment_filters_to_requested_symbols(mock_stock_cls, mock_news_cls):
    mock_news_client = mock_news_cls.return_value
    newsset = MagicMock()
    newsset.data = {
        "news": [
            _news("Apple beats earnings", datetime(2026, 8, 3), ["AAPL"]),
            _news("Random other-company news", datetime(2026, 8, 3), ["ZZZZ"]),
            _news("Tech rally lifts megacaps", datetime(2026, 8, 3), ["AAPL", "MSFT"]),
        ]
    }
    mock_news_client.get_news.return_value = newsset

    provider = AlpacaProvider(api_key="k", secret_key="s")
    df = provider.get_news_sentiment(["AAPL", "MSFT"], "2026-08-01", "2026-08-05")

    assert set(df["symbol"]) == {"AAPL", "MSFT"}
    aapl_row = df[df["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["news_volume"] == 2  # two articles mention AAPL


@patch("firm.data.providers.alpaca.NewsClient")
@patch("firm.data.providers.alpaca.StockHistoricalDataClient")
def test_get_news_sentiment_empty_when_no_articles(mock_stock_cls, mock_news_cls):
    mock_news_client = mock_news_cls.return_value
    newsset = MagicMock()
    newsset.data = {"news": []}
    mock_news_client.get_news.return_value = newsset

    provider = AlpacaProvider(api_key="k", secret_key="s")
    df = provider.get_news_sentiment(["AAPL"], "2026-08-01", "2026-08-05")

    assert df.empty


def test_missing_api_key_raises():
    settings = MagicMock()
    settings.require.side_effect = ValueError("Settings field 'alpaca_api_key' is required")
    with pytest.raises(ValueError):
        AlpacaProvider(settings=settings)


def test_unimplemented_capabilities_raise_not_implemented():
    provider = AlpacaProvider(api_key="k", secret_key="s")
    with pytest.raises(NotImplementedError):
        provider.get_fundamentals(["AAPL"], "2026-08-01", "2026-08-05")
    with pytest.raises(NotImplementedError):
        provider.get_corporate_actions(["AAPL"], "2026-08-01", "2026-08-05")
    with pytest.raises(NotImplementedError):
        provider.get_universe_constituents("SP500", "2026-08-05")
    with pytest.raises(NotImplementedError):
        provider.get_analyst_ratings(["AAPL"], "2026-08-01", "2026-08-05")
    with pytest.raises(NotImplementedError):
        provider.get_ai_scores(["AAPL"], "2026-08-01", "2026-08-05")
    with pytest.raises(NotImplementedError):
        provider.get_live_signals(["AAPL"])
    with pytest.raises(NotImplementedError):
        provider.get_best_stocks()
