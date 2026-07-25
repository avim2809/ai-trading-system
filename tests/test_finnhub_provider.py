"""Tests for Finnhub fundamentals provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from firm.data.providers.finnhub import FinnhubProvider, _metric_payload_to_rows


@patch("firm.data.providers.finnhub.RestClient")
def test_company_news_sentiment(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.get_json.return_value = [
        {
            "datetime": 1717200000,
            "headline": "Apple beats earnings estimates",
            "source": "Reuters",
        },
        {
            "datetime": 1717200000,
            "headline": "Apple warns on weak demand",
            "source": "Reuters",
        },
    ]
    provider = FinnhubProvider(api_key="test", settings=MagicMock(
        request_timeout_seconds=10, max_retries=1,
    ))
    df = provider.get_news_sentiment(["AAPL"], "2024-05-01", "2024-06-01")
    assert not df.empty
    assert df.iloc[0]["symbol"] == "AAPL"
    assert df.iloc[0]["news_volume"] == 2
    mock_client.get_json.assert_called_once()
    params = mock_client.get_json.call_args.kwargs["params"]
    assert params["symbol"] == "AAPL"
    assert params["from"] == "2024-05-01"


@patch("firm.data.providers.finnhub.RestClient")
def test_metric_quarterly_rows(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.get_json.return_value = {
        "series": {
            "quarterly": {
                "pe": [{"period": "2024-03-31", "v": 28.5}],
                "roe": [{"period": "2024-03-31", "v": 0.15}],
            }
        }
    }
    provider = FinnhubProvider(api_key="test", settings=MagicMock(
        request_timeout_seconds=10, max_retries=1,
    ))
    payload = mock_client.get_json.return_value
    rows = _metric_payload_to_rows(
        "AAPL", payload, pd.Timestamp("2020-01-01"), pd.Timestamp("2030-01-01"),
    )
    assert len(rows) == 1
    assert rows[0]["pe_ratio"] == 28.5
    assert rows[0]["roe"] == 0.15

    df = provider.get_fundamentals(["AAPL"], "2020-01-01", "2030-01-01")
    assert not df.empty
    mock_client.get_json.assert_called_once()
