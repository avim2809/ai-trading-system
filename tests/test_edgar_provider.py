"""Tests for SEC EDGAR fundamentals provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.data.providers.edgar import EdgarProvider, _companyfacts_to_rows, _parse_cik_map


@pytest.fixture()
def provider():
    return EdgarProvider(api_key="edgar", settings=MagicMock(
        data=MagicMock(cache_dir="/tmp/test-cache"),
        sec_edgar_user_agent="test-agent test@example.com",
        request_timeout_seconds=10,
        max_retries=1,
    ))


class TestEdgarCikMap:
    def test_parse_cik_map(self):
        raw = {"0": {"cik_str": 320193, "ticker": "AAPL"}}
        assert _parse_cik_map(raw) == {"AAPL": 320193}


class TestEdgarCompanyFacts:
    def test_companyfacts_to_rows_revenue(self):
        payload = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2024-09-28",
                                    "val": 100.0,
                                    "form": "10-K",
                                }
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2024-09-28",
                                    "val": 10.0,
                                    "form": "10-K",
                                }
                            ]
                        }
                    },
                }
            }
        }
        rows = _companyfacts_to_rows(
            "AAPL", payload, pd.Timestamp("2020-01-01"), pd.Timestamp("2030-01-01"),
        )
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["revenue"] == 100.0
        assert rows[0]["net_income"] == 10.0

    @patch.object(EdgarProvider, "_load_cik_map", return_value={"AAPL": 320193})
    @patch("firm.data.providers.edgar.RestClient")
    def test_get_fundamentals_skips_etf(self, mock_client, mock_cik, provider):
        df = provider.get_fundamentals(["SPY"], "2020-01-01", "2030-01-01")
        assert df.empty
        mock_client.return_value.get_json.assert_not_called()
