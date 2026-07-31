"""Tests for DanelfinProvider.get_ai_scores.

Fixture shapes match a real live response captured 2026-07-31 (AAPL) via
apirest.danelfin.com/ranking?ticker=AAPL&page=N — a genuine paid REST API,
not a scraper (see the module docstring in
firm.data.providers.danelfin for the verified auth/endpoint/pagination
details).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.data.providers.base import ProviderError
from firm.data.providers.danelfin import DanelfinProvider


@pytest.fixture()
def provider():
    with patch("firm.data.providers.danelfin.get_settings"):
        return DanelfinProvider(api_key="test-key")


def _page(dates_and_scores: dict) -> dict:
    """Build a fake /ranking page response: {date: {aiscore, technical, fundamental, sentiment, low_risk}}."""
    return dict(dates_and_scores)


class TestDanelfinAiScores:
    def test_maps_real_response_shape_to_ai_score_cols(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            return_value=_page({
                "2026-07-30": {"aiscore": 7, "technical": 5, "fundamental": 7, "sentiment": 5, "low_risk": 6},
                "2026-07-29": {"aiscore": 6, "technical": 5, "fundamental": 7, "sentiment": 5, "low_risk": 6},
            })
        )

        df = provider.get_ai_scores(["AAPL"], "2020-01-01", "2027-01-01")

        assert list(df.columns) == [
            "date", "symbol", "ai_score", "fundamental_score",
            "technical_score", "sentiment_score", "low_risk_score",
        ]
        assert len(df) == 2
        row = df[df["date"] == pd.Timestamp("2026-07-30")].iloc[0]
        assert row["ai_score"] == 7
        assert row["technical_score"] == 5
        assert row["fundamental_score"] == 7
        assert row["sentiment_score"] == 5
        assert row["low_risk_score"] == 6

    def test_null_fundamental_score_preserved_not_crashed(self, provider):
        """Danelfin's own data has real gaps (confirmed live: AAPL 2019 rows
        with fundamental=null) — must not raise, must preserve the null."""
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            return_value=_page({
                "2019-06-14": {"aiscore": 6, "technical": 4, "fundamental": None, "sentiment": 7, "low_risk": 6},
            })
        )
        df = provider.get_ai_scores(["AAPL"], "2019-01-01", "2019-12-31")
        assert pd.isna(df.iloc[0]["fundamental_score"])

    def test_paginates_back_until_reaching_start_date(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            side_effect=[
                _page({"2026-07-01": {"aiscore": 7, "technical": 5, "fundamental": 7, "sentiment": 5, "low_risk": 6}}),
                _page({"2026-01-01": {"aiscore": 6, "technical": 5, "fundamental": 7, "sentiment": 5, "low_risk": 6}}),
                _page({"2025-06-01": {"aiscore": 5, "technical": 5, "fundamental": 7, "sentiment": 5, "low_risk": 6}}),
            ]
        )
        df = provider.get_ai_scores(["AAPL"], "2025-06-01", "2027-01-01")
        assert provider._client.get_json.call_count == 3
        assert len(df) == 3

    def test_pagination_stops_at_max_pages_not_infinite_loop(self, provider):
        """A pathological provider response that never reaches start_ts must
        not loop forever — bounded by _MAX_PAGES."""
        from firm.data.providers import danelfin as danelfin_module

        provider._client = MagicMock()
        # Every page returns the SAME date -> min date never decreases ->
        # the "no progress" break should trigger well before _MAX_PAGES.
        provider._client.get_json = MagicMock(
            return_value=_page({"2026-07-01": {"aiscore": 7, "technical": 5, "fundamental": 7, "sentiment": 5, "low_risk": 6}})
        )
        with patch.object(danelfin_module.time, "sleep"):
            df = provider.get_ai_scores(["AAPL"], "2000-01-01", "2027-01-01")
        assert provider._client.get_json.call_count < danelfin_module._MAX_PAGES
        assert len(df) == 1

    def test_transient_message_only_page_retried_once_then_skipped(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            side_effect=[
                {"message": "internal error"},
                {"message": "internal error"},  # retry also fails
            ]
        )
        with patch("firm.data.providers.danelfin.time.sleep"):
            df = provider.get_ai_scores(["AAPL"], "2020-01-01", "2027-01-01")
        assert df.empty

    def test_date_range_filter_excludes_rows_outside_window(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            return_value=_page({
                "2026-07-30": {"aiscore": 7, "technical": 5, "fundamental": 7, "sentiment": 5, "low_risk": 6},
                "2018-01-01": {"aiscore": 5, "technical": 5, "fundamental": 5, "sentiment": 5, "low_risk": 5},
            })
        )
        df = provider.get_ai_scores(["AAPL"], "2020-01-01", "2027-01-01")
        assert len(df) == 1
        assert df.iloc[0]["date"] == pd.Timestamp("2026-07-30")

    def test_empty_response_returns_typed_empty_frame(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(return_value={})
        df = provider.get_ai_scores(["AAPL"], "2020-01-01", "2026-01-01")
        assert df.empty
        assert list(df.columns) == [
            "date", "symbol", "ai_score", "fundamental_score",
            "technical_score", "sentiment_score", "low_risk_score",
        ]

    def test_one_bad_symbol_does_not_fail_whole_batch(self, provider):
        good_page = _page({"2026-07-30": {"aiscore": 7, "technical": 5, "fundamental": 7, "sentiment": 5, "low_risk": 6}})
        provider._client = MagicMock()

        def side_effect(path, params, headers):
            if params["ticker"] == "BADSYM":
                raise ProviderError("boom")
            return good_page

        provider._client.get_json = MagicMock(side_effect=side_effect)
        df = provider.get_ai_scores(["BADSYM", "AAPL"], "2020-01-01", "2027-01-01")
        assert not df.empty
        assert set(df["symbol"]) == {"AAPL"}

    def test_uses_x_api_key_header(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(return_value={})
        provider.get_ai_scores(["AAPL"], "2020-01-01", "2026-01-01")
        _, kwargs = provider._client.get_json.call_args
        assert kwargs["headers"] == {"x-api-key": "test-key"}


class TestDanelfinTradeIdeas:
    """Real response shape verified live 2026-07-31: {date: {symbol: {...}}},
    NOT {"items": [...]} — the original implementation assumed the latter
    and silently always returned empty; see get_trade_ideas's docstring."""

    def test_maps_real_response_shape(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            return_value={
                "2026-07-30": {
                    "VIST": {
                        "aiscore": 9, "low_risk": 5, "average_volume_3m": 1017627,
                        "sector": "energy", "win_rate_3m": 0.94,
                    },
                    "HWM": {
                        "aiscore": 7, "low_risk": 6, "average_volume_3m": 2785673,
                        "sector": "industrials", "win_rate_3m": 0.94,
                    },
                }
            }
        )
        df = provider.get_trade_ideas(sector="energy", aiscore=1, low_risk=5, limit=100)
        assert len(df) == 2
        assert set(df["symbol"]) == {"VIST", "HWM"}
        assert df[df["symbol"] == "VIST"].iloc[0]["sector"] == "energy"
        assert df[df["symbol"] == "VIST"].iloc[0]["aiscore"] == 9

    def test_empty_response_returns_empty_frame(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(return_value={})
        df = provider.get_trade_ideas(sector="real-estate")
        assert df.empty

    def test_sibling_metadata_keys_are_skipped_not_iterated(self, provider):
        """Real bug found live: the response carries sibling total/limit/
        offset int keys alongside the date key — iterating them as if they
        were {symbol: {...}} dicts raised AttributeError('int' has no
        attribute 'items')."""
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(
            return_value={
                "2026-07-30": {"TSM": {"aiscore": 8, "sector": "information-technology"}},
                "total": 42,
                "limit": 100,
                "offset": 0,
            }
        )
        df = provider.get_trade_ideas(sector="information-technology", limit=100)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "TSM"

    def test_message_only_response_returns_empty_frame(self, provider):
        provider._client = MagicMock()
        provider._client.get_json = MagicMock(return_value={"message": "no results"})
        df = provider.get_trade_ideas(sector="utilities")
        assert df.empty


class TestDanelfinUnsupportedCapabilities:
    def test_other_capabilities_raise_not_implemented(self, provider):
        with pytest.raises(NotImplementedError):
            provider.get_prices(["AAPL"], "2020-01-01", "2026-01-01")
        with pytest.raises(NotImplementedError):
            provider.get_fundamentals(["AAPL"], "2020-01-01", "2026-01-01")
        with pytest.raises(NotImplementedError):
            provider.get_news_sentiment(["AAPL"], "2020-01-01", "2026-01-01")
        with pytest.raises(NotImplementedError):
            provider.get_corporate_actions(["AAPL"], "2020-01-01", "2026-01-01")
        with pytest.raises(NotImplementedError):
            provider.get_universe_constituents("sp500")
        with pytest.raises(NotImplementedError):
            provider.get_analyst_ratings(["AAPL"], "2020-01-01", "2026-01-01")
