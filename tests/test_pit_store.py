"""Tests for PointInTimeDataStore – the critical no-look-ahead guarantee."""

from datetime import datetime

import pandas as pd
import pytest

from firm.data.pit_store import PointInTimeDataStore


@pytest.fixture()
def store() -> PointInTimeDataStore:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
            ),
            "symbol": ["AAPL"] * 5,
            "open": [300.0, 301.0, 302.0, 303.0, 304.0],
            "high": [305.0, 306.0, 307.0, 308.0, 309.0],
            "low": [295.0, 296.0, 297.0, 298.0, 299.0],
            "close": [302.0, 303.0, 304.0, 305.0, 306.0],
            "volume": [1e6] * 5,
            "adj_close": [302.0, 303.0, 304.0, 305.0, 306.0],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"]),
            "symbol": ["AAPL"] * 3,
            "market_cap": [1e12, 1.01e12, 1.02e12],
            "pe_ratio": [25.0, 25.5, 26.0],
        }
    )
    s = PointInTimeDataStore()
    s.load(prices, fundamentals)
    return s


class TestGetPrices:
    def test_no_future_data(self, store: PointInTimeDataStore) -> None:
        """get_prices must never return rows beyond the asof date."""
        asof = datetime(2020, 1, 3)
        result = store.get_prices(["AAPL"], asof)
        assert not result.empty
        assert result["date"].max() <= pd.Timestamp(asof)

    def test_lookback_respected(self, store: PointInTimeDataStore) -> None:
        """Only rows within the lookback window should be returned."""
        asof = datetime(2020, 1, 5)
        result = store.get_prices(["AAPL"], asof, lookback_days=2)
        assert result["date"].min() >= pd.Timestamp("2020-01-03")


class TestGetFundamentals:
    def test_latest_as_of(self, store: PointInTimeDataStore) -> None:
        """Only the latest row per symbol as-of asof should be returned."""
        asof = datetime(2020, 1, 4)
        result = store.get_fundamentals(["AAPL"], asof)
        assert len(result) == 1
        assert result.iloc[0]["date"] <= pd.Timestamp(asof)
        assert result.iloc[0]["date"] == pd.Timestamp("2020-01-03")

    def test_no_future_fundamentals(self, store: PointInTimeDataStore) -> None:
        """Fundamentals released after asof must be excluded."""
        asof = datetime(2020, 1, 2)
        result = store.get_fundamentals(["AAPL"], asof)
        assert len(result) == 1
        assert result.iloc[0]["date"] == pd.Timestamp("2020-01-01")


class TestEmptyResult:
    def test_asof_before_any_data(self, store: PointInTimeDataStore) -> None:
        """When asof is before all data, an empty DataFrame must be returned."""
        asof = datetime(2019, 12, 31)
        result = store.get_prices(["AAPL"], asof)
        assert result.empty

    def test_unknown_symbol(self, store: PointInTimeDataStore) -> None:
        """Unknown symbols should yield an empty result, not an error."""
        result = store.get_prices(["ZZZZ"], datetime(2020, 1, 5))
        assert result.empty
