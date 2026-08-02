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

    def test_lookback_is_trading_days_not_calendar(self) -> None:
        """Regression: lookback_days counts trading rows, so a continuous
        business-day feed delivers the full requested window (a calendar-day
        filter would silently truncate ~252 days to ~174)."""
        dates = pd.bdate_range("2020-01-01", periods=300)
        prices = pd.DataFrame({
            "date": dates,
            "symbol": ["AAPL"] * 300,
            "close": range(300),
        })
        s = PointInTimeDataStore()
        s.load(prices)
        result = s.get_prices(["AAPL"], dates[-1], lookback_days=252)
        # Must return exactly 252 trading rows, not ~174 calendar-bounded ones.
        assert len(result) == 252

    def test_universe_resolver_hook(self) -> None:
        """A survivorship-aware resolver, when installed, is authoritative."""
        from firm.data.universe import UniverseResolver

        prices = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAPL", "DELISTED"],
            "close": [1.0, 2.0],
        })
        s = PointInTimeDataStore()
        s.load(prices)
        # Without a resolver: every loaded symbol is returned.
        assert set(s.get_universe(datetime(2020, 1, 2))) == {"AAPL", "DELISTED"}
        # With a resolver excluding DELISTED after removal:
        constituents = pd.DataFrame({
            "symbol": ["AAPL", "DELISTED"],
            "added_date": [pd.NaT, pd.NaT],
            "removed_date": [pd.NaT, pd.Timestamp("2019-06-01")],
        })
        s.set_universe_resolver(UniverseResolver(constituents))
        assert s.get_universe(datetime(2020, 1, 2)) == ["AAPL"]

    def test_get_universe_union_uses_resolver_symbols_between(self) -> None:
        """The feed-loading superset must include a name that joins mid-
        window even though it wasn't a member on the start date."""
        from firm.data.universe import UniverseResolver

        prices = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01"] * 3),
            "symbol": ["AAPL", "LATE_JOINER", "GONE_EARLY"],
            "close": [1.0, 2.0, 3.0],
        })
        s = PointInTimeDataStore()
        s.load(prices)
        constituents = pd.DataFrame({
            "symbol": ["AAPL", "LATE_JOINER", "GONE_EARLY"],
            "added_date": [pd.NaT, pd.Timestamp("2020-06-01"), pd.NaT],
            "removed_date": [pd.NaT, pd.NaT, pd.Timestamp("2020-02-01")],
        })
        s.set_universe_resolver(UniverseResolver(constituents))

        # A single-date snapshot at the start would miss LATE_JOINER and
        # still include GONE_EARLY at the end snapshot would miss it too.
        union = s.get_universe_union(datetime(2020, 1, 1), datetime(2020, 12, 31))
        assert set(union) == {"AAPL", "LATE_JOINER", "GONE_EARLY"}
        assert s.get_universe(datetime(2020, 1, 1)) == ["AAPL", "GONE_EARLY"]

    def test_get_universe_union_degrades_without_symbols_between(self) -> None:
        """A resolver that's just a plain callable (no symbols_between)
        degrades to the union of the start/end snapshots."""
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAPL", "MSFT"],
            "close": [1.0, 2.0],
        })
        s = PointInTimeDataStore()
        s.load(prices)
        s.set_universe_resolver(lambda asof: ["AAPL"] if asof.year == 2020 else ["MSFT"])
        union = s.get_universe_union(datetime(2020, 6, 1), datetime(2021, 6, 1))
        assert set(union) == {"AAPL", "MSFT"}

    def test_get_universe_union_no_resolver_falls_back_to_loaded_prices(self) -> None:
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAPL", "MSFT"],
            "close": [1.0, 2.0],
        })
        s = PointInTimeDataStore()
        s.load(prices)
        union = s.get_universe_union(datetime(2020, 1, 1), datetime(2020, 12, 31))
        assert set(union) == {"AAPL", "MSFT"}


class TestGetFundamentals:
    def test_returns_recent_snapshots_not_just_one(self, store: PointInTimeDataStore) -> None:
        """Regression: used to always collapse to exactly ONE row per
        symbol via groupby("symbol").last(), so event-driven surprise
        detection (which needs >= 2 snapshots to compute anything) could
        never see more than one data point in production and always fell
        through to its fallback. Now returns up to lookback_reports recent
        snapshots, oldest first / newest last."""
        asof = datetime(2020, 1, 4)
        result = store.get_fundamentals(["AAPL"], asof)
        # Both 01-01 and 01-03 are <= asof; 01-05 is excluded (future).
        assert len(result) == 2
        assert list(result["date"]) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")]
        assert (result["date"] <= pd.Timestamp(asof)).all()

    def test_no_future_fundamentals(self, store: PointInTimeDataStore) -> None:
        """Fundamentals released after asof must be excluded."""
        asof = datetime(2020, 1, 2)
        result = store.get_fundamentals(["AAPL"], asof)
        assert len(result) == 1
        assert result.iloc[0]["date"] == pd.Timestamp("2020-01-01")

    def test_lookback_reports_limits_how_far_back(self, store: PointInTimeDataStore) -> None:
        asof = datetime(2020, 1, 10)  # all 3 fundamentals rows are <= asof
        result = store.get_fundamentals(["AAPL"], asof, lookback_reports=2)
        assert len(result) == 2
        assert list(result["date"]) == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-05")]

    def test_default_lookback_reports_is_4(self, store: PointInTimeDataStore) -> None:
        asof = datetime(2020, 1, 10)
        result = store.get_fundamentals(["AAPL"], asof)
        assert len(result) == 3  # only 3 rows exist total, all within the default of 4
        assert result.iloc[-1]["date"] == pd.Timestamp("2020-01-05")  # latest is last


class TestGetEstimates:
    @pytest.fixture()
    def store_with_estimates(self, store: PointInTimeDataStore) -> PointInTimeDataStore:
        estimates = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-06-01", "2025-12-01", "2026-06-01"]),
                "symbol": ["AAPL"] * 3,
                "strong_buy": [4, 5, 6],
                "buy": [20, 22, 23],
                "hold": [12, 15, 17],
                "sell": [1, 1, 2],
                "strong_sell": [0, 0, 2],
            }
        )
        store.load(
            store._prices, store._fundamentals, store._sentiment, estimates=estimates,
        )
        return store

    def test_returns_rows_within_lookback_window(self, store_with_estimates):
        asof = datetime(2026, 7, 1)
        result = store_with_estimates.get_estimates(["AAPL"], asof, lookback_days=365)
        # 2025-06-01 is > 365 days before 2026-07-01; only the other two qualify.
        assert list(result["date"]) == [pd.Timestamp("2025-12-01"), pd.Timestamp("2026-06-01")]

    def test_no_future_estimates(self, store_with_estimates):
        asof = datetime(2025, 12, 15)
        result = store_with_estimates.get_estimates(["AAPL"], asof, lookback_days=365)
        assert list(result["date"]) == [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-12-01")]
        assert (result["date"] <= pd.Timestamp(asof)).all()

    def test_empty_when_not_loaded(self, store: PointInTimeDataStore):
        result = store.get_estimates(["AAPL"], datetime(2026, 1, 1))
        assert result.empty


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


class TestGetMarketPercentilePool:
    """Unlike get_ai_scores/get_best_stocks, this returns a whole
    cross-sectional POPULATION snapshot for a single date (not filtered by
    symbol), picking the latest available date at-or-before asof within the
    lookback window — a real PIT-safety nuance worth testing directly."""

    @pytest.fixture()
    def store_with_pool(self) -> PointInTimeDataStore:
        s = PointInTimeDataStore()
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01"]),
            "symbol": ["AAPL"], "open": [1.0], "high": [1.0], "low": [1.0],
            "close": [1.0], "volume": [1.0], "adj_close": [1.0],
        })
        pool = pd.DataFrame([
            {"date": pd.Timestamp("2020-01-01"), "symbol": "AAPL", "sector": "tech", "ai_score": 8.0},
            {"date": pd.Timestamp("2020-01-01"), "symbol": "MSFT", "sector": "tech", "ai_score": 6.0},
            {"date": pd.Timestamp("2020-01-08"), "symbol": "AAPL", "sector": "tech", "ai_score": 9.0},
            {"date": pd.Timestamp("2020-01-08"), "symbol": "MSFT", "sector": "tech", "ai_score": 5.0},
        ])
        s.load(prices, market_percentile=pool)
        return s

    def test_returns_latest_date_at_or_before_asof(self, store_with_pool: PointInTimeDataStore):
        result = store_with_pool.get_market_percentile_pool(datetime(2020, 1, 8), lookback_days=30)
        assert set(result["date"].unique()) == {pd.Timestamp("2020-01-08")}
        assert set(result["symbol"]) == {"AAPL", "MSFT"}

    def test_does_not_leak_future_dates(self, store_with_pool: PointInTimeDataStore):
        """asof strictly before the later snapshot must never see it (no-look-ahead)."""
        result = store_with_pool.get_market_percentile_pool(datetime(2020, 1, 3), lookback_days=30)
        assert set(result["date"].unique()) == {pd.Timestamp("2020-01-01")}
        assert (result["ai_score"] == 8.0).any()  # the 2020-01-01 AAPL row, not 2020-01-08's 9.0

    def test_not_filtered_by_symbol_returns_whole_population(self, store_with_pool: PointInTimeDataStore):
        """Deliberately NOT restricted to any caller universe — the whole
        point is ranking against the broader population."""
        result = store_with_pool.get_market_percentile_pool(datetime(2020, 1, 8))
        assert len(result) == 2  # both AAPL and MSFT, not just one caller's symbol

    def test_outside_lookback_window_returns_empty(self, store_with_pool: PointInTimeDataStore):
        result = store_with_pool.get_market_percentile_pool(datetime(2020, 6, 1), lookback_days=7)
        assert result.empty

    def test_empty_when_not_loaded(self, store: PointInTimeDataStore):
        result = store.get_market_percentile_pool(datetime(2020, 1, 5))
        assert result.empty
