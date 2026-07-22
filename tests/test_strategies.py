"""Comprehensive tests for all 10 alpha strategy modules.

Tests validate that every strategy:
- Returns a list of Signal objects with valid fields
- Handles empty universes and missing data gracefully
- Produces scores in a reasonable range
- Is discoverable via the strategy registry

A shared ``MockPitView`` provides synthetic price, fundamental, and
sentiment data so tests run without any external dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy

# Force all strategy modules to register themselves
import firm.strategies  # noqa: F401
from firm.strategies.registry import get, list_strategies


def test_multi_factor_mean_available_no_dampening():
    """Regression: a symbol with only one sub-metric keeps its full z-score
    rather than being halved by a fixed /2 divisor."""
    from firm.strategies.multi_factor import _mean_available

    pe = pd.Series({"A": 1.0, "B": 2.0})       # both symbols have PE
    pb = pd.Series({"A": 0.5})                  # only A also has PB
    result = _mean_available([pe, pb])
    assert result["A"] == pytest.approx((1.0 + 0.5) / 2)  # averaged
    assert result["B"] == pytest.approx(2.0)              # NOT 2.0 / 2


# ---------------------------------------------------------------------------
# Mock PitView
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]


def _make_price_df(
    symbols: list[str],
    n_days: int = 300,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for *symbols* over *n_days*."""
    if end_date is None:
        end_date = datetime(2025, 6, 1)
    rng = np.random.RandomState(42)
    rows: list[dict] = []
    dates = pd.bdate_range(end=end_date, periods=n_days)
    for sym_idx, sym in enumerate(symbols):
        price = 100.0 + sym_idx * 20
        for d in dates:
            ret = rng.normal(0.0005, 0.02)
            price *= 1 + ret
            high = price * (1 + abs(rng.normal(0, 0.005)))
            low = price * (1 - abs(rng.normal(0, 0.005)))
            volume = int(rng.uniform(1e6, 1e7))
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": price * (1 + rng.normal(0, 0.002)),
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": volume,
                    "adj_close": price,
                }
            )
    return pd.DataFrame(rows)


def _make_fundamental_df(
    symbols: list[str],
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Generate synthetic fundamental data for *symbols*."""
    if end_date is None:
        end_date = datetime(2025, 6, 1)
    rng = np.random.RandomState(99)
    rows: list[dict] = []
    # Two quarterly snapshots
    for offset_days in [90, 0]:
        d = end_date - timedelta(days=offset_days)
        for sym in symbols:
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "market_cap": rng.uniform(1e10, 3e12),
                    "pe_ratio": rng.uniform(5, 60),
                    "pb_ratio": rng.uniform(0.5, 15),
                    "roe": rng.uniform(0.02, 0.40),
                    "debt_to_equity": rng.uniform(0.1, 3.0),
                    "revenue": rng.uniform(1e9, 4e11),
                    "net_income": rng.uniform(1e8, 8e10),
                    "eps": rng.uniform(0.5, 20.0),
                    "dividend_yield": rng.uniform(0.0, 0.05),
                }
            )
    return pd.DataFrame(rows)


def _make_sentiment_df(
    symbols: list[str],
    lookback_days: int = 10,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Generate synthetic sentiment data for *symbols*."""
    if end_date is None:
        end_date = datetime(2025, 6, 1)
    rng = np.random.RandomState(77)
    rows: list[dict] = []
    for offset in range(lookback_days):
        d = end_date - timedelta(days=offset)
        for sym in symbols:
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "sentiment_score": rng.uniform(-1, 1),
                    "news_volume": rng.randint(1, 50),
                    "source": "synthetic",
                    "headline": f"Test headline for {sym}",
                }
            )
    return pd.DataFrame(rows)


class MockPitView:
    """In-memory PitView implementation for testing."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        asof: datetime | None = None,
        n_price_days: int = 300,
        include_fundamentals: bool = True,
        include_sentiment: bool = True,
    ):
        self._symbols = symbols or SYMBOLS
        self._asof = asof or datetime(2025, 6, 1)
        self._price_df = _make_price_df(self._symbols, n_price_days, self._asof)
        self._fund_df = (
            _make_fundamental_df(self._symbols, self._asof)
            if include_fundamentals
            else pd.DataFrame()
        )
        self._sent_df = (
            _make_sentiment_df(self._symbols, 10, self._asof)
            if include_sentiment
            else pd.DataFrame()
        )

    @property
    def asof(self) -> datetime:
        return self._asof

    @property
    def universe(self) -> list[str]:
        return list(self._symbols)

    def prices(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        df = self._price_df.copy()
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        cutoff = pd.Timestamp(self._asof) - pd.Timedelta(days=lookback_days)
        df = df[df["date"] >= cutoff]
        return df

    def fundamentals(
        self,
        symbols: list[str] | None = None,
        lookback_reports: int = 4,
    ) -> pd.DataFrame:
        df = self._fund_df.copy()
        if symbols and not df.empty:
            df = df[df["symbol"].isin(symbols)]
        if not df.empty:
            df = (
                df.sort_values("date")
                .groupby("symbol", group_keys=False)
                .tail(lookback_reports)
                .reset_index(drop=True)
            )
        return df

    def sentiment(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        df = self._sent_df.copy()
        if symbols and not df.empty:
            df = df[df["symbol"].isin(symbols)]
        if not df.empty:
            cutoff = pd.Timestamp(self._asof) - pd.Timedelta(days=lookback_days)
            df = df[pd.to_datetime(df["date"]) >= cutoff]
        return df


class EmptyPitView:
    """PitView that returns empty data for edge-case testing."""

    @property
    def asof(self) -> datetime:
        return datetime(2025, 6, 1)

    @property
    def universe(self) -> list[str]:
        return []

    def prices(self, symbols=None, lookback_days=252) -> pd.DataFrame:
        return pd.DataFrame()

    def fundamentals(self, symbols=None, lookback_reports=4) -> pd.DataFrame:
        return pd.DataFrame()

    def sentiment(self, symbols=None, lookback_days=5) -> pd.DataFrame:
        return pd.DataFrame()


class SingleSymbolPitView:
    """PitView with a single symbol for edge-case testing."""

    def __init__(self):
        self._sym = ["AAPL"]
        self._asof = datetime(2025, 6, 1)
        self._price_df = _make_price_df(self._sym, 300, self._asof)
        self._fund_df = _make_fundamental_df(self._sym, self._asof)
        self._sent_df = _make_sentiment_df(self._sym, 10, self._asof)

    @property
    def asof(self) -> datetime:
        return self._asof

    @property
    def universe(self) -> list[str]:
        return list(self._sym)

    def prices(self, symbols=None, lookback_days=252) -> pd.DataFrame:
        df = self._price_df.copy()
        cutoff = pd.Timestamp(self._asof) - pd.Timedelta(days=lookback_days)
        return df[df["date"] >= cutoff]

    def fundamentals(self, symbols=None, lookback_reports=4) -> pd.DataFrame:
        return self._fund_df.copy()

    def sentiment(self, symbols=None, lookback_days=5) -> pd.DataFrame:
        df = self._sent_df.copy()
        cutoff = pd.Timestamp(self._asof) - pd.Timedelta(days=lookback_days)
        return df[pd.to_datetime(df["date"]) >= cutoff]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pit_view() -> MockPitView:
    return MockPitView()


@pytest.fixture
def empty_view() -> EmptyPitView:
    return EmptyPitView()


@pytest.fixture
def single_symbol_view() -> SingleSymbolPitView:
    return SingleSymbolPitView()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

ALL_STRATEGY_NAMES = [
    "momentum",
    "trend",
    "mean_reversion",
    "stat_arb",
    "multi_factor",
    "sentiment",
    "event_driven",
    "ml_prediction",
    "volatility_breakout",
    "seasonality",
    "gann",
]


class TestRegistry:
    def test_all_10_strategies_registered(self):
        registered = list_strategies()
        for name in ALL_STRATEGY_NAMES:
            assert name in registered, f"Strategy '{name}' not in registry"

    def test_registry_count(self):
        assert len(list_strategies()) >= 11

    def test_get_returns_base_subclass(self):
        for name in ALL_STRATEGY_NAMES:
            cls = get(name)
            assert issubclass(cls, BaseStrategy)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get("nonexistent_strategy")

    def test_instantiation(self):
        for name in ALL_STRATEGY_NAMES:
            cls = get(name)
            instance = cls()
            assert instance.name == name


# ---------------------------------------------------------------------------
# Helpers for signal validation
# ---------------------------------------------------------------------------


def _validate_signals(signals: list[Signal], strategy_name: str) -> None:
    """Assert every signal has valid fields."""
    assert isinstance(signals, list)
    for sig in signals:
        assert isinstance(sig, Signal), f"Expected Signal, got {type(sig)}"
        assert isinstance(sig.symbol, str) and len(sig.symbol) > 0
        assert sig.strategy == strategy_name
        assert isinstance(sig.score, float)
        assert -10 <= sig.score <= 10, f"Score {sig.score} out of range"
        assert 0 <= sig.confidence <= 1.0, f"Confidence {sig.confidence} out of range"
        assert isinstance(sig.horizon, str) and len(sig.horizon) > 0
        assert sig.asof is not None
        assert isinstance(sig.meta, dict)


# ---------------------------------------------------------------------------
# Per-strategy tests
# ---------------------------------------------------------------------------


class TestMomentum:
    def test_generate(self, pit_view):
        strat = get("momentum")()
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "momentum")

    def test_empty_universe(self, empty_view):
        strat = get("momentum")()
        assert strat.generate(empty_view) == []

    def test_custom_params(self, pit_view):
        strat = get("momentum")(params={"lookback_months": 6, "skip_months": 1})
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "momentum")


class TestTrend:
    def test_generate(self, pit_view):
        strat = get("trend")(params={"fast_window": 20, "slow_window": 50})
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "trend")

    def test_empty_universe(self, empty_view):
        strat = get("trend")()
        assert strat.generate(empty_view) == []

    def test_directions_are_valid(self, pit_view):
        strat = get("trend")(params={"fast_window": 20, "slow_window": 50})
        signals = strat.generate(pit_view)
        for sig in signals:
            assert sig.meta["direction"] in (-1.0, 0.0, 1.0)


class TestMeanReversion:
    def test_generate(self, pit_view):
        strat = get("mean_reversion")()
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "mean_reversion")

    def test_empty_universe(self, empty_view):
        strat = get("mean_reversion")()
        assert strat.generate(empty_view) == []

    def test_zscore_cap_respected(self, pit_view):
        cap = 2.0
        strat = get("mean_reversion")(params={"zscore_cap": cap})
        signals = strat.generate(pit_view)
        for sig in signals:
            assert abs(sig.score) <= cap + 1e-9


class TestStatArb:
    def test_generate(self, pit_view):
        strat = get("stat_arb")(params={"max_pairs": 3})
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "stat_arb")

    def test_empty_universe(self, empty_view):
        strat = get("stat_arb")()
        assert strat.generate(empty_view) == []

    def test_pairs_produce_two_legs(self, pit_view):
        strat = get("stat_arb")(params={"max_pairs": 1})
        signals = strat.generate(pit_view)
        # Each pair produces exactly 2 signals
        assert len(signals) % 2 == 0

    def test_single_symbol(self, single_symbol_view):
        strat = get("stat_arb")()
        signals = strat.generate(single_symbol_view)
        assert signals == []


class TestMultiFactor:
    def test_generate(self, pit_view):
        strat = get("multi_factor")()
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "multi_factor")

    def test_empty_universe(self, empty_view):
        strat = get("multi_factor")()
        assert strat.generate(empty_view) == []

    def test_custom_weights(self, pit_view):
        strat = get("multi_factor")(
            params={"factor_weights": {"value": 0.5, "momentum": 0.5}}
        )
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "multi_factor")

    def test_zero_price_at_momentum_window_start_does_not_nan_the_universe(self, pit_view):
        """Regression: a zero/bad price at the start of the momentum window
        used to produce +/-inf, which _zscore's old dropna()-only cleanup
        didn't strip — an inf in the series makes std() == inf, slipping
        past the "std==0 or NaN" guard, so EVERY symbol's momentum z-score
        (not just the corrupted one) came out NaN."""
        strat = get("multi_factor")()
        lookback = max(252, 12 * 21) + 10
        filtered = pit_view.prices(lookback_days=lookback)
        target_sym = filtered["symbol"].iloc[0]
        earliest_date = filtered[filtered["symbol"] == target_sym]["date"].min()

        corrupted = pit_view._price_df.copy()
        mask = (corrupted["symbol"] == target_sym) & (corrupted["date"] == earliest_date)
        corrupted.loc[mask, "adj_close"] = 0.0
        pit_view._price_df = corrupted

        signals = strat.generate(pit_view)
        assert len(signals) > 0
        for sig in signals:
            assert np.isfinite(sig.score), f"{sig.symbol} got a non-finite score"


class TestMultiFactorZscoreHardening:
    def test_zscore_strips_inf_before_computing_stats(self):
        from firm.strategies.multi_factor import _zscore

        s = pd.Series({"A": 1.0, "B": 2.0, "C": np.inf, "D": -np.inf})
        result = _zscore(s)

        assert set(result.index) == {"A", "B"}
        assert not result.isna().any()
        assert np.isfinite(result).all()


class TestMultiFactorWeightedComposite:
    """Regression: combining per-factor scores used to divide every symbol
    by the SAME total factor weight regardless of which factors that symbol
    actually had a value for — silently diluting a symbol contributing to
    only 1 of 4 factors by weight it never earned."""

    def test_partial_coverage_symbol_is_not_diluted_by_missing_factors(self):
        from firm.strategies.multi_factor import _weighted_composite

        scores = {
            "value": pd.Series({"A": 2.0, "B": 2.0}),
            "quality": pd.Series({"A": 2.0}),
            "momentum": pd.Series({"A": 2.0}),
            "low_vol": pd.Series({"A": 2.0}),
        }
        weights = {"value": 0.25, "quality": 0.25, "momentum": 0.25, "low_vol": 0.25}

        result = _weighted_composite(scores, weights)

        # A has all 4 factors at 2.0 -> composite 2.0. B has only "value" at
        # 2.0 -> composite must also be 2.0 (its own factor's value,
        # unchanged), NOT 2.0 * 0.25 / 1.0 = 0.5 (diluted by weights for
        # factors it never had).
        assert result["A"] == pytest.approx(2.0)
        assert result["B"] == pytest.approx(2.0)

    def test_empty_scores_returns_empty(self):
        from firm.strategies.multi_factor import _weighted_composite

        result = _weighted_composite({}, {"value": 1.0})
        assert result.empty


class TestSentiment:
    def test_generate(self, pit_view):
        strat = get("sentiment")()
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "sentiment")

    def test_empty_universe(self, empty_view):
        strat = get("sentiment")()
        assert strat.generate(empty_view) == []

    def test_no_sentiment_data(self):
        view = MockPitView(include_sentiment=False)
        strat = get("sentiment")()
        assert strat.generate(view) == []

    def test_fewer_than_3_symbols_emits_nothing_not_a_raw_score(self):
        """Regression: used to fall back to a raw [-1,1] score for <2
        symbols while normally emitting a z-score clipped to [-3,3] — two
        incompatible scales from the same strategy depending on how many
        symbols happened to have data that day."""
        view = MockPitView(symbols=["AAPL", "MSFT"])
        strat = get("sentiment")()
        assert strat.generate(view) == []

    def test_news_volume_weighting_changes_the_aggregate(self):
        """Regression: the docstring promises volume-weighted sentiment,
        but the code used to take a plain unweighted mean — a symbol with
        one wildly positive article scored identically to one with 100
        moderately positive articles on the same day."""
        view = MockPitView(symbols=["AAPL", "MSFT", "GOOG"])
        asof = pd.Timestamp(view.asof)
        # AAPL: one mildly-positive article with huge volume, one wildly
        # positive article with volume=1 -> volume-weighted mean ~0.108,
        # far from the naive unweighted mean of 0.5.
        rows = [
            {"date": asof, "symbol": "AAPL", "sentiment_score": 0.1, "news_volume": 100,
             "source": "t", "headline": "h1"},
            {"date": asof, "symbol": "AAPL", "sentiment_score": 0.9, "news_volume": 1,
             "source": "t", "headline": "h2"},
            {"date": asof, "symbol": "MSFT", "sentiment_score": -0.3, "news_volume": 5,
             "source": "t", "headline": "h3"},
            {"date": asof, "symbol": "GOOG", "sentiment_score": 0.4, "news_volume": 5,
             "source": "t", "headline": "h4"},
        ]
        view._sent_df = pd.DataFrame(rows)
        strat = get("sentiment")()

        unweighted_mean = (0.1 + 0.9) / 2
        weighted_mean = (0.1 * 100 + 0.9 * 1) / 101

        signals = strat.generate(view)
        aapl = next(s for s in signals if s.symbol == "AAPL")

        assert aapl.meta["sentiment_level"] == pytest.approx(weighted_mean, abs=1e-6)
        assert aapl.meta["sentiment_level"] != pytest.approx(unweighted_mean, abs=0.05)

    def test_delta_uses_lookback_days_not_the_full_fetch_buffer(self):
        """Regression: delta used the FIRST row of the whole fetched buffer
        (lookback_days + 5) as "old", not a row ~lookback_days before asof —
        the effective window silently drifted with data sparsity instead of
        tracking the lookback_days parameter."""
        view = MockPitView(symbols=["AAPL", "MSFT", "GOOG"])
        asof = pd.Timestamp(view.asof)
        rows = []
        # 10 days of flat sentiment per symbol (days -10..-1), then a jump
        # on the asof day itself for AAPL only. lookback_days default is 5,
        # so "old" should land on day -5, well after the buffer's actual
        # earliest day (-9 once fetched with the +5 padding).
        for offset in range(10, 0, -1):
            d = asof - pd.Timedelta(days=offset)
            rows.append({"date": d, "symbol": "AAPL", "sentiment_score": 0.0, "news_volume": 5,
                         "source": "t", "headline": "h"})
            rows.append({"date": d, "symbol": "MSFT", "sentiment_score": 0.1, "news_volume": 5,
                         "source": "t", "headline": "h"})
            rows.append({"date": d, "symbol": "GOOG", "sentiment_score": -0.1, "news_volume": 5,
                         "source": "t", "headline": "h"})
        # asof day: AAPL jumps to 0.8, others stay flat.
        rows.append({"date": asof, "symbol": "AAPL", "sentiment_score": 0.8, "news_volume": 5,
                     "source": "t", "headline": "jump"})
        rows.append({"date": asof, "symbol": "MSFT", "sentiment_score": 0.1, "news_volume": 5,
                     "source": "t", "headline": "h"})
        rows.append({"date": asof, "symbol": "GOOG", "sentiment_score": -0.1, "news_volume": 5,
                     "source": "t", "headline": "h"})
        view._sent_df = pd.DataFrame(rows)
        strat = get("sentiment")()

        signals = strat.generate(view)
        aapl = next(s for s in signals if s.symbol == "AAPL")
        # "old" (~5 days back) was flat at 0.0, so delta should be ~0.8,
        # not diluted by averaging against the buffer's much-older days.
        assert aapl.meta["sentiment_delta"] == pytest.approx(0.8, abs=0.05)


class TestEventDriven:
    def test_generate(self, pit_view):
        strat = get("event_driven")()
        signals = strat.generate(pit_view)
        # Event-driven may or may not find events in synthetic data
        _validate_signals(signals, "event_driven")

    def test_empty_universe(self, empty_view):
        strat = get("event_driven")()
        assert strat.generate(empty_view) == []

    def test_with_clear_surprise(self):
        """Construct data with an obvious earnings surprise."""
        asof = datetime(2025, 6, 1)
        symbols = ["TEST"]
        fund_rows = [
            {"date": asof - timedelta(days=90), "symbol": "TEST",
             "market_cap": 1e10, "pe_ratio": 15, "pb_ratio": 3,
             "roe": 0.15, "debt_to_equity": 0.5, "revenue": 1e9,
             "net_income": 1e8, "eps": 1.0, "dividend_yield": 0.02},
            {"date": asof, "symbol": "TEST",
             "market_cap": 1e10, "pe_ratio": 12, "pb_ratio": 3,
             "roe": 0.18, "debt_to_equity": 0.5, "revenue": 1.2e9,
             "net_income": 1.5e8, "eps": 1.50, "dividend_yield": 0.02},
        ]

        class SurpriseView:
            @property
            def asof(self):
                return asof

            @property
            def universe(self):
                return symbols

            def prices(self, symbols=None, lookback_days=252):
                return _make_price_df(["TEST"], 60, asof)

            def fundamentals(self, symbols=None):
                return pd.DataFrame(fund_rows)

            def sentiment(self, symbols=None, lookback_days=5):
                return pd.DataFrame()

        strat = get("event_driven")()
        signals = strat.generate(SurpriseView())
        assert len(signals) == 1
        assert signals[0].score > 0  # positive surprise → bullish
        _validate_signals(signals, "event_driven")

    def test_real_production_path_detects_earnings_surprise(self):
        """Regression for a critical bug: PointInTimeDataStore.get_
        fundamentals() used to always collapse to exactly ONE row per
        symbol (groupby("symbol").last()), so _signals_from_fundamentals'
        own "need >= 2 snapshots to compute a surprise" check could never
        pass in production (only in this file's hand-rolled SurpriseView
        mock above, which never had the bug since it returns raw rows
        directly, bypassing the real PIT store entirely). event_driven had
        never actually detected a real earnings surprise via the real
        backtest/live adapters — it silently always fell back to the
        price-move proxy. This test goes through the REAL
        PointInTimeDataStore + PitViewAdapter, not a shortcut mock."""
        from firm.backtest.firm_strategy import PitViewAdapter
        from firm.data.pit_store import PointInTimeDataStore

        asof = datetime(2025, 6, 1)
        fund_df = pd.DataFrame([
            {"date": pd.Timestamp(asof) - pd.Timedelta(days=90), "symbol": "TEST", "eps": 1.0,
             "market_cap": 1e10, "pe_ratio": 15, "pb_ratio": 3, "roe": 0.15,
             "debt_to_equity": 0.5, "revenue": 1e9, "net_income": 1e8, "dividend_yield": 0.02},
            {"date": pd.Timestamp(asof), "symbol": "TEST", "eps": 1.5,
             "market_cap": 1e10, "pe_ratio": 12, "pb_ratio": 3, "roe": 0.18,
             "debt_to_equity": 0.5, "revenue": 1.2e9, "net_income": 1.5e8, "dividend_yield": 0.02},
        ])
        prices_df = pd.DataFrame([{
            "date": pd.Timestamp(asof), "symbol": "TEST",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 1000.0, "adj_close": 100.0,
        }])

        store = PointInTimeDataStore()
        store.load(prices=prices_df, fundamentals=fund_df)
        pit_view = PitViewAdapter(store, asof, ["TEST"])

        strat = get("event_driven")()
        signals = strat.generate(pit_view)

        assert len(signals) == 1
        assert signals[0].score > 0
        assert signals[0].meta["earnings_surprise"] == pytest.approx(0.5)
        assert "proxy_event" not in signals[0].meta  # real surprise, not the price-move fallback


class TestMLPrediction:
    def test_generate(self):
        view = MockPitView(n_price_days=600)
        strat = get("ml_prediction")(
            params={"train_lookback_days": 300, "model_type": "ridge"}
        )
        signals = strat.generate(view)
        assert len(signals) > 0
        _validate_signals(signals, "ml_prediction")

    def test_empty_universe(self, empty_view):
        strat = get("ml_prediction")()
        assert strat.generate(empty_view) == []

    def test_gbr_model(self):
        view = MockPitView(n_price_days=600)
        strat = get("ml_prediction")(
            params={"train_lookback_days": 300, "model_type": "gbr"}
        )
        signals = strat.generate(view)
        assert len(signals) > 0
        _validate_signals(signals, "ml_prediction")
        assert signals[0].meta["model_type"] == "gbr"

    def test_insufficient_data(self):
        view = MockPitView(n_price_days=30)
        strat = get("ml_prediction")()
        signals = strat.generate(view)
        assert signals == []


class TestVolatilityBreakout:
    def test_generate(self, pit_view):
        strat = get("volatility_breakout")(params={"vol_threshold": 999.0})
        signals = strat.generate(pit_view)
        _validate_signals(signals, "volatility_breakout")

    def test_empty_universe(self, empty_view):
        strat = get("volatility_breakout")()
        assert strat.generate(empty_view) == []

    def test_score_range(self, pit_view):
        strat = get("volatility_breakout")(params={"vol_threshold": 999.0})
        signals = strat.generate(pit_view)
        for sig in signals:
            assert -1.0 <= sig.score <= 1.0


class TestSeasonality:
    def test_generate(self, pit_view):
        strat = get("seasonality")()
        signals = strat.generate(pit_view)
        assert len(signals) == len(pit_view.universe)
        _validate_signals(signals, "seasonality")

    def test_empty_universe(self, empty_view):
        strat = get("seasonality")()
        assert strat.generate(empty_view) == []

    def test_tom_score_first_of_month(self):
        asof = datetime(2025, 6, 1)  # first day of month
        view = MockPitView(asof=asof)
        strat = get("seasonality")(params={"tom_days_after": 3})
        signals = strat.generate(view)
        assert len(signals) > 0
        # tom_score should be 1.0 for day 1
        assert signals[0].meta["tom_score"] == 1.0

    def test_tom_score_mid_month(self):
        asof = datetime(2025, 6, 15)  # mid-month
        view = MockPitView(asof=asof)
        strat = get("seasonality")(params={"tom_days_before": 1, "tom_days_after": 3})
        signals = strat.generate(view)
        assert len(signals) > 0
        assert signals[0].meta["tom_score"] == 0.0

    def test_uniform_scores(self, pit_view):
        strat = get("seasonality")()
        signals = strat.generate(pit_view)
        scores = {s.score for s in signals}
        assert len(scores) == 1  # all symbols get the same score


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------


class TestGann:
    def test_generate(self, pit_view):
        strat = get("gann")()
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "gann")

    def test_empty_universe(self, empty_view):
        strat = get("gann")()
        assert strat.generate(empty_view) == []

    def test_single_symbol(self, single_symbol_view):
        strat = get("gann")()
        signals = strat.generate(single_symbol_view)
        # Single symbol should still produce a signal (raw, not z-scored)
        assert len(signals) == 1
        _validate_signals(signals, "gann")

    def test_score_range(self, pit_view):
        strat = get("gann")()
        signals = strat.generate(pit_view)
        for sig in signals:
            assert -3.0 <= sig.score <= 3.0, f"Score {sig.score} out of [-3, 3]"

    def test_meta_contains_sub_scores(self, pit_view):
        strat = get("gann")()
        signals = strat.generate(pit_view)
        for sig in signals:
            for key in ("angle_score", "sq9_score", "cycle_score",
                        "swing_score", "retracement_score", "atr",
                        "trend_strength", "trend_dampener"):
                assert key in sig.meta, f"Missing meta key '{key}'"

    def test_custom_sub_weights(self, pit_view):
        strat = get("gann")(params={
            "sub_weights": {"angles": 1.0, "sq9": 0.0, "cycles": 0.0,
                            "swing": 0.0, "retracement": 0.0}
        })
        signals = strat.generate(pit_view)
        assert len(signals) > 0
        _validate_signals(signals, "gann")

    def test_sub_scores_bounded(self, pit_view):
        strat = get("gann")()
        signals = strat.generate(pit_view)
        for sig in signals:
            for key in ("angle_score", "sq9_score", "cycle_score",
                        "swing_score", "retracement_score"):
                v = sig.meta[key]
                assert -1.0 <= v <= 1.0, f"Sub-score {key}={v} out of [-1, 1]"

    def test_trend_filter_dampens_choppy_markets(self):
        """When trend strength is low, scores should be dampened."""
        strat_strict = get("gann")(params={"trend_filter_threshold": 0.99})
        strat_loose = get("gann")(params={"trend_filter_threshold": 0.01})
        view = MockPitView()
        signals_strict = strat_strict.generate(view)
        signals_loose = strat_loose.generate(view)
        if signals_strict and signals_loose:
            avg_conf_strict = sum(s.confidence for s in signals_strict) / len(signals_strict)
            avg_conf_loose = sum(s.confidence for s in signals_loose) / len(signals_loose)
            assert avg_conf_strict <= avg_conf_loose + 0.01

    def test_confidence_floor_filters_weak_signals(self):
        """High min_confidence should filter out low-conviction symbols."""
        strat_low = get("gann")(params={"min_confidence": 0.0})
        strat_high = get("gann")(params={"min_confidence": 0.99})
        view = MockPitView()
        signals_low = strat_low.generate(view)
        signals_high = strat_high.generate(view)
        assert len(signals_high) <= len(signals_low)

    def test_trend_strength_in_meta(self, pit_view):
        strat = get("gann")()
        signals = strat.generate(pit_view)
        for sig in signals:
            ts = sig.meta["trend_strength"]
            assert 0.0 <= ts <= 1.0, f"trend_strength {ts} out of [0, 1]"
            td = sig.meta["trend_dampener"]
            assert 0.0 <= td <= 1.0, f"trend_dampener {td} out of [0, 1]"

    def test_flat_prices_no_crash(self):
        """Flat prices (zero vol) should not crash."""
        asof = datetime(2025, 6, 1)
        syms = ["FLAT1", "FLAT2", "FLAT3"]
        dates = pd.bdate_range(end=asof, periods=200)
        rows = []
        for sym in syms:
            for d in dates:
                rows.append({
                    "date": d, "symbol": sym,
                    "open": 100.0, "high": 100.0, "low": 100.0,
                    "close": 100.0, "volume": 1000000, "adj_close": 100.0,
                })

        class FlatView:
            @property
            def asof(self): return asof

            @property
            def universe(self): return syms

            def prices(self, symbols=None, lookback_days=252):
                df = pd.DataFrame(rows)
                if symbols:
                    df = df[df["symbol"].isin(symbols)]
                cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days)
                return df[df["date"] >= cutoff]

            def fundamentals(self, symbols=None):
                return pd.DataFrame()

            def sentiment(self, symbols=None, lookback_days=5):
                return pd.DataFrame()

        strat = get("gann")()
        signals = strat.generate(FlatView())
        _validate_signals(signals, "gann")


class TestAllStrategies:
    """Generic tests applied to every registered strategy."""

    @pytest.mark.parametrize("name", ALL_STRATEGY_NAMES)
    def test_returns_list_of_signals(self, name, pit_view):
        if name == "ml_prediction":
            view = MockPitView(n_price_days=600)
        else:
            view = pit_view
        cls = get(name)
        params = {}
        if name == "trend":
            params = {"fast_window": 20, "slow_window": 50}
        if name == "ml_prediction":
            params = {"train_lookback_days": 300, "model_type": "ridge"}
        if name == "volatility_breakout":
            params = {"vol_threshold": 999.0}
        strat = cls(params=params)
        signals = strat.generate(view)
        assert isinstance(signals, list)
        for sig in signals:
            assert isinstance(sig, Signal)

    @pytest.mark.parametrize("name", ALL_STRATEGY_NAMES)
    def test_empty_universe_returns_empty(self, name, empty_view):
        cls = get(name)
        strat = cls()
        result = strat.generate(empty_view)
        assert result == []

    @pytest.mark.parametrize("name", ALL_STRATEGY_NAMES)
    def test_is_base_strategy_subclass(self, name):
        cls = get(name)
        assert issubclass(cls, BaseStrategy)
