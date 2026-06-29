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
    ) -> pd.DataFrame:
        df = self._fund_df.copy()
        if symbols and not df.empty:
            df = df[df["symbol"].isin(symbols)]
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

    def fundamentals(self, symbols=None) -> pd.DataFrame:
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

    def fundamentals(self, symbols=None) -> pd.DataFrame:
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
