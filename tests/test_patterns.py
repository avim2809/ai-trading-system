"""Tests for chart pattern detection (firm.patterns) and Strategy #13.

Fixtures are hand-built via linear interpolation between chosen anchor
points (index, price) rather than a random walk, so each test targets one
specific geometric pattern deterministically — a random walk won't reliably
contain any given pattern. Each leg between anchors is strictly monotonic
and exceeds the 3% zigzag threshold, so the resulting confirmed pivots land
exactly on the chosen anchors (see firm.patterns.extrema.zigzag_pivots:
bar 0 itself never becomes a pivot — it's the bootstrap anchor — so every
fixture's *first* pattern pivot is anchor[1], not anchor[0]).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from firm.contracts.models import Signal
from firm.data.pit_store import PointInTimeDataStore
from firm.patterns.extrema import zigzag_pivots
from firm.patterns.rules.continuation import detect_flag_pennant
from firm.patterns.rules.cup_handle import detect_cup_handle
from firm.patterns.rules.reversal import (
    detect_double_bottom,
    detect_double_top,
    detect_head_shoulders,
    detect_inverse_head_shoulders,
    detect_triple_bottom,
    detect_triple_top,
)
from firm.patterns.rules.triangle import detect_triangle_wedge_rectangle
from firm.patterns.scanner import scan_symbol
from firm.patterns.scorer import score_pattern
from firm.strategies.pattern_recognition import PatternRecognitionStrategy


def _ohlcv(anchors: list[tuple[int, float]], total_bars: int, *, wick: float = 0.002):
    idxs = [a[0] for a in anchors]
    prices = [a[1] for a in anchors]
    x = np.arange(total_bars)
    close = np.interp(x, idxs, prices)
    high = close * (1 + wick)
    low = close * (1 - wick)
    volume = np.full(total_bars, 1_000_000.0)
    return high, low, close, volume


def _spike(volume: np.ndarray, at: int, multiple: float = 2.5) -> np.ndarray:
    volume = volume.copy()
    volume[at] = volume[at] * multiple
    return volume


def _frame(anchors: list[tuple[int, float]], total_bars: int, *, spike_at: int | None = None) -> pd.DataFrame:
    high, low, close, volume = _ohlcv(anchors, total_bars)
    if spike_at is not None:
        volume = _spike(volume, spike_at)
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# ZigZag extrema
# ---------------------------------------------------------------------------

def test_zigzag_finds_alternating_pivots_at_exact_anchors():
    anchors = [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0)]
    high, low, close, _ = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    assert [p.index for p in pivots] == [10, 20]
    assert [p.kind for p in pivots] == ["peak", "trough"]
    assert pivots[0].price == pytest.approx(130.0, rel=0.01)
    assert pivots[1].price == pytest.approx(110.0, rel=0.01)


def test_zigzag_ignores_moves_below_threshold():
    anchors = [(0, 100.0), (10, 101.0), (20, 100.5), (30, 130.0)]
    high, low, close, _ = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    assert pivots == []  # still tracking the initial up-move; nothing confirmed yet


# ---------------------------------------------------------------------------
# Reversal patterns
# ---------------------------------------------------------------------------

def test_head_and_shoulders_top():
    anchors = [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0), (40, 112.0), (50, 131.0), (70, 100.0)]
    high, low, close, volume = _ohlcv(anchors, 71)
    volume = _spike(volume, 70)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_head_shoulders(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("head_shoulders_top", "short")
    assert match.confirm_index == 70
    assert match.stop > match.entry > match.target


def test_inverse_head_and_shoulders():
    anchors = [(0, 100.0), (10, 70.0), (20, 90.0), (30, 60.0), (40, 88.0), (50, 70.0), (70, 100.0)]
    high, low, close, volume = _ohlcv(anchors, 71)
    volume = _spike(volume, 70)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_inverse_head_shoulders(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("inverse_head_shoulders", "long")
    assert match.target > match.entry > match.stop


def test_double_top():
    anchors = [(0, 90.0), (10, 120.0), (20, 100.0), (30, 121.0), (45, 85.0)]
    high, low, close, volume = _ohlcv(anchors, 46)
    volume = _spike(volume, 45)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_double_top(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("double_top", "short")


def test_double_bottom():
    anchors = [(0, 120.0), (10, 90.0), (20, 110.0), (30, 89.0), (45, 130.0)]
    high, low, close, volume = _ohlcv(anchors, 46)
    volume = _spike(volume, 45)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_double_bottom(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("double_bottom", "long")


def test_triple_top():
    anchors = [(0, 90.0), (8, 120.0), (16, 102.0), (24, 121.0), (32, 103.0), (40, 120.0), (55, 90.0)]
    high, low, close, volume = _ohlcv(anchors, 56)
    volume = _spike(volume, 55)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triple_top(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("triple_top", "short")


def test_triple_bottom():
    anchors = [(0, 130.0), (8, 100.0), (16, 118.0), (24, 99.0), (32, 117.0), (40, 100.0), (55, 125.0)]
    high, low, close, volume = _ohlcv(anchors, 56)
    volume = _spike(volume, 55)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triple_bottom(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("triple_bottom", "long")


# ---------------------------------------------------------------------------
# Triangle / wedge / rectangle
# ---------------------------------------------------------------------------

def test_ascending_triangle():
    anchors = [(0, 100.0), (6, 120.0), (12, 108.0), (18, 120.5), (24, 113.0), (30, 125.0)]
    high, low, close, volume = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("ascending_triangle", "long")


def test_descending_triangle():
    # Trough2 needs its own confirming bounce (Peak3) before the real
    # breakdown, else it never registers as a confirmed pivot at all (a
    # continued decline straight through it doesn't "reverse" away from it).
    anchors = [(0, 100.0), (6, 120.0), (12, 105.0), (18, 113.0), (24, 106.0), (27, 110.0), (33, 95.0)]
    high, low, close, volume = _ohlcv(anchors, 34)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("descending_triangle", "short")


def test_symmetrical_triangle():
    anchors = [(0, 100.0), (6, 130.0), (12, 100.0), (18, 120.0), (24, 108.0), (30, 115.0)]
    high, low, close, volume = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("symmetrical_triangle", "long")


def test_rising_wedge():
    # Same confirming-bounce requirement as descending_triangle above.
    anchors = [(0, 100.0), (6, 110.0), (12, 100.0), (18, 118.0), (24, 112.0), (27, 116.0), (33, 100.0)]
    high, low, close, volume = _ohlcv(anchors, 34)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("rising_wedge", "short")


def test_falling_wedge():
    anchors = [(0, 100.0), (6, 130.0), (12, 110.0), (18, 118.0), (24, 104.0), (30, 115.0)]
    high, low, close, volume = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("falling_wedge", "long")


def test_rectangle():
    anchors = [(0, 100.0), (6, 120.0), (12, 100.0), (18, 120.3), (24, 99.7), (30, 125.0)]
    high, low, close, volume = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert match.pattern == "rectangle"
    assert match.direction == "long"


# ---------------------------------------------------------------------------
# Continuation: flag / pennant
# ---------------------------------------------------------------------------

def test_bull_flag():
    # A leading pivot (4, 90.0) before the flagpole is required: bar 0 never
    # becomes a pivot itself (it's zigzag's bootstrap anchor), and once the
    # breakout eventually confirms the consolidation's own trough as the
    # newest pivot, the detector's pole-pair search needs an earlier,
    # already-confirmed pivot to fall back to for the *real* flagpole.
    anchors = [
        (0, 100.0), (4, 90.0), (10, 120.0), (14, 117.0), (18, 119.0),
        (22, 116.0), (26, 118.0), (30, 130.0),
    ]
    high, low, close, volume = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_flag_pennant(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert match.pattern in ("bull_flag", "pennant")
    assert match.direction == "long"


def test_bear_flag():
    anchors = [
        (0, 100.0), (4, 110.0), (10, 80.0), (14, 83.0), (18, 81.0),
        (22, 84.0), (26, 82.0), (30, 70.0),
    ]
    high, low, close, volume = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_flag_pennant(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert match.pattern in ("bear_flag", "pennant")
    assert match.direction == "short"


# ---------------------------------------------------------------------------
# Cup & Handle / Rounding Bottom
# ---------------------------------------------------------------------------

def test_cup_and_handle():
    # Pure parabola for the cup span [5, 45] -> a>0, r2==1.0 by construction.
    cup_x = np.arange(5, 46)
    cup_close = 90.0 + 30.0 * ((cup_x - 25.0) / 20.0) ** 2  # cup_close[0] is x=5 itself
    pre = np.interp(np.arange(0, 5), [0, 5], [100.0, cup_close[0]])  # positions 0..4, up to (not incl.) x=5
    handle_x = np.arange(46, 53)
    handle_close = np.interp(handle_x, [45, 50, 53], [120.0, 113.0, 120.2])
    breakout = np.array([125.0])
    close = np.concatenate([pre, cup_close, handle_close, breakout])
    high = close * 1.002
    low = close * 0.998
    volume = np.full(len(close), 1_000_000.0)
    volume = _spike(volume, len(close) - 1)

    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_cup_handle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert match.direction == "long"
    assert match.pattern in ("cup_handle", "rounding_bottom")
    assert match.stop < match.entry < match.target


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------

def test_no_patterns_on_monotonic_trend():
    high, low, close, volume = _ohlcv([(0, 100.0), (100, 200.0)], 101)
    pivots = zigzag_pivots(high, low, pct=0.03)
    assert pivots == []  # nothing to reverse against — no pattern should fire
    assert detect_head_shoulders(high, low, close, volume, pivots) == []
    assert detect_double_top(high, low, close, volume, pivots) == []
    assert detect_triangle_wedge_rectangle(high, low, close, volume, pivots) == []
    assert detect_flag_pennant(high, low, close, volume, pivots) == []
    assert detect_cup_handle(high, low, close, volume, pivots) == []


def test_scan_symbol_never_raises_on_degenerate_input():
    df = pd.DataFrame({"high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
    assert scan_symbol(df) == []
    empty = pd.DataFrame({"high": [], "low": [], "close": [], "volume": []})
    assert scan_symbol(empty) == []


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def test_score_pattern_rewards_clean_confirmed_setups():
    score = score_pattern(
        geometry_tolerance_used=1.0,
        fit_quality=1.0,
        volume_ratio=2.0,
        duration_bars=40,
        follow_through_atr=3.0,
    )
    assert score.total == pytest.approx(100.0, abs=0.01)


def test_score_pattern_penalizes_weak_setups():
    score = score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,  # at the 20d average — no confirmation at all
        duration_bars=1,  # implausibly short
        follow_through_atr=0.0,  # barely crossed the level
    )
    assert score.total < 15.0


# ---------------------------------------------------------------------------
# recent_pivot_windows retry: collects every valid window, not just the first
# ---------------------------------------------------------------------------

def test_double_top_collects_all_valid_windows_best_quality_first():
    # Two overlapping double-top candidates share pivot P1: a loose,
    # lower-quality one from the most-recent window (P1,T1,P2) and a much
    # tighter, higher-quality one from an older window (P0,T0,P1) that
    # recent_pivot_windows's retry logic also tries. Both confirm on the
    # same final breakdown bar. Before the §6.2 fix, the raw detector
    # returned only the first (lower-quality, most-recent) window it found;
    # now it returns both, and scan_symbol's quality-score sort must put
    # the tighter one first regardless of which window was tried first.
    anchors = [
        (0, 100.0), (6, 120.0), (12, 100.0), (18, 120.5),
        (24, 95.0), (30, 118.0), (36, 85.0),
    ]
    df = _frame(anchors, 37, spike_at=36)
    matches = scan_symbol(df, min_score=0.0, zigzag_pct=0.03, enabled_patterns={"double_top"})
    double_tops = [m for m in matches if m.pattern == "double_top"]
    assert len(double_tops) == 2
    assert double_tops[0].quality_score > double_tops[1].quality_score
    assert double_tops[0].geometry_tolerance_used > double_tops[1].geometry_tolerance_used


# ---------------------------------------------------------------------------
# scan_symbol integration (min_score filtering + best-match ordering)
# ---------------------------------------------------------------------------

def test_scan_symbol_filters_by_min_score_and_sorts_best_first():
    df = _frame(
        [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0), (40, 112.0), (50, 131.0), (70, 100.0)],
        71,
        spike_at=70,
    )
    matches_lenient = scan_symbol(df, min_score=0.0, zigzag_pct=0.03)
    assert len(matches_lenient) >= 1
    assert all(matches_lenient[i].quality_score >= matches_lenient[i + 1].quality_score for i in range(len(matches_lenient) - 1))

    matches_strict = scan_symbol(df, min_score=200.0, zigzag_pct=0.03)  # impossible threshold
    assert matches_strict == []


# ---------------------------------------------------------------------------
# End-to-end: Strategy #13 against the real BaseStrategy/Signal/PitView contract
# ---------------------------------------------------------------------------

class _FakePitView:
    """Minimal PitView stand-in — only implements what the strategy actually
    calls (.universe, .prices()), matching the Protocol in strategies/base.py.
    """

    def __init__(self, prices_df: pd.DataFrame, symbols: list[str], asof: datetime):
        self._prices_df = prices_df
        self._symbols = symbols
        self._asof = asof

    @property
    def asof(self) -> datetime:
        return self._asof

    @property
    def universe(self) -> list[str]:
        return self._symbols

    def prices(self, symbols=None, lookback_days: int = 252) -> pd.DataFrame:
        return self._prices_df


def _build_prices_df(symbol: str, anchors: list[tuple[int, float]], total_bars: int, spike_at: int) -> pd.DataFrame:
    high, low, close, volume = _ohlcv(anchors, total_bars)
    volume = _spike(volume, spike_at)
    dates = pd.date_range("2024-01-01", periods=total_bars, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,  # no splits/divs in this fixture -> factor 1.0
            "volume": volume,
        }
    )


def test_pattern_recognition_strategy_emits_signal_for_confirmed_pattern():
    # Bull flag, not double-bottom: a double top/bottom's measured-move target
    # height and its trough-based stop distance are both ~= the peak-to-trough
    # swing, so risk:reward structurally comes out near 1:1 -- correctly
    # filtered out by min_risk_reward (1.5 default) rather than a fixture or
    # wiring bug. A flag's target (flagpole height) vs. its much tighter
    # consolidation-range stop clears that bar comfortably instead.
    anchors = [
        (0, 100.0), (4, 90.0), (10, 120.0), (14, 117.0), (18, 119.0),
        (22, 116.0), (26, 118.0), (30, 130.0),
    ]
    prices_df = _build_prices_df("AAPL", anchors, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    strategy = PatternRecognitionStrategy()
    signals = strategy.generate(pit_view)

    assert len(signals) == 1
    sig = signals[0]
    assert isinstance(sig, Signal)
    assert sig.symbol == "AAPL"
    assert sig.strategy == "pattern_recognition"
    assert sig.score > 0  # bullish pattern -> positive score
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.meta["pattern"] in ("bull_flag", "pennant")
    assert sig.meta["direction"] == "long"
    assert sig.meta["stop"] < sig.meta["entry"] < sig.meta["target"]


def test_pattern_recognition_strategy_handles_empty_universe():
    pit_view = _FakePitView(pd.DataFrame(), [], datetime(2024, 3, 1))
    assert PatternRecognitionStrategy().generate(pit_view) == []


def test_pattern_recognition_strategy_never_raises_on_flat_data():
    total_bars = 60
    dates = pd.date_range("2024-01-01", periods=total_bars, freq="B")
    flat = pd.DataFrame(
        {
            "date": dates,
            "symbol": "FLAT",
            "open": 50.0,
            "high": 50.0,
            "low": 50.0,
            "close": 50.0,
            "adj_close": 50.0,
            "volume": 1_000_000.0,
        }
    )
    pit_view = _FakePitView(flat, ["FLAT"], datetime(2024, 3, 1))
    assert PatternRecognitionStrategy().generate(pit_view) == []


def test_pattern_recognition_is_registered():
    from firm.strategies.registry import get, list_strategies

    assert "pattern_recognition" in list_strategies()
    assert get("pattern_recognition") is PatternRecognitionStrategy
