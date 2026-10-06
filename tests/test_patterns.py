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
    # 2026-09-27 (find_confirmation edge-trigger fix, Workstream C): the tail
    # used to run straight from the right shoulder (50, 131.0) to the final
    # bar (70, 100.0) — under the old is-beyond confirmation test that
    # crossed the neckline once around bar ~61 and just stayed "confirmed"
    # (i.e. still beyond) all the way to bar 70, which is what made the old
    # test pass on a phantom "today" confirmation. Now confirmation requires
    # an actual close[i-1]-inside -> close[i]-beyond transition, so the
    # fixture holds just above the neckline (67, 116.0) through the last
    # quiet bars and only breaks through in the final 3-bar confirm window,
    # landing the genuine crossing at bar 68 (verified against the real
    # detector, not hand-derived).
    anchors = [
        (0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0), (40, 112.0), (50, 131.0),
        (67, 116.0), (70, 100.0),
    ]
    high, low, close, volume = _ohlcv(anchors, 71)
    volume = _spike(volume, 70)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_head_shoulders(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("head_shoulders_top", "short")
    assert match.confirm_index == 68
    assert match.stop > match.entry > match.target


def test_inverse_head_and_shoulders():
    # Mirrors test_head_and_shoulders_top's 2026-09-27 fix: hold just below
    # the neckline (67, 85.0) so the genuine upward crossing lands inside
    # the last 3-bar confirm window instead of having already happened
    # bars ago.
    anchors = [
        (0, 100.0), (10, 70.0), (20, 90.0), (30, 60.0), (40, 88.0), (50, 70.0),
        (67, 85.0), (70, 100.0),
    ]
    high, low, close, volume = _ohlcv(anchors, 71)
    volume = _spike(volume, 70)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_inverse_head_shoulders(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("inverse_head_shoulders", "long")
    assert match.target > match.entry > match.stop


def test_double_top():
    # 2026-09-27: breakout moved from 15 bars past the last peak (45) to 3
    # (33) so the genuine level-crossing (not just "still past it") falls
    # inside find_confirmation's 3-bar lookback window.
    anchors = [(0, 90.0), (10, 120.0), (20, 100.0), (30, 121.0), (33, 85.0)]
    high, low, close, volume = _ohlcv(anchors, 34)
    volume = _spike(volume, 33)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_double_top(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("double_top", "short")


def test_double_bottom():
    # 2026-09-27: same shrink as test_double_top, mirrored.
    anchors = [(0, 120.0), (10, 90.0), (20, 110.0), (30, 89.0), (33, 130.0)]
    high, low, close, volume = _ohlcv(anchors, 34)
    volume = _spike(volume, 33)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_double_bottom(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("double_bottom", "long")


def test_triple_top():
    # 2026-09-27: breakout shrunk from 15 bars past the last peak (40) to 2
    # (42) for the same reason as test_double_top above.
    anchors = [(0, 90.0), (8, 120.0), (16, 102.0), (24, 121.0), (32, 103.0), (40, 120.0), (42, 90.0)]
    high, low, close, volume = _ohlcv(anchors, 43)
    volume = _spike(volume, 42)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triple_top(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("triple_top", "short")


def test_triple_bottom():
    # 2026-09-27: same shrink as test_triple_top, mirrored.
    anchors = [(0, 130.0), (8, 100.0), (16, 118.0), (24, 99.0), (32, 117.0), (40, 100.0), (42, 130.0)]
    high, low, close, volume = _ohlcv(anchors, 43)
    volume = _spike(volume, 42)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triple_bottom(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("triple_bottom", "long")


# ---------------------------------------------------------------------------
# Triangle / wedge / rectangle
# ---------------------------------------------------------------------------

def test_ascending_triangle():
    # 2026-09-27: breakout shrunk from 6 bars past the last pivot (30) to 3
    # (27) so find_confirmation's genuine crossing (not just "still past
    # it") falls inside the 3-bar lookback window.
    anchors = [(0, 100.0), (6, 120.0), (12, 108.0), (18, 120.5), (24, 113.0), (27, 125.0)]
    high, low, close, volume = _ohlcv(anchors, 28)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("ascending_triangle", "long")


def test_descending_triangle():
    # Trough2 needs its own confirming bounce (Peak3) before the real
    # breakdown, else it never registers as a confirmed pivot at all (a
    # continued decline straight through it doesn't "reverse" away from it).
    # 2026-09-27: breakout shrunk from 6 bars past Peak3 (27) to 3 (30) --
    # same find_confirmation lookback-window reasoning as the reversal
    # fixtures above.
    anchors = [(0, 100.0), (6, 120.0), (12, 105.0), (18, 113.0), (24, 106.0), (27, 110.0), (30, 95.0)]
    high, low, close, volume = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("descending_triangle", "short")


def test_symmetrical_triangle():
    # 2026-09-27: breakout shrunk from 6 bars past the last pivot (24) to 3
    # (27) -- find_confirmation lookback-window reasoning as above.
    anchors = [(0, 100.0), (6, 130.0), (12, 100.0), (18, 120.0), (24, 108.0), (27, 115.0)]
    high, low, close, volume = _ohlcv(anchors, 28)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("symmetrical_triangle", "long")


def test_rising_wedge():
    # Same confirming-bounce requirement as descending_triangle above.
    # 2026-09-27: breakout shrunk from 6 bars past Peak3 (27) to 3 (30) --
    # same find_confirmation lookback-window reasoning as above.
    anchors = [(0, 100.0), (6, 110.0), (12, 100.0), (18, 118.0), (24, 112.0), (27, 116.0), (30, 100.0)]
    high, low, close, volume = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("rising_wedge", "short")


def test_falling_wedge():
    # 2026-09-27: breakout shrunk from 6 bars past the last pivot (24) to 3
    # (27) -- find_confirmation lookback-window reasoning as above.
    anchors = [(0, 100.0), (6, 130.0), (12, 110.0), (18, 118.0), (24, 104.0), (27, 115.0)]
    high, low, close, volume = _ohlcv(anchors, 28)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_triangle_wedge_rectangle(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert (match.pattern, match.direction) == ("falling_wedge", "long")


def test_rectangle():
    # 2026-09-27: breakout shrunk from 6 bars past the last pivot (24) to 3
    # (27) -- find_confirmation lookback-window reasoning as above.
    anchors = [(0, 100.0), (6, 120.0), (12, 100.0), (18, 120.3), (24, 99.7), (27, 125.0)]
    high, low, close, volume = _ohlcv(anchors, 28)
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
    #
    # 2026-09-27 (find_confirmation edge-trigger fix): the consolidation
    # used to sit right against the upper trendline (116-119) and the final
    # breakout ramp spanned 4 bars (26 -> 30) -- the genuine crossing landed
    # around bar 26/27, one to two bars *before* the 3-bar confirm window
    # (fit_end=n-3=28), so it went undetected once "still past the level"
    # stopped counting as confirmation. The consolidation now sits further
    # below the eventual breakout (112-118) with an added flat hold bar
    # (27) right at the fit boundary, so the upper trendline stays low and
    # the whole crossing happens inside bars 28-30 (verified against the
    # real detector).
    anchors = [
        (0, 100.0), (4, 90.0), (10, 120.0), (14, 116.0), (18, 118.0),
        (22, 114.0), (26, 112.0), (27, 112.0), (30, 140.0),
    ]
    high, low, close, volume = _ohlcv(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    matches = detect_flag_pennant(high, low, close, volume, pivots)
    assert matches
    match = matches[0]
    assert match.pattern in ("bull_flag", "pennant")
    assert match.direction == "long"


def test_bear_flag():
    # 2026-09-27: mirrors test_bull_flag's fix above.
    anchors = [
        (0, 100.0), (4, 110.0), (10, 80.0), (14, 84.0), (18, 82.0),
        (22, 86.0), (26, 88.0), (27, 88.0), (30, 60.0),
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
    # Also exercises the two 2026-09 optional components (breakout_distance,
    # pre_breakout_compression — see scorer.py's module docstring): a
    # "clean confirmed setup" now means clean on all seven, not just the
    # original five, so this fixture is deliberately clean on those too --
    # bar 24 (confirm_index - 1) is compressed vs. its trailing 20-bar
    # average (0.5 vs. a ~0.975 rolling mean -> full pre_breakout_compression
    # credit), and bar 25 (confirm_index)'s ATR of 5.0 makes the 5-point
    # close/level gap exactly 1.0x ATR -> full breakout_distance credit.
    atr_series = np.full(30, 1.0)
    atr_series[24] = 0.5
    atr_series[25] = 5.0
    score = score_pattern(
        geometry_tolerance_used=1.0,
        fit_quality=1.0,
        volume_ratio=2.0,
        duration_bars=40,
        follow_through_atr=3.0,
        close_at_confirm=105.0,
        level_at_confirm=100.0,
        atr_series=atr_series,
        confirm_index=25,
    )
    assert score.total == pytest.approx(100.0, abs=0.01)


def test_score_pattern_without_new_optional_context_caps_at_85():
    # Documented backward-compat contract (scorer.py module docstring): a
    # caller that doesn't supply close_at_confirm/level_at_confirm/
    # atr_series/confirm_index gets a zero contribution from both new
    # components, not a crash -- effective max of 85, not 100.
    score = score_pattern(
        geometry_tolerance_used=1.0,
        fit_quality=1.0,
        volume_ratio=2.0,
        duration_bars=40,
        follow_through_atr=3.0,
    )
    assert score.total == pytest.approx(85.0, abs=0.01)
    assert score.breakout_distance == 0.0
    assert score.pre_breakout_compression == 0.0


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
    #
    # 2026-09-27 (find_confirmation edge-trigger fix): added a flat hold bar
    # (33, 118.0) before the final breakdown -- the old single 6-bar ramp
    # (30 -> 36) crossed both windows' levels (95.0 / 100.0) well before the
    # 3-bar confirm window, so neither window's now-genuine crossing test
    # found anything. Verified both windows still confirm (at their own,
    # possibly different, crossing bars) and the quality ordering holds.
    anchors = [
        (0, 100.0), (6, 120.0), (12, 100.0), (18, 120.5),
        (24, 95.0), (30, 118.0), (33, 118.0), (36, 80.0),
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
    # 2026-09-27: same hold-then-break fix as test_head_and_shoulders_top
    # (the anchors are otherwise identical) -- see that test's comment.
    df = _frame(
        [
            (0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0), (40, 112.0), (50, 131.0),
            (67, 116.0), (70, 100.0),
        ],
        71,
        spike_at=70,
    )
    matches_lenient = scan_symbol(df, min_score=0.0, zigzag_pct=0.03)
    assert len(matches_lenient) >= 1
    assert all(matches_lenient[i].quality_score >= matches_lenient[i + 1].quality_score for i in range(len(matches_lenient) - 1))

    matches_strict = scan_symbol(df, min_score=200.0, zigzag_pct=0.03)  # impossible threshold
    assert matches_strict == []


# ---------------------------------------------------------------------------
# 2026-09 scanner.py integration: ATR-scaled zigzag threshold, retest and
# weekly-confluence quality modifiers. All three default off (see
# scan_symbol's own docstring) -- these tests exercise the *wiring*
# (scan_symbol correctly reaches each new module and applies its result),
# not the underlying business logic itself (already covered in isolation by
# test_extrema.py/test_confirmation.py/test_confluence.py).
# ---------------------------------------------------------------------------

# 2026-09-27 (find_confirmation edge-trigger fix): added a hold bar (42,
# 105.0) that keeps price above the neckline (100.0) until the final 3-bar
# confirm window (n=46, lookback=3 -> bars 43-45) -- the old single 15-bar
# ramp (30 -> 45) crossed the neckline around bar 39, well before "today",
# so it no longer counts as confirmed under the corrected, genuinely-
# crossing-based test.
_DOUBLE_TOP_ANCHORS = [(0, 90.0), (10, 120.0), (20, 100.0), (30, 121.0), (42, 105.0), (45, 85.0)]


def test_scan_symbol_zigzag_atr_mult_changes_pivot_detection():
    df = _frame(_DOUBLE_TOP_ANCHORS, 46, spike_at=45)

    baseline = scan_symbol(df, min_score=0.0, zigzag_pct=0.03)
    assert baseline  # the fixed-pct threshold confirms the double top

    # An absurdly large ATR multiple makes every per-bar threshold far
    # larger than any real retracement in this fixture, so *no* reversal
    # ever confirms -- zero pivots, zero matches. This alone proves
    # `zigzag_atr_mult` is actually reaching zigzag_pivots' threshold_fn
    # (not silently ignored): the same df with the same `zigzag_pct=0.03`
    # given as a fallback now returns nothing once the ATR path is active.
    gated = scan_symbol(df, min_score=0.0, zigzag_pct=0.03, zigzag_atr_mult=100.0)
    assert gated == []


def test_scan_symbol_retest_modifier_applies_scaled_adjustment(monkeypatch):
    import firm.patterns.scanner as scanner_module

    df = _frame(_DOUBLE_TOP_ANCHORS, 46, spike_at=45)
    baseline = scan_symbol(df, min_score=0.0, zigzag_pct=0.03)
    top = next(m for m in baseline if m.pattern == "double_top")
    assert "retest_outcome" not in top.score_breakdown  # off by default -- no key at all

    monkeypatch.setattr(scanner_module, "retest_outcome", lambda *a, **k: "held")
    with_retest = scan_symbol(
        df, min_score=0.0, zigzag_pct=0.03,
        retest_modifier_enabled=True, retest_modifier_scale=4.0,
    )
    match = next(
        m for m in with_retest if m.pattern == top.pattern and m.confirm_index == top.confirm_index
    )
    assert match.score_breakdown["retest_outcome"] == "held"
    assert match.score_breakdown["retest_modifier"] == pytest.approx(4.0)  # retest_score_modifier("held") == 1.0
    assert match.quality_score == pytest.approx(min(100.0, top.quality_score + 4.0), abs=1e-6)


def test_scan_symbol_confluence_modifier_applies_scaled_adjustment(monkeypatch):
    import firm.patterns.scanner as scanner_module

    df = _frame(_DOUBLE_TOP_ANCHORS, 46, spike_at=45)
    dates = pd.date_range("2024-01-01", periods=46, freq="B")
    baseline = scan_symbol(df, min_score=0.0, zigzag_pct=0.03)
    top = next(m for m in baseline if m.pattern == "double_top")
    assert top.direction == "short"
    assert "weekly_trend" not in top.score_breakdown  # off by default -- no key at all

    # A "down" weekly trend agrees with a "short" pattern direction (see
    # confluence.py's _AGREEING_TREND) -> +1.0 * scale.
    monkeypatch.setattr(scanner_module, "weekly_trend_direction", lambda *a, **k: "down")
    with_confluence = scan_symbol(
        df, min_score=0.0, zigzag_pct=0.03,
        confluence_modifier_enabled=True, confluence_modifier_scale=2.5, dates=dates,
    )
    match = next(
        m for m in with_confluence if m.pattern == top.pattern and m.confirm_index == top.confirm_index
    )
    assert match.score_breakdown["weekly_trend"] == "down"
    assert match.score_breakdown["confluence_modifier"] == pytest.approx(2.5)
    assert match.quality_score == pytest.approx(min(100.0, top.quality_score + 2.5), abs=1e-6)


def test_scan_symbol_confluence_modifier_skipped_without_dates(caplog):
    df = _frame(_DOUBLE_TOP_ANCHORS, 46, spike_at=45)
    # confluence_modifier_enabled with no `dates` supplied must degrade
    # gracefully (no crash, no modifier applied) rather than raising --
    # scan_symbol's own docstring documents this as always-safe.
    matches = scan_symbol(
        df, min_score=0.0, zigzag_pct=0.03, confluence_modifier_enabled=True, dates=None,
    )
    assert matches
    assert all("weekly_trend" not in m.score_breakdown for m in matches)


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
    #
    # 2026-09-27: same find_confirmation edge-trigger fix as test_bull_flag
    # above, but tuned to also clear min_score=60 through the *full* scorer
    # pipeline (test_bull_flag calls detect_flag_pennant directly, bypassing
    # scoring entirely, so its simpler fixture doesn't need this) --
    # identical anchors to _BULL_FLAG_ANCHORS below, see that constant's
    # comment for the quality-score tuning rationale.
    anchors = [
        (0, 100.0), (4, 90.0), (10, 150.0), (14, 144.0), (18, 147.0),
        (22, 141.0), (26, 135.0), (29, 135.0), (30, 180.0),
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


# ---------------------------------------------------------------------------
# CNN/GAF quality-scoring wiring (firm.patterns.ml.inference) -- candidate
# detection is unchanged either way; only which score becomes the signal's
# quality/confidence should differ. See test_pattern_ml_inference.py for the
# scoring module itself.
# ---------------------------------------------------------------------------

# 2026-09-27 (find_confirmation edge-trigger fix): the old consolidation
# (116-119) sat right against the upper trendline and the final breakout
# ramp (26 -> 30) crossed it a bar or two *before* the 3-bar confirm window,
# so it went undetected once "still past the level" stopped counting.
# Reshaped so bars 26-29 hold flat comfortably below the trendline (its
# extrapolated value at 28/29 is ~137/136) and only bar 30 breaks through --
# a genuine crossing landing on the fixture's own last bar (matching every
# call site's `spike_at=30` unchanged). Also scaled up (flagpole 90->150
# instead of 90->120, tighter consolidation) so the *full* scorer pipeline
# (not just detect_flag_pennant's raw candidate) clears min_score=60 --
# every downstream test here goes through PatternRecognitionStrategy /
# scan_symbol, unlike test_bull_flag's direct detect_flag_pennant call.
# Verified against the real detector + scorer, not hand-derived:
# quality_score ~= 69.9, confirm_index == 30, single unambiguous match.
_BULL_FLAG_ANCHORS = [
    (0, 100.0), (4, 90.0), (10, 150.0), (14, 144.0), (18, 147.0),
    (22, 141.0), (26, 135.0), (29, 135.0), (30, 180.0),
]


def test_pattern_recognition_falls_back_to_rule_based_when_cnn_unavailable(monkeypatch):
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.cnn_inference, "is_available", lambda: False)

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy().generate(pit_view)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.meta["scoring_mode"] == "rule_based"
    assert sig.meta["cnn_quality_fraction"] is None
    expected_fraction = min(sig.meta["quality_score"] / 100.0, 1.0)
    assert sig.confidence == pytest.approx(expected_fraction)
    assert sig.score == pytest.approx(expected_fraction)  # bull flag -> long -> +sign


def test_pattern_recognition_uses_cnn_quality_when_available(monkeypatch):
    """CNN score should replace the rule-based quality score whenever the
    scorer returns a value, without touching candidate detection at all --
    same matches, same pattern/direction, only the score/confidence numbers
    change.
    """
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.cnn_inference, "is_available", lambda: True)
    monkeypatch.setattr(
        pr_module.cnn_inference,
        "score_pattern_quality",
        lambda close, confirm_index, **kwargs: 0.9,
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    # cnn_scoring_enabled defaults False (2026-09-20) -- explicit opt-in
    # required in addition to is_available(), since the on-disk model
    # artifact was validated to hurt performance; must be set explicitly
    # here to exercise the CNN path at all.
    signals = PatternRecognitionStrategy(
        params={"cnn_scoring_enabled": True},
    ).generate(pit_view)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.meta["scoring_mode"] == "cnn"
    assert sig.meta["cnn_quality_fraction"] == pytest.approx(0.9)
    assert sig.confidence == pytest.approx(0.9)
    assert sig.score == pytest.approx(0.9)
    # Rule-based score is still recorded for comparison, just not used.
    assert sig.meta["rule_based_quality_fraction"] == pytest.approx(
        min(sig.meta["quality_score"] / 100.0, 1.0)
    )
    assert sig.meta["pattern"] in ("bull_flag", "pennant")
    assert sig.meta["direction"] == "long"


def test_pattern_recognition_falls_back_when_cnn_scorer_returns_none(monkeypatch):
    """CNN reports itself available but declines to score this particular
    match (e.g. not enough history before confirm_index) -- must fall back
    to the rule-based score for that signal, not drop it or crash.
    """
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.cnn_inference, "is_available", lambda: True)
    monkeypatch.setattr(
        pr_module.cnn_inference,
        "score_pattern_quality",
        lambda close, confirm_index, **kwargs: None,
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    # Pre-existing gap fixed 2026-09: without this override,
    # cnn_scoring_enabled defaults False and the CNN path (mocked above)
    # never actually runs, so the assertions below would previously pass
    # vacuously regardless of what score_pattern_quality returned.
    signals = PatternRecognitionStrategy(params={"cnn_scoring_enabled": True}).generate(pit_view)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.meta["scoring_mode"] == "rule_based"
    expected_fraction = min(sig.meta["quality_score"] / 100.0, 1.0)
    assert sig.confidence == pytest.approx(expected_fraction)


# ---------------------------------------------------------------------------
# 2026-09 XGBoost pattern-confirmation ensemble wiring
# (firm.patterns.ml.xgb_inference) -- off by default; exercises the
# fixed-weight blend + agreement-gate formula and the
# meta["calibrated_probability"] convention TraderAgent._kelly consumes.
# ---------------------------------------------------------------------------

def test_pattern_recognition_xgb_ensemble_blends_and_gates_on_disagreement(monkeypatch):
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.xgb_inference, "is_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference,
        "score_pattern_confirmation",
        lambda features, **kwargs: (0.1, 0.1, 0.8),  # (p_stop, p_timeout, p_target)
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={
            "xgb_confirmation_enabled": True,
            "xgb_blend_weight": 0.5,
            "xgb_agreement_gate": True,
            "xgb_agreement_gate_threshold": 0.35,
            "xgb_agreement_gate_dampen": 0.7,
        },
    ).generate(pit_view)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.meta["xgb_p_target"] == pytest.approx(0.8)
    # No xgb_calibration_path configured -- xgb_p_target is RAW/uncalibrated
    # XGBoost output here, so calibrated_probability (the meta-labeling
    # convention TraderAgent._signal_calibrated_edge trusts at face value)
    # must stay None rather than mislabeling it. See
    # test_pattern_recognition_xgb_ensemble_populates_calibrated_probability_when_calibration_configured
    # for the case where it *is* populated.
    assert sig.meta["calibrated_probability"] is None
    assert sig.meta["scoring_mode"] == "rule_based+xgb"

    rule_based_fraction = sig.meta["rule_based_quality_fraction"]
    disagreement = abs(0.8 - rule_based_fraction)
    blended = 0.5 * 0.8 + 0.5 * rule_based_fraction
    if disagreement > 0.35:
        blended *= 0.7
    expected = min(1.0, max(0.0, blended))
    assert sig.confidence == pytest.approx(expected, abs=1e-6)
    assert sig.score == pytest.approx(expected, abs=1e-6)  # bull flag -> long -> + sign


def test_pattern_recognition_xgb_ensemble_passes_aligned_market_context_to_build_features(monkeypatch):
    """Part B item 6 (2026-09-27): when xgb_confirmation_enabled, generate()
    must build a real, positionally-aligned market_ohlcv and pass it to
    build_features -- not leave the new market-context feature block
    permanently unused (the same "not a decorative modifier" bar Part A
    item 6 held the regime discount to)."""
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.xgb_inference, "is_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_confirmation", lambda features, **kwargs: (0.1, 0.1, 0.8),
    )
    captured_market_ohlcv = {}
    real_build_features = pr_module.build_features

    def _capturing_build_features(match, ohlcv, *, market_ohlcv=None, **kwargs):
        captured_market_ohlcv[match.direction] = market_ohlcv  # unique enough per call here
        return real_build_features(match, ohlcv, market_ohlcv=market_ohlcv, **kwargs)

    monkeypatch.setattr(pr_module, "build_features", _capturing_build_features)

    # Same anchors/date range for both symbols -> dates align exactly,
    # so alignment must succeed (not silently fall back to None).
    prices_df = pd.concat([
        _build_prices_df("AAA", _BULL_FLAG_ANCHORS, 31, spike_at=30),
        _build_prices_df("BBB", _BULL_FLAG_ANCHORS, 31, spike_at=30),
    ], ignore_index=True)
    pit_view = _FakePitView(prices_df, ["AAA", "BBB"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(params={"xgb_confirmation_enabled": True}).generate(pit_view)

    assert len(signals) == 2
    assert len(captured_market_ohlcv) == 1  # both AAA/BBB confirm "long" (same anchors) -> 1 dict key
    market_ohlcv = next(iter(captured_market_ohlcv.values()))
    assert market_ohlcv is not None
    assert "close" in market_ohlcv.columns
    # AAA and BBB are IDENTICAL price series here -> the equal-weight
    # market proxy must equal each one's own close exactly.
    aaa_close = prices_df[prices_df["symbol"] == "AAA"].sort_values("date")["adj_close"].to_numpy()
    np.testing.assert_allclose(market_ohlcv["close"].to_numpy(), aaa_close)


def test_pattern_recognition_xgb_ensemble_populates_calibrated_probability_when_calibration_configured(monkeypatch, tmp_path):
    from firm.patterns.ml.calibration import save_calibration
    from firm.strategies import pattern_recognition as pr_module

    calibration_path = tmp_path / "xgb_calibration.json"
    save_calibration({"type": "sigmoid", "model": "xgboost", "a": 1.0, "b": 0.0}, calibration_path)

    monkeypatch.setattr(pr_module.xgb_inference, "is_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference,
        "score_pattern_confirmation",
        lambda features, **kwargs: (0.1, 0.1, 0.8),
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={
            "xgb_confirmation_enabled": True,
            "xgb_calibration_path": str(calibration_path),
        },
    ).generate(pit_view)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.meta["xgb_p_target"] == pytest.approx(0.8)
    # A real "sigmoid" calibration file was configured and load_calibration
    # confirms its type -- calibrated_probability may now be honored.
    assert sig.meta["calibrated_probability"] == pytest.approx(0.8)


def test_pattern_recognition_xgb_ensemble_ignores_wrong_calibration_type(monkeypatch, tmp_path):
    from firm.patterns.ml.calibration import save_calibration
    from firm.strategies import pattern_recognition as pr_module

    # A "temperature" calibration is meaningful for the CNN's pre-softmax
    # logits, not this ONNX graph's already-softmaxed probabilities --
    # xgb_p_target must not be relabeled calibrated on a type mismatch.
    calibration_path = tmp_path / "xgb_calibration.json"
    save_calibration({"type": "temperature", "temperature": 1.2}, calibration_path)

    monkeypatch.setattr(pr_module.xgb_inference, "is_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference,
        "score_pattern_confirmation",
        lambda features, **kwargs: (0.1, 0.1, 0.8),
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={
            "xgb_confirmation_enabled": True,
            "xgb_calibration_path": str(calibration_path),
        },
    ).generate(pit_view)

    assert signals[0].meta["calibrated_probability"] is None


def test_pattern_recognition_xgb_ensemble_ignores_calibration_fit_on_wrong_model(monkeypatch, tmp_path):
    """Regression (2026-09-27): a calibration file with the RIGHT type
    ("sigmoid") but fit on the WRONG model's raw score (model="rule_based",
    i.e. quality_score, not XGBoost's p_target) must be rejected too --
    "type" matching alone is not sufficient, since two different models'
    raw scores can both be validly sigmoid-calibrated while being
    completely incompatible fits."""
    from firm.patterns.ml.calibration import save_calibration
    from firm.strategies import pattern_recognition as pr_module

    calibration_path = tmp_path / "mismatched_calibration.json"
    save_calibration({"type": "sigmoid", "model": "rule_based", "a": 1.0, "b": 0.0}, calibration_path)

    monkeypatch.setattr(pr_module.xgb_inference, "is_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference,
        "score_pattern_confirmation",
        lambda features, **kwargs: (0.1, 0.1, 0.8),
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={
            "xgb_confirmation_enabled": True,
            "xgb_calibration_path": str(calibration_path),
        },
    ).generate(pit_view)

    assert signals[0].meta["calibrated_probability"] is None


# ---------------------------------------------------------------------------
# XGBoost meta-labeling secondary model (Part B item 2, 2026-09-27) --
# distinct on/off switch and distinct model from the 3-class ensemble above.
# ---------------------------------------------------------------------------

def test_pattern_recognition_xgb_meta_off_by_default_leaves_meta_none():
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy().generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["xgb_meta_p_act"] is None


def test_pattern_recognition_xgb_meta_populates_p_act_when_enabled(monkeypatch):
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.xgb_inference, "is_meta_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_meta_confirmation", lambda features, **kwargs: 0.72,
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(params={"xgb_meta_confirmation_enabled": True}).generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["xgb_meta_p_act"] == pytest.approx(0.72)
    # No calibration file configured -- raw p_act, so calibrated_probability
    # stays None (same "never trust raw output as calibrated" convention as
    # the 3-class model).
    assert signals[0].meta["calibrated_probability"] is None
    # The 3-class ensemble is off -- quality_fraction/scoring_mode must be
    # completely unaffected by the meta model (it never touches score, only
    # calibrated_probability).
    assert signals[0].meta["scoring_mode"] == "rule_based"
    assert signals[0].meta["xgb_p_target"] is None


def test_pattern_recognition_xgb_meta_calibration_configured_populates_calibrated_probability(monkeypatch, tmp_path):
    from firm.patterns.ml.calibration import save_calibration
    from firm.strategies import pattern_recognition as pr_module

    calibration_path = tmp_path / "xgb_meta_calibration.json"
    save_calibration({"type": "sigmoid", "model": "xgboost_meta", "a": 1.0, "b": 0.0}, calibration_path)

    monkeypatch.setattr(pr_module.xgb_inference, "is_meta_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_meta_confirmation", lambda features, **kwargs: 0.65,
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={
            "xgb_meta_confirmation_enabled": True,
            "xgb_meta_calibration_path": str(calibration_path),
        },
    ).generate(pit_view)

    assert signals[0].meta["calibrated_probability"] == pytest.approx(0.65)


def test_pattern_recognition_xgb_meta_ignores_calibration_fit_on_3class_model(monkeypatch, tmp_path):
    """Regression: a calibration file fit for the 3-class model
    (model="xgboost") must be rejected for the meta model even though both
    are "xgboost" family and both use type="sigmoid" -- the two models have
    different output distributions (p_target vs. p_act), exactly the
    mismatch the "xgboost_meta" discriminator exists to catch."""
    from firm.patterns.ml.calibration import save_calibration
    from firm.strategies import pattern_recognition as pr_module

    calibration_path = tmp_path / "wrong_model_calibration.json"
    save_calibration({"type": "sigmoid", "model": "xgboost", "a": 1.0, "b": 0.0}, calibration_path)

    monkeypatch.setattr(pr_module.xgb_inference, "is_meta_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_meta_confirmation", lambda features, **kwargs: 0.65,
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={
            "xgb_meta_confirmation_enabled": True,
            "xgb_meta_calibration_path": str(calibration_path),
        },
    ).generate(pit_view)

    assert signals[0].meta["xgb_meta_p_act"] == pytest.approx(0.65)  # still scored...
    assert signals[0].meta["calibrated_probability"] is None  # ...just not trusted as calibrated


def test_pattern_recognition_xgb_meta_takes_priority_over_3class_for_calibrated_probability(monkeypatch, tmp_path):
    """When BOTH models are enabled and calibrated, the meta model's own
    output wins for calibrated_probability -- it IS the de Prado meta-label
    confidence gate by construction; the 3-class model's p_target answers a
    different question (see default_params' docstring for this knob)."""
    from firm.patterns.ml.calibration import save_calibration
    from firm.strategies import pattern_recognition as pr_module

    xgb_cal_path = tmp_path / "xgb_calibration.json"
    save_calibration({"type": "sigmoid", "model": "xgboost", "a": 1.0, "b": 0.0}, xgb_cal_path)
    meta_cal_path = tmp_path / "xgb_meta_calibration.json"
    save_calibration({"type": "sigmoid", "model": "xgboost_meta", "a": 1.0, "b": 0.0}, meta_cal_path)

    monkeypatch.setattr(pr_module.xgb_inference, "is_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_confirmation", lambda features, **kwargs: (0.1, 0.1, 0.8),
    )
    monkeypatch.setattr(pr_module.xgb_inference, "is_meta_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_meta_confirmation", lambda features, **kwargs: 0.55,
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={
            "xgb_confirmation_enabled": True,
            "xgb_calibration_path": str(xgb_cal_path),
            "xgb_meta_confirmation_enabled": True,
            "xgb_meta_calibration_path": str(meta_cal_path),
        },
    ).generate(pit_view)

    sig = signals[0]
    assert sig.meta["xgb_p_target"] == pytest.approx(0.8)
    assert sig.meta["xgb_meta_p_act"] == pytest.approx(0.55)
    # Meta model wins for calibrated_probability, NOT the 3-class p_target.
    assert sig.meta["calibrated_probability"] == pytest.approx(0.55)
    # ...but quality_fraction/score are still driven by the 3-class blend
    # only -- the meta model never touches scoring, just the Kelly signal.
    assert sig.meta["scoring_mode"] == "rule_based+xgb"


def test_pattern_recognition_xgb_meta_and_3class_independently_toggleable(monkeypatch):
    """The two models must be independently switchable -- enabling one
    must not silently enable or require the other."""
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.xgb_inference, "is_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_confirmation", lambda features, **kwargs: (0.1, 0.1, 0.8),
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(params={"xgb_confirmation_enabled": True}).generate(pit_view)

    assert signals[0].meta["xgb_p_target"] == pytest.approx(0.8)
    assert signals[0].meta["xgb_meta_p_act"] is None


def test_pattern_recognition_xgb_meta_features_built_even_when_3class_disabled(monkeypatch):
    """xgb_features_needed must trigger on xgb_meta_confirmation_enabled
    ALONE -- market-proxy construction and build_features must not be
    silently skipped just because the (separately-toggled) 3-class ensemble
    is off."""
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.xgb_inference, "is_available", lambda: False)
    monkeypatch.setattr(pr_module.xgb_inference, "is_meta_available", lambda: True)
    captured = {}

    def _capturing_meta_score(features, **kwargs):
        captured["called"] = True
        return 0.9

    monkeypatch.setattr(pr_module.xgb_inference, "score_pattern_meta_confirmation", _capturing_meta_score)

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(params={"xgb_meta_confirmation_enabled": True}).generate(pit_view)

    assert captured.get("called") is True
    assert signals[0].meta["xgb_meta_p_act"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Meta-confidence hard gate (Workstream D harness, 2026-09-27 -- see
# xgb_meta_min_confidence's own docstring in default_params). Unlike
# calibrated_probability (a down-weight input to Kelly sizing only), this
# knob DROPS a match from the emitted signal set entirely.
# ---------------------------------------------------------------------------

def test_pattern_recognition_meta_min_confidence_none_is_noop(monkeypatch):
    """Default (None) must not drop anything, even with the meta model on
    and scoring low -- every existing caller/test is unaffected."""
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.xgb_inference, "is_meta_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_meta_confirmation", lambda features, **kwargs: 0.1,
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={"xgb_meta_confirmation_enabled": True, "xgb_meta_min_confidence": None},
    ).generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["xgb_meta_p_act"] == pytest.approx(0.1)


def test_pattern_recognition_meta_min_confidence_drops_below_threshold(monkeypatch):
    """A match scored below the threshold must be DROPPED, not merely
    down-weighted -- this is the lever that lets a walk-forward test whether
    trading only a high-confidence subset is itself profitable."""
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.xgb_inference, "is_meta_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_meta_confirmation", lambda features, **kwargs: 0.40,
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={"xgb_meta_confirmation_enabled": True, "xgb_meta_min_confidence": 0.60},
    ).generate(pit_view)

    assert signals == []


def test_pattern_recognition_meta_min_confidence_keeps_above_threshold(monkeypatch):
    """A match scored at or above the threshold passes through unchanged."""
    from firm.strategies import pattern_recognition as pr_module

    monkeypatch.setattr(pr_module.xgb_inference, "is_meta_available", lambda: True)
    monkeypatch.setattr(
        pr_module.xgb_inference, "score_pattern_meta_confirmation", lambda features, **kwargs: 0.80,
    )

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={"xgb_meta_confirmation_enabled": True, "xgb_meta_min_confidence": 0.60},
    ).generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["xgb_meta_p_act"] == pytest.approx(0.80)


def test_pattern_recognition_meta_min_confidence_never_gates_without_meta_score(monkeypatch):
    """A signal with no xgb_meta_p_act at all (meta model off/unavailable)
    must never be gated by this knob -- it only ever applies to matches the
    meta model actually scored."""
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    # xgb_meta_confirmation_enabled left at its False default -- meta never
    # scores anything -- but the gate threshold is set anyway.
    signals = PatternRecognitionStrategy(
        params={"xgb_meta_min_confidence": 0.99},
    ).generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["xgb_meta_p_act"] is None


def test_pattern_recognition_xgb_ensemble_off_by_default_leaves_meta_none():
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy().generate(pit_view)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.meta["xgb_p_target"] is None
    assert sig.meta["calibrated_probability"] is None
    assert "+xgb" not in sig.meta["scoring_mode"]


# ---------------------------------------------------------------------------
# 2026-09-27 Part A: statistical-significance test against a matched-
# volatility null (firm.patterns.significance). Off by default.
# ---------------------------------------------------------------------------

def test_pattern_recognition_significance_test_off_by_default_leaves_meta_none():
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy().generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["significance_p_value"] is None


def test_pattern_recognition_significance_test_rejects_insignificant_match(monkeypatch):
    from firm.strategies import pattern_recognition as pr_module

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    # Force a high (insignificant) p-value regardless of the real null.
    monkeypatch.setattr(
        pr_module.significance, "cached_null_score_distribution",
        lambda *a, **k: __import__("numpy").array([90.0, 91.0, 92.0]),
    )
    monkeypatch.setattr(pr_module.significance, "pattern_p_value", lambda score, null: 0.99)

    signals = PatternRecognitionStrategy(
        params={"significance_test_enabled": True},
    ).generate(pit_view)

    assert signals == []


def test_pattern_recognition_significance_test_accepts_significant_match(monkeypatch):
    from firm.strategies import pattern_recognition as pr_module

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    monkeypatch.setattr(
        pr_module.significance, "cached_null_score_distribution",
        lambda *a, **k: __import__("numpy").array([10.0, 20.0, 30.0]),
    )
    monkeypatch.setattr(pr_module.significance, "pattern_p_value", lambda score, null: 0.01)

    signals = PatternRecognitionStrategy(
        params={"significance_test_enabled": True, "significance_max_p_value": 0.05},
    ).generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["significance_p_value"] == pytest.approx(0.01)


def test_pattern_recognition_significance_test_failure_degrades_to_none_not_crash(monkeypatch):
    """A significance-test exception (e.g. scan_symbol raising on a
    pathological surrogate) must not take down the whole strategy --
    degrades to p_value=None (not rejected), same fail-soft convention as
    the CNN/XGBoost layers."""
    from firm.strategies import pattern_recognition as pr_module

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(pr_module.significance, "cached_null_score_distribution", _boom)

    signals = PatternRecognitionStrategy(
        params={"significance_test_enabled": True},
    ).generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["significance_p_value"] is None


def test_pattern_recognition_significance_applies_fdr_across_symbols_not_a_naive_per_symbol_threshold(monkeypatch):
    """The crux of the FDR-control design: two candidates with p-values
    (0.04, 0.06) would BOTH clear a naive per-symbol threshold of 0.05 in
    AAA's case (0.04 < 0.05) and both would be borderline -- but under
    real Benjamini-Hochberg correction for m=2 simultaneous tests at
    q=0.05, the BH thresholds are (1/2)*0.05=0.025 and (2/2)*0.05=0.05, so
    NEITHER 0.04 nor 0.06 clears its own rank's threshold and BOTH are
    correctly rejected -- demonstrating this is a real cross-sectional
    correction, not just re-deriving the same per-symbol accept/reject
    decision a naive threshold would already give."""
    from firm.strategies import pattern_recognition as pr_module

    prices_df = pd.concat([
        _build_prices_df("AAA", _BULL_FLAG_ANCHORS, 31, spike_at=30),
        _build_prices_df("BBB", _BULL_FLAG_ANCHORS, 31, spike_at=30),
    ], ignore_index=True)
    pit_view = _FakePitView(prices_df, ["AAA", "BBB"], datetime(2024, 3, 1))

    p_values = iter([0.04, 0.06])  # groupby("symbol") -> alphabetical: AAA then BBB
    monkeypatch.setattr(
        pr_module.significance, "cached_null_score_distribution",
        lambda *a, **k: np.array([1.0]),  # never inspected; pattern_p_value is mocked directly
    )
    monkeypatch.setattr(pr_module.significance, "pattern_p_value", lambda score, null: next(p_values))

    signals = PatternRecognitionStrategy(
        params={"significance_test_enabled": True, "significance_max_p_value": 0.05},
    ).generate(pit_view)

    assert signals == []  # both rejected under real BH correction, not just BBB


def test_pattern_recognition_significance_fdr_accepts_a_genuinely_significant_pair(monkeypatch):
    """Sanity complement to the test above: two candidates with p-values
    (0.005, 0.01) both clear BH's thresholds at m=2, q=0.05
    ((1/2)*0.05=0.025, (2/2)*0.05=0.05) -- both accepted."""
    from firm.strategies import pattern_recognition as pr_module

    prices_df = pd.concat([
        _build_prices_df("AAA", _BULL_FLAG_ANCHORS, 31, spike_at=30),
        _build_prices_df("BBB", _BULL_FLAG_ANCHORS, 31, spike_at=30),
    ], ignore_index=True)
    pit_view = _FakePitView(prices_df, ["AAA", "BBB"], datetime(2024, 3, 1))

    p_values = iter([0.005, 0.01])
    monkeypatch.setattr(
        pr_module.significance, "cached_null_score_distribution",
        lambda *a, **k: np.array([1.0]),
    )
    monkeypatch.setattr(pr_module.significance, "pattern_p_value", lambda score, null: next(p_values))

    signals = PatternRecognitionStrategy(
        params={"significance_test_enabled": True, "significance_max_p_value": 0.05},
    ).generate(pit_view)

    assert {s.symbol for s in signals} == {"AAA", "BBB"}


# ---------------------------------------------------------------------------
# 2026-09-27 Part A: per-pattern minimum-sample confidence discount
# (firm.patterns.sample_size). Off by default.
# ---------------------------------------------------------------------------

def test_pattern_recognition_sample_discount_off_by_default_leaves_meta_at_one():
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy().generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["sample_size_discount"] == 1.0


def test_pattern_recognition_sample_discount_shrinks_rare_pattern_confidence(tmp_path):
    from firm.patterns.sample_size import save_sample_counts

    counts_path = tmp_path / "pattern_sample_counts.json"
    # _BULL_FLAG_ANCHORS confirms as bull_flag or pennant (family-ambiguous,
    # see tests/pattern_fixtures.py) -- give it a heavily-discounted count.
    save_sample_counts({"bull_flag": 1, "pennant": 1}, counts_path)

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    baseline = PatternRecognitionStrategy().generate(pit_view)
    discounted = PatternRecognitionStrategy(
        params={
            "sample_size_discount_enabled": True,
            "sample_counts_path": str(counts_path),
            "min_reliable_samples": 30,
        },
    ).generate(pit_view)

    assert len(baseline) == 1 and len(discounted) == 1
    assert discounted[0].meta["sample_size_discount"] == pytest.approx(1 / 30)
    assert abs(discounted[0].confidence) == pytest.approx(baseline[0].confidence * (1 / 30))
    assert abs(discounted[0].score) < abs(baseline[0].score)


def test_pattern_recognition_sample_discount_no_file_is_a_no_op(tmp_path):
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={
            "sample_size_discount_enabled": True,
            "sample_counts_path": str(tmp_path / "does_not_exist.json"),
        },
    ).generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["sample_size_discount"] == 1.0


def test_pattern_recognition_sample_discount_well_represented_pattern_unaffected(tmp_path):
    from firm.patterns.sample_size import save_sample_counts

    counts_path = tmp_path / "pattern_sample_counts.json"
    save_sample_counts({"bull_flag": 500, "pennant": 500}, counts_path)

    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={
            "sample_size_discount_enabled": True,
            "sample_counts_path": str(counts_path),
            "min_reliable_samples": 30,
        },
    ).generate(pit_view)

    assert signals[0].meta["sample_size_discount"] == 1.0


# ---------------------------------------------------------------------------
# Market-regime-aware confidence discount (Part A item 6)
# ---------------------------------------------------------------------------

class _FakeRegimeState:
    """Stand-in for firm.regime.model.RegimeState -- only the two fields
    _regime_confidence_discount actually reads."""

    def __init__(self, label: str, separation: float = float("inf")):
        self.label = label
        self.separation = separation


class _FakeRegimeDetector:
    """Stand-in for firm.regime.detector.MarketRegimeDetector -- pre-set on
    the strategy instance's ``_regime_detector`` attribute so generate()'s
    lazy-init branch (``if self._regime_detector is None``) is skipped and
    detection is fully deterministic, without needing 120+ bars of real
    history or a real HMM fit (this is the same monkeypatch-the-collaborator
    idiom used for cnn_inference/xgb_inference elsewhere in this file)."""

    def __init__(self, state):
        self._state = state

    def detect(self, pit_view):
        return self._state


class TestRegimeConfidenceDiscount:
    """Pure unit tests for _regime_confidence_discount -- exhaustive over
    the aligned/misaligned/Chop branches and the separation-based damping,
    independent of any real HMM fit."""

    def test_long_pattern_in_bull_regime_is_aligned_no_discount(self):
        from firm.strategies.pattern_recognition import _regime_confidence_discount

        state = _FakeRegimeState("Bull", separation=float("inf"))
        discount = _regime_confidence_discount(
            state, "long", chop_discount=0.85, misaligned_discount=0.7,
            min_separation=0.5, separation_damping_floor=0.15,
        )
        assert discount == 1.0

    def test_short_pattern_in_bear_regime_is_aligned_no_discount(self):
        from firm.strategies.pattern_recognition import _regime_confidence_discount

        state = _FakeRegimeState("Bear", separation=float("inf"))
        discount = _regime_confidence_discount(
            state, "short", chop_discount=0.85, misaligned_discount=0.7,
            min_separation=0.5, separation_damping_floor=0.15,
        )
        assert discount == 1.0

    def test_long_pattern_in_bear_regime_is_misaligned_full_discount_at_full_separation(self):
        from firm.strategies.pattern_recognition import _regime_confidence_discount

        state = _FakeRegimeState("Bear", separation=10.0)  # far above threshold
        discount = _regime_confidence_discount(
            state, "long", chop_discount=0.85, misaligned_discount=0.7,
            min_separation=0.5, separation_damping_floor=0.15,
        )
        assert discount == pytest.approx(0.7)

    def test_short_pattern_in_bull_regime_is_misaligned_full_discount_at_full_separation(self):
        from firm.strategies.pattern_recognition import _regime_confidence_discount

        state = _FakeRegimeState("Bull", separation=10.0)
        discount = _regime_confidence_discount(
            state, "short", chop_discount=0.85, misaligned_discount=0.7,
            min_separation=0.5, separation_damping_floor=0.15,
        )
        assert discount == pytest.approx(0.7)

    def test_chop_discount_applies_regardless_of_direction(self):
        from firm.strategies.pattern_recognition import _regime_confidence_discount

        for direction in ("long", "short"):
            state = _FakeRegimeState("Chop", separation=float("inf"))
            discount = _regime_confidence_discount(
                state, direction, chop_discount=0.85, misaligned_discount=0.7,
                min_separation=0.5, separation_damping_floor=0.15,
            )
            assert discount == pytest.approx(0.85)

    def test_thin_separation_damps_misaligned_discount_toward_floor(self):
        from firm.strategies.pattern_recognition import _regime_confidence_discount

        # separation == 0 -> damping clamps to separation_damping_floor
        state = _FakeRegimeState("Bear", separation=0.0)
        discount = _regime_confidence_discount(
            state, "long", chop_discount=0.85, misaligned_discount=0.7,
            min_separation=0.5, separation_damping_floor=0.15,
        )
        # damping=0.15 -> discount = 1 + (0.7-1)*0.15 = 0.955
        assert discount == pytest.approx(1.0 + (0.7 - 1.0) * 0.15)

    def test_infinite_separation_is_never_damped(self):
        from firm.strategies.pattern_recognition import _regime_confidence_discount

        state = _FakeRegimeState("Bear", separation=float("inf"))
        discount = _regime_confidence_discount(
            state, "long", chop_discount=0.85, misaligned_discount=0.7,
            min_separation=0.5, separation_damping_floor=0.15,
        )
        assert discount == pytest.approx(0.7)

    def test_zero_min_separation_is_a_no_op_not_divide_by_zero(self):
        from firm.strategies.pattern_recognition import _regime_confidence_discount

        state = _FakeRegimeState("Bear", separation=0.0)
        discount = _regime_confidence_discount(
            state, "long", chop_discount=0.85, misaligned_discount=0.7,
            min_separation=0.0, separation_damping_floor=0.15,
        )
        assert discount == pytest.approx(0.7)

    def test_misaligned_discount_is_clipped_to_valid_range(self):
        from firm.strategies.pattern_recognition import _regime_confidence_discount

        state = _FakeRegimeState("Bear", separation=10.0)
        discount = _regime_confidence_discount(
            state, "long", chop_discount=0.85, misaligned_discount=1.5,
            min_separation=0.5, separation_damping_floor=0.15,
        )
        assert discount == 1.0


def test_pattern_recognition_regime_discount_off_by_default_leaves_meta_at_one():
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy().generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["regime_discount"] == 1.0
    assert signals[0].meta["regime_label"] is None


def test_pattern_recognition_regime_discount_unavailable_is_a_no_op():
    # min_data_points defaults to 120; this fixture only has 31 bars, so the
    # real MarketRegimeDetector.detect() returns None (insufficient history)
    # -- exercising the actual fail-soft path, not a mock.
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    signals = PatternRecognitionStrategy(
        params={"regime_discount_enabled": True},
    ).generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["regime_discount"] == 1.0
    assert signals[0].meta["regime_label"] is None


def test_pattern_recognition_regime_discount_shrinks_misaligned_pattern():
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    baseline = PatternRecognitionStrategy().generate(pit_view)
    assert len(baseline) == 1 and baseline[0].meta["direction"] == "long"

    strategy = PatternRecognitionStrategy(
        params={"regime_discount_enabled": True, "regime_misaligned_discount": 0.7},
    )
    strategy._regime_detector = _FakeRegimeDetector(_FakeRegimeState("Bear", separation=10.0))
    discounted = strategy.generate(pit_view)

    assert len(discounted) == 1
    assert discounted[0].meta["regime_label"] == "Bear"
    assert discounted[0].meta["regime_discount"] == pytest.approx(0.7)
    assert abs(discounted[0].confidence) == pytest.approx(baseline[0].confidence * 0.7)
    assert abs(discounted[0].score) < abs(baseline[0].score)


def test_pattern_recognition_regime_discount_leaves_aligned_pattern_unaffected():
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    baseline = PatternRecognitionStrategy().generate(pit_view)
    assert len(baseline) == 1 and baseline[0].meta["direction"] == "long"

    strategy = PatternRecognitionStrategy(params={"regime_discount_enabled": True})
    strategy._regime_detector = _FakeRegimeDetector(_FakeRegimeState("Bull", separation=10.0))
    aligned = strategy.generate(pit_view)

    assert len(aligned) == 1
    assert aligned[0].meta["regime_label"] == "Bull"
    assert aligned[0].meta["regime_discount"] == 1.0
    assert aligned[0].confidence == pytest.approx(baseline[0].confidence)


def test_pattern_recognition_regime_discount_applies_chop_discount_regardless_of_direction():
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    baseline = PatternRecognitionStrategy().generate(pit_view)

    strategy = PatternRecognitionStrategy(
        params={"regime_discount_enabled": True, "regime_chop_discount": 0.85},
    )
    strategy._regime_detector = _FakeRegimeDetector(_FakeRegimeState("Chop"))
    signals = strategy.generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["regime_label"] == "Chop"
    assert signals[0].meta["regime_discount"] == pytest.approx(0.85)
    assert abs(signals[0].confidence) == pytest.approx(baseline[0].confidence * 0.85)


def test_pattern_recognition_regime_detection_failure_is_a_no_op(monkeypatch):
    prices_df = _build_prices_df("AAPL", _BULL_FLAG_ANCHORS, 31, spike_at=30)
    pit_view = _FakePitView(prices_df, ["AAPL"], datetime(2024, 3, 1))

    class _RaisingDetector:
        def detect(self, pit_view):
            raise RuntimeError("boom")

    strategy = PatternRecognitionStrategy(params={"regime_discount_enabled": True})
    strategy._regime_detector = _RaisingDetector()
    signals = strategy.generate(pit_view)

    assert len(signals) == 1
    assert signals[0].meta["regime_discount"] == 1.0
    assert signals[0].meta["regime_label"] is None

