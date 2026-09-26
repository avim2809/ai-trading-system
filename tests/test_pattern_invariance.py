"""Future-append / prefix-invariance regression tests for the pattern
detection pipeline (firm.patterns.extrema.zigzag_pivots, firm.patterns.scanner.scan_symbol).

The invariant under test: for any decision time T, appending observations
after T must not change any pattern/signal result *emitted at or before T*.
This is the P0 correctness property manually verified (function-by-function,
no automated coverage) in the 2026-09-25 pattern-recognition adversarial
review -- this file is the missing regression coverage for that finding.

Fixtures use the same anchor-interpolation idiom as tests/test_patterns.py
(hand-built via linear interpolation between chosen anchor points, not a
random walk, so a specific pattern is deterministically present).

Two genuine subtleties were found while writing these tests and are handled
explicitly below rather than papered over:

1. ``scanner._score_and_finalize`` floors ``stop`` at ``stop_atr_floor *
   current_atr`` where ``current_atr = atr[-1]`` -- the *last bar of the
   whole input window*, not the ATR as of ``confirm_index``. Intentional
   adaptive-stop design (the stop reflects *current* volatility, not
   volatility when the pattern formed), not a look-ahead bug -- live trading
   never re-scans the past, so no previously *emitted* signal is ever
   altered. But ``stop``, ``risk_reward``, ``follow_through_atr``,
   ``quality_score``, and ``score_breakdown`` are consequently *not*
   prefix-invariant by design and are excluded from the structural-equality
   checks below.
2. ``confirmation.find_confirmation`` searches *newest-first* within its
   lookback window and returns the freshest bar that still satisfies the
   breakout condition (see its own docstring: "the most recent such close").
   This means ``confirm_index`` deliberately *drifts forward* as more bars
   are appended, for as long as the breakout persists -- comparing
   ``scan_symbol(data[:T])`` against ``scan_symbol(data[:T+N])`` directly
   and expecting the *same* ``confirm_index`` to reappear is therefore the
   wrong test (it would fail on correct-by-design behavior, not a bug). The
   right test instead: find a match on the *longer* series, note the
   ``confirm_index`` it actually reports, then truncate to exactly
   ``data[:confirm_index+1]`` and re-scan -- everything strictly after
   ``confirm_index`` must have been irrelevant to that match's structural
   fields, which is the real "no look-ahead beyond confirmation" property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from firm.patterns.extrema import zigzag_pivots
from firm.patterns.scanner import scan_symbol

_STRUCTURAL_FIELDS = (
    "pattern",
    "direction",
    "confirm_index",
    "entry",
    "target",
    "pivots",
    "fit_quality",
    "geometry_tolerance_used",
    "volume_ratio",
    "duration_bars",
)


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


def _structural(match) -> tuple:
    return tuple(getattr(match, f) for f in _STRUCTURAL_FIELDS)


def _find(matches, *, pattern: str, confirm_index: int | None = None):
    candidates = [m for m in matches if m.pattern == pattern]
    if confirm_index is not None:
        candidates = [m for m in candidates if m.confirm_index == confirm_index]
    # Best (highest quality_score) first -- matches are already sorted that
    # way by scan_symbol, but be explicit since callers may re-derive a
    # confirm_index from this result.
    return candidates[0]


# A double-top confirmed mid-series, with a long decline afterward so the
# fixture genuinely has bars strictly after confirmation to test against.
_DOUBLE_TOP_ANCHORS = [(0, 90.0), (10, 120.0), (20, 100.0), (30, 121.0), (45, 85.0), (75, 60.0)]
_TOTAL_BARS = 76


def _full_df() -> pd.DataFrame:
    return _frame(_DOUBLE_TOP_ANCHORS, _TOTAL_BARS, spike_at=45)


def test_scan_symbol_unaffected_by_bars_after_confirmation():
    full_df = _full_df()
    full_matches = scan_symbol(full_df, min_score=0.0, zigzag_pct=0.03)
    top = _find(full_matches, pattern="double_top")

    truncated_df = full_df.iloc[: top.confirm_index + 1]
    truncated_matches = scan_symbol(truncated_df, min_score=0.0, zigzag_pct=0.03)
    truncated_top = _find(truncated_matches, pattern="double_top", confirm_index=top.confirm_index)

    assert _structural(truncated_top) == _structural(top)


def test_scan_symbol_unaffected_by_bars_after_confirmation_with_modifiers_enabled():
    # retest_outcome/confluence_modifier are *supposed* to use bars after
    # confirm_index (that's their entire purpose) -- so with modifiers on,
    # quality_score/score_breakdown will legitimately differ between the
    # full and truncated runs. The structural fields must not.
    full_df = _full_df()
    dates = pd.date_range("2024-01-01", periods=_TOTAL_BARS, freq="B")
    kwargs = dict(min_score=0.0, zigzag_pct=0.03, retest_modifier_enabled=True, confluence_modifier_enabled=True)

    full_matches = scan_symbol(full_df, dates=dates, **kwargs)
    top = _find(full_matches, pattern="double_top")

    truncated_df = full_df.iloc[: top.confirm_index + 1]
    truncated_dates = dates[: top.confirm_index + 1]
    truncated_matches = scan_symbol(truncated_df, dates=truncated_dates, **kwargs)
    truncated_top = _find(truncated_matches, pattern="double_top", confirm_index=top.confirm_index)

    assert _structural(truncated_top) == _structural(top)


def test_scan_symbol_confirm_index_unaffected_by_a_single_non_crossing_bar():
    # One extra bar that does NOT satisfy the breakout-direction check (a
    # bounce back above the broken level) must not shift confirm_index
    # forward -- find_confirmation's newest-first search should fall
    # through to the same confirmation bar as before.
    full_df = _full_df()
    full_matches = scan_symbol(full_df, min_score=0.0, zigzag_pct=0.03)
    top = _find(full_matches, pattern="double_top")

    truncated_df = full_df.iloc[: top.confirm_index + 1]
    bounce_row = truncated_df.iloc[[-1]].copy()
    bounce_row.iloc[0, bounce_row.columns.get_loc("close")] = top.entry * 1.10  # back above the neckline
    bounce_row.iloc[0, bounce_row.columns.get_loc("high")] = bounce_row.iloc[0]["close"] * 1.002
    bounce_row.iloc[0, bounce_row.columns.get_loc("low")] = bounce_row.iloc[0]["close"] * 0.998
    with_bounce_df = pd.concat([truncated_df, bounce_row], ignore_index=True)

    with_bounce_matches = scan_symbol(with_bounce_df, min_score=0.0, zigzag_pct=0.03)
    with_bounce_top = _find(with_bounce_matches, pattern="double_top", confirm_index=top.confirm_index)

    assert _structural(with_bounce_top) == _structural(top)


def test_zigzag_pivots_confirmed_prefix_is_stable_under_future_append():
    # Every anchor below is followed by a further >=3% reversal *within
    # the short window itself*, so every pivot in short_pivots is
    # unambiguously already-confirmed (not "still forming") -- the
    # meaningful invariant is that extending the series further can only
    # *append* new confirmed pivots, never alter the ones already found.
    anchors = [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0), (40, 115.0), (50, 145.0)]
    extended_anchors = anchors + [(60, 120.0), (70, 150.0)]

    short_high, short_low, _, _ = _ohlcv(anchors, 51)
    extended_high, extended_low, _, _ = _ohlcv(extended_anchors, 71)

    short_pivots = zigzag_pivots(short_high, short_low, pct=0.03)
    extended_pivots = zigzag_pivots(extended_high, extended_low, pct=0.03)

    assert len(short_pivots) >= 3  # sanity: the fixture found real pivots
    assert extended_pivots[: len(short_pivots)] == short_pivots
