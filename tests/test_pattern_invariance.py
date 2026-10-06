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

1. ``scanner._score_and_finalize`` used to floor ``stop`` at
   ``stop_atr_floor * current_atr`` where ``current_atr = atr[-1]`` -- the
   *last bar of the whole input window*, not the ATR as of
   ``confirm_index``. That made some sense while ``confirm_index`` was
   *also* effectively "today" (the pre-2026-09-27 is-beyond confirmation
   bug -- see point 2), since entry and stop were both implicitly anchored
   to "now" together. Now that ``confirm_index`` is a genuine, stable
   historical bar (fix below), leaving the stop floored on ``atr[-1]``
   would have made the stop -- and ``risk_reward`` -- drift purely with
   how many extra bars happen to be in the scanned window, even though
   ``entry`` itself wouldn't move: an inconsistent bracket describing two
   different points in time. Fixed 2026-09-27 (same commit as the
   confirmation fix) to freeze the floor to ``atr[confirm_index]``
   instead -- see ``scanner.py``'s own comment at that line. ``stop`` and
   ``risk_reward`` are consequently now genuinely prefix-invariant too and
   are included in the structural-equality checks below (test 1
   specifically verifies this). ``follow_through_atr``, ``quality_score``,
   and ``score_breakdown`` remain *not* prefix-invariant by design --
   ``follow_through_atr`` deliberately measures drift relative to
   *current* volatility/price (see its own definition in
   ``scanner.py::_score_and_finalize``), and the other two are downstream
   of it -- so those three stay excluded.
2. ``confirmation.find_confirmation`` searches *newest-first* within its
   lookback window, but (2026-09-27 fix -- see that module's docstring for
   the full bug history) now requires a genuine ``close[i-1]``-inside ->
   ``close[i]``-beyond transition, not merely "still beyond." Before that
   fix, this was a plain is-beyond test with no edge condition, so
   ``confirm_index`` was *always* ``n - 1`` (today) for as long as the
   breakout persisted -- which, read literally, made the comparison this
   file exists to make **vacuous**: truncating to ``data[:confirm_index+1]``
   truncated to nothing at all (the confirm_index equalled the full
   series' own last index), so ``truncated_df == full_df`` by construction
   and every "prefix invariance" assertion below was comparing a scan
   against itself. That was itself a symptom of the same bug this test
   module was meant to guard against, not a property of prefix invariance.
   Now that ``confirm_index`` is a genuine historical crossing bar, it's
   still bounded to the last ``lookback_bars`` of *whatever window is
   scanned* (``start = max(min_index + 1, n - lookback_bars)`` in
   ``find_confirmation`` -- unchanged by the fix), so a "confirmed" match on
   the full series still has its ``confirm_index`` within a few bars of the
   series' own end. The fixture below is built so the genuine crossing
   lands with 1-2 real bars still after it in the full series, making the
   truncation non-trivial (a real, if short, comparison) while the sharper
   check remains test 3 below: appending one bar that does *not* satisfy
   the crossing condition (a bounce back above the broken level) must not
   shift ``confirm_index`` at all -- that's the real "no look-ahead /
   no false re-confirmation" property post-fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    # 2026-09-27: now genuinely prefix-invariant -- see point 1 above.
    "stop",
    "risk_reward",
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


# A double-top whose genuine breakdown crossing lands at bar 73 (verified
# against the real detector, not hand-derived) -- inside find_confirmation's
# 3-bar lookback window for this 76-bar series, with bars 74-75 left over as
# real (if few) bars after confirmation. 2026-09-27 (find_confirmation
# edge-trigger fix): the previous anchors ran a single 15-bar ramp straight
# from the last peak (30, 121.0) down to (45, 85.0) and then a further
# decline to (75, 60.0) -- the genuine crossing happened around bar 39, and
# since the tail from 45 to 75 never re-crosses back above the neckline, no
# match is found at all under the corrected semantics (the is-beyond bug
# this file's own module docstring now documents is exactly what used to
# paper over that). Holding just above the neckline (72, 105.0) until a
# short final drop confines the real crossing to the last 3 bars instead.
_DOUBLE_TOP_ANCHORS = [(0, 90.0), (10, 120.0), (20, 100.0), (30, 121.0), (72, 105.0), (75, 60.0)]
_TOTAL_BARS = 76


def _full_df() -> pd.DataFrame:
    return _frame(_DOUBLE_TOP_ANCHORS, _TOTAL_BARS, spike_at=73)  # spike at the real breakout bar


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
