"""Cup & Handle and Rounding Bottom — the one pattern family that isn't
defined by pivot geometry at all, but by a smooth concave-up curve fit
across the whole span between two rim peaks.

Both patterns share the same cup validation (rim symmetry, depth, and a
degree-2 polynomial fit requiring genuine concavity); they differ only in
whether a shallow "handle" pullback forms after the right rim before the
breakout. When no valid handle forms, the direct breakout above the right
rim is scored as a Rounding Bottom instead — the classical pattern's own
definition doesn't require a handle.

Tries a few recent pivot windows (see
:func:`firm.patterns.extrema.recent_pivot_windows`) rather than assuming
``pivots[-3:]`` is always ``(left_rim, bottom, right_rim)`` — a handle's own
pullback-then-bounce is often large enough to register as its own confirmed
pivot, which would otherwise push the right rim out of the last-3 window.
"""

from __future__ import annotations

import numpy as np

from firm.patterns.extrema import Pivot, recent_pivot_windows
from firm.patterns.match import PatternMatch
from firm.patterns.trendline import fit_poly2

_MAX_PIVOT_LOOKBACK = 3


def _cup_handle(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: list[Pivot],
    n: int,
    *,
    rim_tolerance: float,
    min_depth: float,
    max_depth: float,
    min_poly_r2: float,
    min_handle_bars: int,
    max_handle_bars: int,
    max_handle_retrace: float,
    confirm_lookback_bars: int,
) -> PatternMatch | None:
    left_rim, bottom, right_rim = window
    if bottom.kind != "trough":
        return None

    avg_rim = (left_rim.price + right_rim.price) / 2.0
    if avg_rim <= 0:
        return None
    rim_diff = abs(left_rim.price - right_rim.price) / avg_rim
    if rim_diff > rim_tolerance:
        return None
    depth = (avg_rim - bottom.price) / avg_rim
    if not (min_depth <= depth <= max_depth):
        return None

    span_close = close[left_rim.index : right_rim.index + 1]
    if len(span_close) < 10:
        return None
    a, _b, _c, r2 = fit_poly2(span_close)
    if a <= 0 or r2 < min_poly_r2:
        return None  # not concave-up enough to be a "U", not a "V"

    cup_depth_abs = avg_rim - bottom.price
    if cup_depth_abs <= 0:
        return None

    # The handle (if any) is whatever comes strictly *before* the breakout —
    # using a "most recent bar past the level" search here would let
    # already-broken-out bars leak into the handle-high measurement, which
    # by definition exceeds the rim and would always fail the "handle stays
    # below the rim" check. So find the first close back above the rim
    # explicitly, and only consider bars before it part of the handle.
    level = right_rim.price
    handle_start = right_rim.index + 1
    first_breach = next((i for i in range(handle_start, n) if close[i] > level), None)
    if first_breach is None or n - 1 - first_breach >= confirm_lookback_bars:
        return None  # never broke out, or broke out too long ago to be fresh
    confirm_index = first_breach

    handle_bars = confirm_index - handle_start
    handle_valid = False
    if min_handle_bars <= handle_bars <= max_handle_bars:
        handle_low = float(low[handle_start:confirm_index].min())
        handle_high = float(high[handle_start:confirm_index].max())
        if handle_high <= right_rim.price and handle_low > bottom.price:
            handle_retrace = (right_rim.price - handle_low) / cup_depth_abs
            handle_valid = handle_retrace <= max_handle_retrace

    return PatternMatch(
        pattern="cup_handle" if handle_valid else "rounding_bottom",
        direction="long",
        pivots=(left_rim, bottom, right_rim),
        confirm_index=confirm_index,
        entry=level,
        stop=bottom.price,
        target=level + cup_depth_abs,
        fit_quality=r2,
        geometry_tolerance_used=max(0.0, 1.0 - rim_diff / rim_tolerance),
        meta={
            "cup_depth_pct": depth,
            "poly_a": a,
            "handle_bars": handle_bars,
            "handle_valid": handle_valid,
        },
    )


def detect_cup_handle(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivots: list[Pivot],
    *,
    rim_tolerance: float = 0.08,
    min_depth: float = 0.12,
    max_depth: float = 0.50,
    min_poly_r2: float = 0.75,
    min_handle_bars: int = 3,
    max_handle_bars: int = 15,
    max_handle_retrace: float = 0.5,
    confirm_lookback_bars: int = 3,
) -> list[PatternMatch]:
    n = len(close)
    matches: list[PatternMatch] = []
    for window in recent_pivot_windows(pivots, 3, _MAX_PIVOT_LOOKBACK):
        if window[-1].kind != "peak":
            continue
        match = _cup_handle(
            high, low, close, window, n,
            rim_tolerance=rim_tolerance, min_depth=min_depth, max_depth=max_depth,
            min_poly_r2=min_poly_r2, min_handle_bars=min_handle_bars,
            max_handle_bars=max_handle_bars, max_handle_retrace=max_handle_retrace,
            confirm_lookback_bars=confirm_lookback_bars,
        )
        if match is not None:
            matches.append(match)
    return matches
