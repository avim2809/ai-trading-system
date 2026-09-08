"""Triangle, wedge and rectangle patterns — all fit from the same two
trendlines (resistance through recent peaks, support through recent
troughs), classified purely by the sign/relative-magnitude of their slopes.

Ascending/descending triangles and rectangles have a fixed breakout
direction implied by their shape; symmetrical triangles and rectangles can
break either way, so both boundaries are checked and whichever confirms
(most recently, if both do) wins.

Tries a few recent pivot windows (see
:func:`firm.patterns.extrema.recent_pivot_windows`) rather than assuming
``pivots[-window:]`` is always the right slice — same reasoning as
``rules/reversal.py``/``rules/cup_handle.py``: a noise pivot from the
eventual breakout can land inside a naive trailing window and skew the
fitted slope enough to misclassify (or miss) the pattern.
"""

from __future__ import annotations

import numpy as np

from firm.patterns.confirmation import find_confirmation
from firm.patterns.extrema import Pivot, recent_pivot_windows
from firm.patterns.match import PatternMatch
from firm.patterns.trendline import fit_trendline

# Per-bar slope, normalised by average price, treated as "flat" (resistance
# or support essentially horizontal) below this fraction.
_FLAT_EPS = 0.0015
# Minimum extra normalised-slope gap required between the two lines of a
# wedge before "converging" is trusted over fit noise.
_CONVERGE_MARGIN = 0.0008
_MAX_PIVOT_LOOKBACK = 3

_DUAL_DIRECTION = {"rectangle", "symmetrical_triangle"}


def _triangle(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window_pivots: list[Pivot],
    *,
    flat_eps: float,
    converge_margin: float,
    confirm_lookback_bars: int,
) -> PatternMatch | None:
    peaks = [p for p in window_pivots if p.kind == "peak"]
    troughs = [p for p in window_pivots if p.kind == "trough"]
    if len(peaks) < 2 or len(troughs) < 2:
        return None

    resistance = fit_trendline([p.index for p in peaks], [p.price for p in peaks])
    support = fit_trendline([p.index for p in troughs], [p.price for p in troughs])
    if resistance is None or support is None:
        return None

    avg_price = float(np.mean([p.price for p in window_pivots]))
    if avg_price <= 0:
        return None
    r_slope = resistance.slope / avg_price
    s_slope = support.slope / avg_price

    start_index = window_pivots[0].index
    last_pivot_index = window_pivots[-1].index
    width_at_start = resistance.value_at(start_index) - support.value_at(start_index)
    if width_at_start <= 0:
        return None  # lines have already crossed — not a valid channel

    pattern_name: str | None
    direction: str | None
    if abs(r_slope) <= flat_eps and s_slope > flat_eps:
        pattern_name, direction = "ascending_triangle", "long"
    elif abs(s_slope) <= flat_eps and r_slope < -flat_eps:
        pattern_name, direction = "descending_triangle", "short"
    elif abs(r_slope) <= flat_eps and abs(s_slope) <= flat_eps:
        pattern_name, direction = "rectangle", None
    elif r_slope < -flat_eps and s_slope > flat_eps:
        pattern_name, direction = "symmetrical_triangle", None
    elif r_slope > flat_eps and s_slope > flat_eps and s_slope > r_slope + converge_margin:
        pattern_name, direction = "rising_wedge", "short"
    elif r_slope < -flat_eps and s_slope < -flat_eps and r_slope < s_slope - converge_margin:
        pattern_name, direction = "falling_wedge", "long"
    else:
        return None

    fit_quality = (resistance.r2 + support.r2) / 2.0

    if pattern_name in _DUAL_DIRECTION:
        up_confirm = find_confirmation(
            close, resistance.value_at, "above", confirm_lookback_bars, min_index=last_pivot_index
        )
        down_confirm = find_confirmation(
            close, support.value_at, "below", confirm_lookback_bars, min_index=last_pivot_index
        )
        if up_confirm < 0 and down_confirm < 0:
            return None
        if up_confirm >= down_confirm:
            confirm_index, direction, level_at = up_confirm, "long", resistance.value_at
        else:
            confirm_index, direction, level_at = down_confirm, "short", support.value_at
    elif direction == "long":
        level_at = resistance.value_at
        confirm_index = find_confirmation(close, level_at, "above", confirm_lookback_bars, min_index=last_pivot_index)
        if confirm_index < 0:
            return None
    else:
        level_at = support.value_at
        confirm_index = find_confirmation(close, level_at, "below", confirm_lookback_bars, min_index=last_pivot_index)
        if confirm_index < 0:
            return None

    entry = level_at(confirm_index)
    target = entry + width_at_start if direction == "long" else entry - width_at_start
    stop = troughs[-1].price if direction == "long" else peaks[-1].price

    return PatternMatch(
        pattern=pattern_name,
        direction=direction,
        pivots=tuple(window_pivots),
        confirm_index=confirm_index,
        entry=entry,
        stop=stop,
        target=target,
        fit_quality=fit_quality,
        geometry_tolerance_used=fit_quality,
        meta={
            "resistance_slope": resistance.slope,
            "support_slope": support.slope,
            "width_at_start": width_at_start,
        },
    )


def detect_triangle_wedge_rectangle(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivots: list[Pivot],
    *,
    window: int = 6,
    flat_eps: float = _FLAT_EPS,
    converge_margin: float = _CONVERGE_MARGIN,
    confirm_lookback_bars: int = 3,
) -> PatternMatch | None:
    if len(pivots) < 4:
        return None
    for window_pivots in recent_pivot_windows(pivots, min(window, len(pivots)), _MAX_PIVOT_LOOKBACK):
        match = _triangle(
            high, low, close, window_pivots,
            flat_eps=flat_eps, converge_margin=converge_margin,
            confirm_lookback_bars=confirm_lookback_bars,
        )
        if match is not None:
            return match
    return None
