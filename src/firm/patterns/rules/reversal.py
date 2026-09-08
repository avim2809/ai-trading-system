"""Reversal patterns: Head & Shoulders, Double/Triple Top & Bottom.

All six patterns here are defined purely on the alternating peak/trough
pivots from :func:`firm.patterns.extrema.zigzag_pivots` — no trendline
fitting needed (the "neckline" for the pair patterns is a single trough/peak
level; for H&S/IHS it's the 2-point line through the two neckline pivots).

Each detector tries a handful of recent windows via
:func:`firm.patterns.extrema.recent_pivot_windows`, most recent first, rather
than assuming the pattern's last structural pivot is always ``pivots[-1]`` —
a big enough post-formation move (the breakout itself, or a bounce after it)
can register its own confirmed reversal and push the real pattern back a
pivot or two.
"""

from __future__ import annotations

import numpy as np

from firm.patterns.confirmation import find_confirmation
from firm.patterns.extrema import Pivot, recent_pivot_windows
from firm.patterns.match import PatternMatch
from firm.patterns.trendline import line_through

_MAX_PIVOT_LOOKBACK = 3


def _tol_score(diff_pct: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, 1.0 - diff_pct / tolerance)


def _pct_diff(a: float, b: float) -> float:
    avg = (a + b) / 2.0
    return abs(a - b) / avg if avg != 0 else float("inf")


def _head_shoulders(
    close: np.ndarray,
    window: list[Pivot],
    *,
    top: bool,
    shoulder_tolerance: float,
    neckline_tolerance: float,
    confirm_lookback_bars: int,
) -> PatternMatch | None:
    ls, t1, head, t2, rs = window
    if top:
        if not (head.price > ls.price and head.price > rs.price):
            return None
    elif not (head.price < ls.price and head.price < rs.price):
        return None

    shoulder_diff = _pct_diff(ls.price, rs.price)
    if shoulder_diff > shoulder_tolerance:
        return None
    neckline_diff = _pct_diff(t1.price, t2.price)
    if neckline_diff > neckline_tolerance:
        return None

    neckline = line_through(t1.index, t1.price, t2.index, t2.price)
    confirm_index = find_confirmation(
        close, neckline.value_at, "below" if top else "above", confirm_lookback_bars, min_index=rs.index
    )
    if confirm_index < 0:
        return None

    entry = neckline.value_at(confirm_index)
    height = (head.price - neckline.value_at(head.index)) if top else (neckline.value_at(head.index) - head.price)
    if height <= 0:
        return None
    geometry = (_tol_score(shoulder_diff, shoulder_tolerance) + _tol_score(neckline_diff, neckline_tolerance)) / 2.0

    return PatternMatch(
        pattern="head_shoulders_top" if top else "inverse_head_shoulders",
        direction="short" if top else "long",
        pivots=(ls, t1, head, t2, rs),
        confirm_index=confirm_index,
        entry=entry,
        stop=rs.price,
        target=(entry - height) if top else (entry + height),
        fit_quality=1.0,  # 2-point neckline always fits exactly
        geometry_tolerance_used=geometry,
        meta={"neckline_slope": neckline.slope, "head_height": height},
    )


def detect_head_shoulders(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivots: list[Pivot],
    *,
    shoulder_tolerance: float = 0.05,
    neckline_tolerance: float = 0.05,
    confirm_lookback_bars: int = 3,
) -> PatternMatch | None:
    """Head & Shoulders top — bearish reversal. ``None`` if not present/confirmed."""
    for window in recent_pivot_windows(pivots, 5, _MAX_PIVOT_LOOKBACK):
        if window[-1].kind != "peak":
            continue
        match = _head_shoulders(
            close, window, top=True, shoulder_tolerance=shoulder_tolerance,
            neckline_tolerance=neckline_tolerance, confirm_lookback_bars=confirm_lookback_bars,
        )
        if match is not None:
            return match
    return None


def detect_inverse_head_shoulders(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivots: list[Pivot],
    *,
    shoulder_tolerance: float = 0.05,
    neckline_tolerance: float = 0.05,
    confirm_lookback_bars: int = 3,
) -> PatternMatch | None:
    """Inverse Head & Shoulders — bullish reversal (mirror of the top)."""
    for window in recent_pivot_windows(pivots, 5, _MAX_PIVOT_LOOKBACK):
        if window[-1].kind != "trough":
            continue
        match = _head_shoulders(
            close, window, top=False, shoulder_tolerance=shoulder_tolerance,
            neckline_tolerance=neckline_tolerance, confirm_lookback_bars=confirm_lookback_bars,
        )
        if match is not None:
            return match
    return None


def _double_pattern(
    window: list[Pivot],
    close: np.ndarray,
    *,
    top: bool,
    tolerance: float,
    min_retrace: float,
    confirm_lookback_bars: int,
) -> PatternMatch | None:
    p1, mid, p2 = window
    diff = _pct_diff(p1.price, p2.price)
    if diff > tolerance:
        return None
    retrace = abs(p1.price - mid.price) / p1.price if p1.price else 0.0
    if retrace < min_retrace:
        return None

    direction_str = "short" if top else "long"
    level = mid.price
    confirm_index = find_confirmation(
        close, lambda _i: level, "below" if top else "above", confirm_lookback_bars, min_index=p2.index
    )
    if confirm_index < 0:
        return None

    avg_peak = (p1.price + p2.price) / 2.0
    height = (avg_peak - level) if top else (level - avg_peak)
    if height <= 0:
        return None
    entry = level
    target = entry - height if top else entry + height
    stop = max(p1.price, p2.price) if top else min(p1.price, p2.price)

    return PatternMatch(
        pattern="double_top" if top else "double_bottom",
        direction=direction_str,
        pivots=(p1, mid, p2),
        confirm_index=confirm_index,
        entry=entry,
        stop=stop,
        target=target,
        fit_quality=1.0,
        geometry_tolerance_used=_tol_score(diff, tolerance),
        meta={"retrace_pct": retrace},
    )


def _detect_double(
    close: np.ndarray,
    pivots: list[Pivot],
    *,
    top: bool,
    tolerance: float,
    min_retrace: float,
    confirm_lookback_bars: int,
) -> PatternMatch | None:
    kind = "peak" if top else "trough"
    for window in recent_pivot_windows(pivots, 3, _MAX_PIVOT_LOOKBACK):
        if window[-1].kind != kind:
            continue
        match = _double_pattern(
            window, close, top=top, tolerance=tolerance, min_retrace=min_retrace,
            confirm_lookback_bars=confirm_lookback_bars,
        )
        if match is not None:
            return match
    return None


def detect_double_top(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivots: list[Pivot],
    *,
    tolerance: float = 0.03,
    min_retrace: float = 0.10,
    confirm_lookback_bars: int = 3,
) -> PatternMatch | None:
    return _detect_double(
        close, pivots, top=True, tolerance=tolerance, min_retrace=min_retrace,
        confirm_lookback_bars=confirm_lookback_bars,
    )


def detect_double_bottom(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivots: list[Pivot],
    *,
    tolerance: float = 0.03,
    min_retrace: float = 0.10,
    confirm_lookback_bars: int = 3,
) -> PatternMatch | None:
    return _detect_double(
        close, pivots, top=False, tolerance=tolerance, min_retrace=min_retrace,
        confirm_lookback_bars=confirm_lookback_bars,
    )


def _triple_pattern(
    window: list[Pivot],
    close: np.ndarray,
    *,
    top: bool,
    outer_tolerance: float,
    inner_tolerance: float,
    confirm_lookback_bars: int,
) -> PatternMatch | None:
    p1, t1, p2, t2, p3 = window
    outer = [p1.price, p2.price, p3.price]
    inner = [t1.price, t2.price]
    outer_mean = sum(outer) / 3.0
    inner_mean = sum(inner) / 2.0
    outer_dev = max(abs(v - outer_mean) for v in outer) / outer_mean if outer_mean else float("inf")
    inner_dev = max(abs(v - inner_mean) for v in inner) / inner_mean if inner_mean else float("inf")
    if outer_dev > outer_tolerance or inner_dev > inner_tolerance:
        return None

    direction_str = "short" if top else "long"
    level = min(t1.price, t2.price) if top else max(t1.price, t2.price)
    confirm_index = find_confirmation(
        close, lambda _i: level, "below" if top else "above", confirm_lookback_bars, min_index=p3.index
    )
    if confirm_index < 0:
        return None

    height = (outer_mean - level) if top else (level - outer_mean)
    if height <= 0:
        return None
    entry = level
    target = entry - height if top else entry + height
    stop = max(outer) if top else min(outer)
    geometry = (_tol_score(outer_dev, outer_tolerance) + _tol_score(inner_dev, inner_tolerance)) / 2.0

    return PatternMatch(
        pattern="triple_top" if top else "triple_bottom",
        direction=direction_str,
        pivots=(p1, t1, p2, t2, p3),
        confirm_index=confirm_index,
        entry=entry,
        stop=stop,
        target=target,
        fit_quality=1.0,
        geometry_tolerance_used=geometry,
        meta={"outer_dev_pct": outer_dev, "inner_dev_pct": inner_dev},
    )


def _detect_triple(
    close: np.ndarray,
    pivots: list[Pivot],
    *,
    top: bool,
    outer_tolerance: float,
    inner_tolerance: float,
    confirm_lookback_bars: int,
) -> PatternMatch | None:
    kind = "peak" if top else "trough"
    for window in recent_pivot_windows(pivots, 5, _MAX_PIVOT_LOOKBACK):
        if window[-1].kind != kind:
            continue
        match = _triple_pattern(
            window, close, top=top, outer_tolerance=outer_tolerance,
            inner_tolerance=inner_tolerance, confirm_lookback_bars=confirm_lookback_bars,
        )
        if match is not None:
            return match
    return None


def detect_triple_top(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivots: list[Pivot],
    *,
    outer_tolerance: float = 0.02,
    inner_tolerance: float = 0.02,
    confirm_lookback_bars: int = 3,
) -> PatternMatch | None:
    return _detect_triple(
        close, pivots, top=True, outer_tolerance=outer_tolerance,
        inner_tolerance=inner_tolerance, confirm_lookback_bars=confirm_lookback_bars,
    )


def detect_triple_bottom(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivots: list[Pivot],
    *,
    outer_tolerance: float = 0.02,
    inner_tolerance: float = 0.02,
    confirm_lookback_bars: int = 3,
) -> PatternMatch | None:
    return _detect_triple(
        close, pivots, top=False, outer_tolerance=outer_tolerance,
        inner_tolerance=inner_tolerance, confirm_lookback_bars=confirm_lookback_bars,
    )
