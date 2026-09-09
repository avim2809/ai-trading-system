"""Continuation patterns: Bull/Bear Flag and Pennant.

Unlike the reversal and triangle families, the consolidation leg here is
usually too short and shallow to register its own zigzag pivots, so its
upper/lower boundaries are fit directly on the raw bars since the flagpole's
end rather than on further pivots.

Naively, "the flagpole" would be ``pivots[-2:]`` (the last confirmed swing).
But once the eventual breakout move is itself large enough, zigzag confirms
the *consolidation's own extreme* as the newest pivot (the breakout is what
finally reverses far enough from it to confirm it) — which pushes the real
flagpole one pivot pair further back, and makes the naive last-pair reading
name the wrong two points as pole_start/pole_end entirely. So this tries a
handful of recent consecutive pivot pairs, most recent first, and returns
the first one that actually validates as a flagpole with a confirmed
breakout — a spurious near-term pair (like the consolidation-extreme case
above) fails validation naturally (its "breakout" direction contradicts the
actual subsequent price action) and the search falls through to the real one.
"""

from __future__ import annotations

import numpy as np

from firm.patterns.confirmation import find_confirmation
from firm.patterns.extrema import Pivot
from firm.patterns.match import PatternMatch
from firm.patterns.trendline import fit_trendline

_FLAT_EPS = 0.0015
_MAX_CONSOLIDATION_RETRACE = 0.6  # consolidation range vs. flagpole size
_MAX_POLE_LOOKBACK = 4  # how many recent consecutive pivot pairs to try


def detect_flag_pennant(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivots: list[Pivot],
    *,
    min_flagpole_pct: float = 0.03,
    flagpole_max_bars: int = 10,
    min_flag_bars: int = 5,
    max_flag_bars: int = 20,
    confirm_lookback_bars: int = 3,
) -> list[PatternMatch]:
    if len(pivots) < 2:
        return []
    n = len(close)
    matches: list[PatternMatch] = []
    earliest_k = max(0, len(pivots) - 1 - _MAX_POLE_LOOKBACK)
    for k in range(len(pivots) - 2, earliest_k - 1, -1):
        match = _try_flag(
            high, low, close, pivots[k], pivots[k + 1], n,
            min_flagpole_pct=min_flagpole_pct,
            flagpole_max_bars=flagpole_max_bars,
            min_flag_bars=min_flag_bars,
            max_flag_bars=max_flag_bars,
            confirm_lookback_bars=confirm_lookback_bars,
        )
        if match is not None:
            matches.append(match)
    return matches


def _try_flag(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pole_start: Pivot,
    pole_end: Pivot,
    n: int,
    *,
    min_flagpole_pct: float,
    flagpole_max_bars: int,
    min_flag_bars: int,
    max_flag_bars: int,
    confirm_lookback_bars: int,
) -> PatternMatch | None:
    flagpole_bars = pole_end.index - pole_start.index
    if flagpole_bars <= 0 or flagpole_bars > flagpole_max_bars:
        return None
    flagpole_move = (pole_end.price - pole_start.price) / pole_start.price if pole_start.price else 0.0
    bullish_pole = flagpole_move >= min_flagpole_pct
    bearish_pole = flagpole_move <= -min_flagpole_pct
    if not (bullish_pole or bearish_pole):
        return None

    # Fit the consolidation boundaries only on bars *before* the confirmation
    # zone — including the breakout bars themselves in the fit (as a naive
    # [consol_start:n] slice would) pulls the trendline toward the breakout,
    # which can make a genuine flat/declining pause look like it was already
    # trending in the breakout's direction and fail the "is this really a
    # pause" check below.
    consol_start = pole_end.index
    fit_end = n - confirm_lookback_bars
    consol_bars = fit_end - consol_start
    if consol_bars < min_flag_bars or consol_bars > max_flag_bars:
        return None

    idx = list(range(consol_start, fit_end))
    highs_seg = high[consol_start:fit_end]
    lows_seg = low[consol_start:fit_end]
    upper = fit_trendline(idx, list(highs_seg))
    lower = fit_trendline(idx, list(lows_seg))
    if upper is None or lower is None:
        return None

    flagpole_height = abs(pole_end.price - pole_start.price)
    consolidation_range = float(highs_seg.max() - lows_seg.min())
    if consolidation_range <= 0 or consolidation_range > _MAX_CONSOLIDATION_RETRACE * flagpole_height:
        return None

    avg_price = float(np.mean(np.concatenate([highs_seg, lows_seg])))
    if avg_price <= 0:
        return None
    upper_slope = upper.slope / avg_price
    lower_slope = lower.slope / avg_price
    converging = upper_slope < -_FLAT_EPS and lower_slope > _FLAT_EPS

    direction = "long" if bullish_pole else "short"
    pattern_name = "pennant" if converging else ("bull_flag" if bullish_pole else "bear_flag")

    # A plain flag (not a pennant) should be a genuine pause, not a
    # continuation of the same move — reject if the relevant boundary is
    # still trending in the pole's direction.
    if pattern_name == "bull_flag" and upper_slope > _FLAT_EPS:
        return None
    if pattern_name == "bear_flag" and lower_slope < -_FLAT_EPS:
        return None

    if direction == "long":
        level_at = upper.value_at
        confirm_index = find_confirmation(close, level_at, "above", confirm_lookback_bars, min_index=consol_start)
    else:
        level_at = lower.value_at
        confirm_index = find_confirmation(close, level_at, "below", confirm_lookback_bars, min_index=consol_start)
    if confirm_index < 0:
        return None

    entry = level_at(confirm_index)
    target = entry + flagpole_height if direction == "long" else entry - flagpole_height
    stop = float(lows_seg.min()) if direction == "long" else float(highs_seg.max())
    fit_quality = (upper.r2 + lower.r2) / 2.0
    tightness = max(0.0, 1.0 - consolidation_range / (_MAX_CONSOLIDATION_RETRACE * flagpole_height))

    return PatternMatch(
        pattern=pattern_name,
        direction=direction,
        pivots=(pole_start, pole_end),
        confirm_index=confirm_index,
        entry=entry,
        stop=stop,
        target=target,
        fit_quality=fit_quality,
        geometry_tolerance_used=tightness,
        meta={
            "flagpole_pct": flagpole_move,
            "flagpole_bars": flagpole_bars,
            "consolidation_bars": consol_bars,
        },
    )
