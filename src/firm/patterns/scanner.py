"""Per-symbol pattern scanning: zigzag pivots -> rule detectors -> quality
scoring -> ATR-floored stops.

Runs every rule detector against one symbol's OHLCV window and returns all
confirmed matches clearing ``min_score``, best first. Each detector itself
returns *every* window it found a valid match in (see
``firm.patterns.extrema.recent_pivot_windows`` and
docs/pattern_recognition_plan.md §6.2) rather than just the first — this is
where those candidates get merged, scored uniformly, and ranked, so a
detector never has to guess which of its own candidate windows is "best."
Callers that want a single signal per symbol (see
docs/pattern_recognition_plan.md deviation #4 — zscore_signals groups by
strategy across the universe, so emitting more than one Signal per symbol
per bar would double-count that symbol) should take the first element.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd

from firm.patterns._indicators import atr14, volume_ratio
from firm.patterns.extrema import zigzag_pivots
from firm.patterns.match import PatternMatch
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
from firm.patterns.scorer import score_pattern

log = logging.getLogger(__name__)

_ALL_DETECTORS = (
    detect_head_shoulders,
    detect_inverse_head_shoulders,
    detect_double_top,
    detect_double_bottom,
    detect_triple_top,
    detect_triple_bottom,
    detect_triangle_wedge_rectangle,
    detect_flag_pennant,
    detect_cup_handle,
)


def scan_symbol(
    df: pd.DataFrame,
    *,
    enabled_patterns: set[str] | None = None,
    zigzag_pct: float = 0.03,
    min_score: float = 60.0,
    confirm_lookback_bars: int = 3,
    stop_atr_floor: float = 1.5,
) -> list[PatternMatch]:
    """Scan one symbol's OHLCV window (ascending by date, columns high/low/
    close/volume) for confirmed, quality-scored patterns.
    """
    if len(df) < 20:
        return []
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    pivots = zigzag_pivots(high, low, pct=zigzag_pct)
    atr = atr14(high, low, close)
    current_atr = float(atr[-1]) if len(atr) and not np.isnan(atr[-1]) else None

    matches: list[PatternMatch] = []
    for detector in _ALL_DETECTORS:
        try:
            candidates = detector(
                high, low, close, volume, pivots,
                confirm_lookback_bars=confirm_lookback_bars,
            )
        except Exception:
            log.debug("pattern detector %s failed", detector.__name__, exc_info=True)
            continue
        for candidate in candidates:
            if enabled_patterns is not None and candidate.pattern not in enabled_patterns:
                continue
            scored = _score_and_finalize(candidate, close, volume, current_atr, stop_atr_floor)
            if scored.quality_score >= min_score:
                matches.append(scored)

    matches.sort(key=lambda m: m.quality_score, reverse=True)
    return matches


def _score_and_finalize(
    candidate: PatternMatch,
    close: np.ndarray,
    volume: np.ndarray,
    current_atr: float | None,
    stop_atr_floor: float,
) -> PatternMatch:
    vr = volume_ratio(volume, candidate.confirm_index)
    duration = candidate.confirm_index - candidate.pivots[0].index

    close_at_confirm = float(close[candidate.confirm_index])
    if current_atr and current_atr > 0:
        follow_through_atr = abs(close_at_confirm - candidate.entry) / current_atr
    else:
        follow_through_atr = 0.0

    score = score_pattern(
        geometry_tolerance_used=candidate.geometry_tolerance_used,
        fit_quality=candidate.fit_quality,
        volume_ratio=vr,
        duration_bars=duration,
        follow_through_atr=follow_through_atr,
    )

    # Floor the structural stop at stop_atr_floor * ATR so a tightly-fit
    # pattern (e.g. a shallow rectangle) never implies a noise-level stop.
    stop = candidate.stop
    if current_atr and current_atr > 0:
        min_distance = stop_atr_floor * current_atr
        if candidate.direction == "long":
            stop = min(stop, candidate.entry - min_distance)
        else:
            stop = max(stop, candidate.entry + min_distance)

    risk = abs(candidate.entry - stop)
    reward = abs(candidate.target - candidate.entry)
    risk_reward = reward / risk if risk > 1e-9 else 0.0

    return dataclasses.replace(
        candidate,
        stop=stop,
        volume_ratio=vr,
        duration_bars=duration,
        follow_through_atr=follow_through_atr,
        risk_reward=risk_reward,
        quality_score=score.total,
        score_breakdown=score.as_dict(),
    )
