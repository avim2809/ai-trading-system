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
from typing import Literal

import numpy as np
import pandas as pd

from firm.patterns._indicators import atr14, volume_ratio
from firm.patterns.confirmation import retest_outcome, retest_score_modifier
from firm.patterns.confluence import (
    WeeklyTrend,
    confluence_modifier,
    resample_to_weekly,
    weekly_trend_direction,
)
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
    zigzag_atr_mult: float | None = None,
    min_score: float = 60.0,
    confirm_lookback_bars: int = 3,
    stop_atr_floor: float = 1.5,
    retest_modifier_enabled: bool = False,
    retest_lookback_bars: int = 10,
    retest_modifier_scale: float = 3.0,
    confluence_modifier_enabled: bool = False,
    confluence_lookback_weeks: int = 8,
    confluence_modifier_scale: float = 3.0,
    dates: pd.Series | np.ndarray | None = None,
) -> list[PatternMatch]:
    """Scan one symbol's OHLCV window (ascending by date, columns high/low/
    close/volume) for confirmed, quality-scored patterns.

    All new (2026-09) keyword args default to the original fixed-``pct``,
    modifier-free behavior -- every existing caller (including every test
    that predates this change) is unaffected unless it explicitly opts in.

    ``zigzag_atr_mult``: when set to a positive float, replaces the fixed
    ``zigzag_pct`` reversal threshold with a per-bar ATR-scaled one
    (``zigzag_atr_mult * atr[i] / close[i]``) via
    :func:`firm.patterns.extrema.zigzag_pivots`'s ``threshold_fn`` hook --
    see that function's docstring for why a single fixed percentage is a
    poor fit across a multi-symbol, multi-regime universe.

    ``retest_modifier_enabled``/``confluence_modifier_enabled``: fold
    :func:`firm.patterns.confirmation.retest_score_modifier` /
    :func:`firm.patterns.confluence.confluence_modifier` into each match's
    ``quality_score`` as a small, bounded (``+-retest_modifier_scale`` /
    ``+-confluence_modifier_scale``) additive adjustment, clipped back into
    ``[0, 100]`` afterward. ``dates`` (parallel to ``df``'s rows) is required
    for the confluence modifier -- it's not derivable from ``df`` alone
    (:mod:`firm.strategies.pattern_recognition`'s adjusted OHLCV frame has no
    date column of its own) -- and is silently ignored (logged at debug) when
    ``confluence_modifier_enabled`` is False, so passing it is always safe.
    """
    if len(df) < 20:
        return []
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    atr = atr14(high, low, close)
    current_atr = float(atr[-1]) if len(atr) and not np.isnan(atr[-1]) else None

    threshold_fn = None
    if zigzag_atr_mult is not None and zigzag_atr_mult > 0:
        def threshold_fn(i: int, _atr: np.ndarray = atr, _close: np.ndarray = close, _mult: float = zigzag_atr_mult) -> float | None:
            if i >= len(_atr):
                return None
            a, c = _atr[i], _close[i]
            if a != a or c <= 0:  # NaN guard
                return None
            return _mult * a / c
        log.debug("scan_symbol: using ATR-scaled zigzag threshold (mult=%.2f)", zigzag_atr_mult)

    pivots = zigzag_pivots(high, low, pct=zigzag_pct, threshold_fn=threshold_fn)

    weekly_trend: WeeklyTrend | None = None
    if confluence_modifier_enabled:
        if dates is None:
            log.debug(
                "scan_symbol: confluence_modifier_enabled but no `dates` supplied -- "
                "skipping weekly confluence for this symbol"
            )
        else:
            weekly_df = resample_to_weekly(dates, high, low, close, volume)
            weekly_trend = weekly_trend_direction(weekly_df, lookback_weeks=confluence_lookback_weeks)

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
            scored = _score_and_finalize(
                candidate, close, low, high, volume, atr, current_atr, stop_atr_floor,
                retest_modifier_enabled=retest_modifier_enabled,
                retest_lookback_bars=retest_lookback_bars,
                retest_modifier_scale=retest_modifier_scale,
                weekly_trend=weekly_trend,
                confluence_modifier_scale=confluence_modifier_scale,
            )
            if scored.quality_score >= min_score:
                matches.append(scored)

    matches.sort(key=lambda m: m.quality_score, reverse=True)
    return matches


def _score_and_finalize(
    candidate: PatternMatch,
    close: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    volume: np.ndarray,
    atr_series: np.ndarray,
    current_atr: float | None,
    stop_atr_floor: float,
    *,
    retest_modifier_enabled: bool = False,
    retest_lookback_bars: int = 10,
    retest_modifier_scale: float = 3.0,
    weekly_trend: WeeklyTrend | None = None,
    confluence_modifier_scale: float = 3.0,
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
        # 2026-09: feeds scorer.py's two newest components
        # (breakout_distance/pre_breakout_compression) -- `entry` doubles as
        # `level_at_confirm` since on this dataclass it already *is* the
        # breakout/confirmation level (see scorer.py's module docstring).
        close_at_confirm=close_at_confirm,
        level_at_confirm=candidate.entry,
        atr_series=atr_series,
        confirm_index=candidate.confirm_index,
    )

    quality_score = score.total
    breakdown = score.as_dict()

    # Retest-hold-vs-fail and weekly-confluence modifiers (2026-09): small,
    # bounded additive adjustments, off unless the caller opts in -- see
    # confirmation.py/confluence.py's module docstrings for why both are
    # scoring modifiers rather than hard gates or independent signals.
    if retest_modifier_enabled:
        direction: Literal["above", "below"] = "above" if candidate.direction == "long" else "below"
        outcome = retest_outcome(
            close, low, high, candidate.entry, direction, candidate.confirm_index,
            lookback_bars=retest_lookback_bars,
        )
        modifier = retest_score_modifier(outcome) * retest_modifier_scale
        quality_score += modifier
        breakdown["retest_outcome"] = outcome
        breakdown["retest_modifier"] = modifier
        log.debug(
            "scan_symbol: pattern=%s retest_outcome=%s modifier=%+.2f",
            candidate.pattern, outcome, modifier,
        )

    if weekly_trend is not None:
        modifier = confluence_modifier(candidate.direction, weekly_trend) * confluence_modifier_scale
        quality_score += modifier
        breakdown["weekly_trend"] = weekly_trend
        breakdown["confluence_modifier"] = modifier
        log.debug(
            "scan_symbol: pattern=%s weekly_trend=%s modifier=%+.2f",
            candidate.pattern, weekly_trend, modifier,
        )

    quality_score = float(np.clip(quality_score, 0.0, 100.0))
    breakdown["total"] = quality_score

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
        quality_score=quality_score,
        score_breakdown=breakdown,
    )
