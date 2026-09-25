"""Pattern quality scoring.

The core empirical justification for this whole module: raw, unscored chart
patterns win at roughly a coin-flip rate, while quality-filtered patterns
(tight geometry, confirmed breakout volume, a clean trendline/neckline fit)
show materially higher historical win rates. Every :class:`~firm.patterns.
match.PatternMatch` must clear ``min_score`` before it becomes a tradeable
signal — see :mod:`firm.patterns.scanner`.

Score is 0-100, split across seven independently-computable components so a
pattern can be inspected component-by-component instead of as one opaque
number:

- ``geometry`` (0-30): how tightly the pivots satisfy the pattern's own
  tolerance rule (e.g. shoulder symmetry, peak/trough closeness).
- ``trendline_fit`` (0-15): R^2 of the neckline / trendline / poly fit.
- ``volume_confirmation`` (0-20): breakout-bar volume vs. its trailing 20-day
  average — the single most-cited confirmation signal in classical TA.
- ``duration`` (0-10): penalises patterns that are implausibly short (noise)
  or implausibly long (stale) for a daily-bar scan.
- ``follow_through`` (0-10): how far the close has already moved beyond the
  breakout level *as of now*, in units of the instrument's *current* (most
  recent bar) ATR — a close that barely ticks across the line is a much
  weaker signal than one that clears it decisively, and this also captures
  continued drift after confirmation.
- ``breakout_distance`` (0-10): how far the breakout-bar close cleared the
  pattern's own structural level (neckline/rectangle top/etc.), in units of
  the ATR *as of the confirmation bar itself* (contemporaneous volatility,
  not "now"). See "``follow_through`` vs. ``breakout_distance``" below for
  why both exist.
- ``pre_breakout_compression`` (0-5): whether ATR(14) was unusually tight
  (compressed) relative to its own trailing 20-bar average in the bar just
  before confirmation — compressed-volatility breakouts are documented as
  more credible/less-faded than breakouts occurring when volatility is
  already elevated.

``follow_through`` vs. ``breakout_distance`` — deliberately NOT the same
signal, so neither was renamed/removed when the second was added:

- ``follow_through_atr`` (existing, required) = ``abs(close_at_confirm -
  entry) / atr_now``, where ``atr_now`` is the ATR of the *most recent* bar
  in the scanned window. This mixes two things: how decisively price broke
  out, *and* how much the volatility regime may have since drifted from what
  it was at breakout time (the scan can run well after ``confirm_index``).
- ``breakout_distance`` (new, optional) = ``abs(close_at_confirm -
  level_at_confirm) / atr_at_confirm``, where ``atr_at_confirm`` is the ATR
  *at the confirmation bar itself*. This isolates "was the breakout close
  decisive relative to the volatility regime it actually happened in,"
  independent of anything that happened afterwards. ``level_at_confirm`` is
  the structural level (neckline/rectangle edge/etc.) rather than ``entry``
  when the two differ conceptually, though in this codebase ``entry`` on
  :class:`~firm.patterns.match.PatternMatch` *is* the breakout/confirmation
  level, so callers may pass the same value for both.

Weight rebalance (2026-09): originally geometry/trendline_fit/
volume_confirmation/duration/follow_through summed to 35+20+25+10+10=100.
Adding two new components proportionally scaled the first three down
(35->30, 20->15, 25->20) while leaving duration/follow_through at their
original 10 each, and gave the new components 10 (``breakout_distance``,
comparable weight to ``follow_through`` since it's the same *kind* of
signal, just measured at a different point in time) and 5
(``pre_breakout_compression``, a smaller, purely corroborating signal) —
30+15+20+10+10+10+5 = 100, so ``min_score`` thresholds tuned against the old
5-component scale (e.g. the 60.0 default in ``pattern_recognition.py``)
remain roughly comparable in meaning.

Backward compatibility contract: ``breakout_distance`` and
``pre_breakout_compression`` are driven entirely by new *optional* keyword
args (``close_at_confirm``, ``level_at_confirm``, ``atr_series``,
``confirm_index``). Any existing caller that doesn't pass them keeps working
unchanged and simply gets a **zero** contribution from both new components
(never a crash) — i.e. those callers are now scored out of an effective
max of 85, not 100, until they're upgraded to supply the new inputs. This is
the explicit, documented trade-off of rebalancing the weights (option (a))
over leaving old weights untouched and only adding bonus points on top
(option (b)): the *shape* of the score stays a clean 0-100, but a caller
that doesn't supply the new context is, by construction, no longer able to
reach 100.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Breakout volume at or above this multiple of the 20d average earns full
# volume_confirmation credit (matches the plan's empirically-cited threshold).
_VOLUME_TARGET_MULTIPLE = 1.5
# Duration band (bars) treated as "textbook" for a daily-bar formation;
# outside of it the score decays linearly to 0 at the floor/ceiling below.
_DURATION_IDEAL_MIN = 15
_DURATION_IDEAL_MAX = 90
_DURATION_FLOOR = 5
_DURATION_CEIL = 180
# Follow-through saturates (full credit) at this many ATRs past the level.
_FOLLOW_THROUGH_SATURATION_ATR = 2.0
# breakout_distance saturates (full credit) at this many ATRs-at-confirmation
# past the structural level. Deliberately lower than the follow-through
# saturation above: 1.0x ATR clearing the level *at breakout time* is a
# well-documented "real breakout" bar, whereas follow_through's 2.0x is
# measuring cumulative drift over however long has elapsed since.
_BREAKOUT_DISTANCE_SATURATION_ATR = 1.0
# pre_breakout_compression: the rolling window (bars) ATR(14) is compared
# against, and the ratio bounds between which credit is linearly
# interpolated. ratio <= _COMPRESSION_FULL_CREDIT_RATIO -> full credit
# (volatility is meaningfully compressed vs. its own recent average);
# ratio >= _COMPRESSION_ZERO_CREDIT_RATIO -> zero credit (volatility already
# elevated going into the breakout, a less-documented/less-credible setup).
_COMPRESSION_ROLLING_WINDOW = 20
_COMPRESSION_FULL_CREDIT_RATIO = 0.8
_COMPRESSION_ZERO_CREDIT_RATIO = 1.2


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class PatternScore:
    geometry: float
    trendline_fit: float
    volume_confirmation: float
    duration: float
    follow_through: float
    # New (2026-09), backward-compatible: default to 0.0 (neutral/zero
    # contribution) so any caller/test constructing a PatternScore directly
    # without these still gets a valid, sensible instance.
    breakout_distance: float = 0.0
    pre_breakout_compression: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.geometry
            + self.trendline_fit
            + self.volume_confirmation
            + self.duration
            + self.follow_through
            + self.breakout_distance
            + self.pre_breakout_compression
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "geometry": self.geometry,
            "trendline_fit": self.trendline_fit,
            "volume_confirmation": self.volume_confirmation,
            "duration": self.duration,
            "follow_through": self.follow_through,
            "breakout_distance": self.breakout_distance,
            "pre_breakout_compression": self.pre_breakout_compression,
            "total": self.total,
        }


def _duration_score(duration_bars: int) -> float:
    if _DURATION_IDEAL_MIN <= duration_bars <= _DURATION_IDEAL_MAX:
        return 10.0
    if duration_bars < _DURATION_IDEAL_MIN:
        span = _DURATION_IDEAL_MIN - _DURATION_FLOOR
        return 10.0 * _clip01((duration_bars - _DURATION_FLOOR) / span) if span > 0 else 0.0
    span = _DURATION_CEIL - _DURATION_IDEAL_MAX
    return 10.0 * _clip01((_DURATION_CEIL - duration_bars) / span) if span > 0 else 0.0


def _breakout_distance_score(
    close_at_confirm: float | None,
    level_at_confirm: float | None,
    atr_series: np.ndarray | None,
    confirm_index: int | None,
) -> float:
    """0-10: ``abs(close_at_confirm - level_at_confirm) / atr[confirm_index]``,
    saturating at :data:`_BREAKOUT_DISTANCE_SATURATION_ATR`. Returns 0.0 (no
    credit, not an error) whenever any input is missing or the ATR at
    ``confirm_index`` isn't usable (NaN/non-positive/out of range) — this is
    the documented "neutral default" degradation path for callers that don't
    supply the new optional context.
    """
    if close_at_confirm is None or level_at_confirm is None or atr_series is None or confirm_index is None:
        return 0.0
    if confirm_index < 0 or confirm_index >= len(atr_series):
        log.debug(
            "breakout_distance: confirm_index=%s out of range for atr_series len=%d, skipping",
            confirm_index, len(atr_series),
        )
        return 0.0
    atr_at_confirm = float(atr_series[confirm_index])
    if atr_at_confirm != atr_at_confirm or atr_at_confirm <= 0:  # NaN or non-positive
        log.debug("breakout_distance: unusable ATR at confirm_index=%d (%s), skipping", confirm_index, atr_at_confirm)
        return 0.0
    distance_atr = abs(close_at_confirm - level_at_confirm) / atr_at_confirm
    return 10.0 * _clip01(distance_atr / _BREAKOUT_DISTANCE_SATURATION_ATR)


def _pre_breakout_compression_score(
    atr_series: np.ndarray | None,
    confirm_index: int | None,
) -> float:
    """0-5: credit for ATR(14) being compressed vs. its own trailing
    :data:`_COMPRESSION_ROLLING_WINDOW`-bar average in the bar just before
    confirmation (``atr[confirm_index - 1] / rolling_mean(atr, 20)[confirm_index
    - 1]``). Returns 0.0 whenever there isn't a full rolling window of clean
    (non-NaN) history available before ``confirm_index`` — again, a neutral
    default rather than an error, since ATR's own warm-up NaNs near the start
    of a series are routine, not exceptional.
    """
    if atr_series is None or confirm_index is None:
        return 0.0
    idx = confirm_index - 1
    if idx < 0 or idx >= len(atr_series):
        return 0.0
    window_start = idx - _COMPRESSION_ROLLING_WINDOW + 1
    if window_start < 0:
        log.debug(
            "pre_breakout_compression: not enough history before confirm_index=%d for a full %d-bar window, skipping",
            confirm_index, _COMPRESSION_ROLLING_WINDOW,
        )
        return 0.0
    window = atr_series[window_start : idx + 1]
    if np.isnan(window).any():
        log.debug("pre_breakout_compression: NaN in ATR window before confirm_index=%d, skipping", confirm_index)
        return 0.0
    rolling_mean = float(window.mean())
    atr_at = float(atr_series[idx])
    if rolling_mean <= 0 or atr_at != atr_at:
        return 0.0
    ratio = atr_at / rolling_mean
    if ratio <= _COMPRESSION_FULL_CREDIT_RATIO:
        frac = 1.0
    elif ratio >= _COMPRESSION_ZERO_CREDIT_RATIO:
        frac = 0.0
    else:
        frac = (_COMPRESSION_ZERO_CREDIT_RATIO - ratio) / (
            _COMPRESSION_ZERO_CREDIT_RATIO - _COMPRESSION_FULL_CREDIT_RATIO
        )
    return 5.0 * _clip01(frac)


def score_pattern(
    *,
    geometry_tolerance_used: float,
    fit_quality: float,
    volume_ratio: float,
    duration_bars: int,
    follow_through_atr: float,
    close_at_confirm: float | None = None,
    level_at_confirm: float | None = None,
    atr_series: np.ndarray | None = None,
    confirm_index: int | None = None,
) -> PatternScore:
    """Compute the seven-component quality score for one detected pattern.

    ``close_at_confirm``, ``level_at_confirm``, ``atr_series`` and
    ``confirm_index`` are all optional and only used for the two newest
    components (``breakout_distance``, ``pre_breakout_compression`` — see
    the module docstring). Omit all four (as every existing caller of this
    function currently does) and both components simply contribute 0.0 —
    no exception is raised. ``atr_series`` must be the *full* ATR(14) array
    (e.g. from :func:`firm.patterns._indicators.atr14`) for the scanned
    window, not a single current-bar scalar, since ``pre_breakout_compression``
    needs the trailing rolling history around ``confirm_index``, not "now".
    """
    geometry = 30.0 * _clip01(geometry_tolerance_used)
    trendline_fit = 15.0 * _clip01(fit_quality)
    volume_confirmation = 20.0 * _clip01((volume_ratio - 1.0) / (_VOLUME_TARGET_MULTIPLE - 1.0))
    duration = _duration_score(duration_bars)
    follow_through = 10.0 * _clip01(follow_through_atr / _FOLLOW_THROUGH_SATURATION_ATR)
    breakout_distance = _breakout_distance_score(close_at_confirm, level_at_confirm, atr_series, confirm_index)
    pre_breakout_compression = _pre_breakout_compression_score(atr_series, confirm_index)
    return PatternScore(
        geometry=geometry,
        trendline_fit=trendline_fit,
        volume_confirmation=volume_confirmation,
        duration=duration,
        follow_through=follow_through,
        breakout_distance=breakout_distance,
        pre_breakout_compression=pre_breakout_compression,
    )
