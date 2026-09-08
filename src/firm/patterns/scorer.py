"""Pattern quality scoring.

The core empirical justification for this whole module: raw, unscored chart
patterns win at roughly a coin-flip rate, while quality-filtered patterns
(tight geometry, confirmed breakout volume, a clean trendline/neckline fit)
show materially higher historical win rates. Every :class:`~firm.patterns.
match.PatternMatch` must clear ``min_score`` before it becomes a tradeable
signal — see :mod:`firm.patterns.scanner`.

Score is 0-100, split across four independently-computable components so a
pattern can be inspected component-by-component instead of as one opaque
number:

- ``geometry`` (0-35): how tightly the pivots satisfy the pattern's own
  tolerance rule (e.g. shoulder symmetry, peak/trough closeness).
- ``trendline_fit`` (0-20): R^2 of the neckline / trendline / poly fit.
- ``volume_confirmation`` (0-25): breakout-bar volume vs. its trailing 20-day
  average — the single most-cited confirmation signal in classical TA.
- ``duration`` (0-10): penalises patterns that are implausibly short (noise)
  or implausibly long (stale) for a daily-bar scan.
- ``follow_through`` (0-10): how far the close has already moved beyond the
  breakout level, in ATR units — a close that barely ticks across the line
  is a much weaker signal than one that clears it decisively.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    @property
    def total(self) -> float:
        return (
            self.geometry
            + self.trendline_fit
            + self.volume_confirmation
            + self.duration
            + self.follow_through
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "geometry": self.geometry,
            "trendline_fit": self.trendline_fit,
            "volume_confirmation": self.volume_confirmation,
            "duration": self.duration,
            "follow_through": self.follow_through,
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


def score_pattern(
    *,
    geometry_tolerance_used: float,
    fit_quality: float,
    volume_ratio: float,
    duration_bars: int,
    follow_through_atr: float,
) -> PatternScore:
    """Compute the four-component quality score for one detected pattern."""
    geometry = 35.0 * _clip01(geometry_tolerance_used)
    trendline_fit = 20.0 * _clip01(fit_quality)
    volume_confirmation = 25.0 * _clip01((volume_ratio - 1.0) / (_VOLUME_TARGET_MULTIPLE - 1.0))
    duration = _duration_score(duration_bars)
    follow_through = 10.0 * _clip01(follow_through_atr / _FOLLOW_THROUGH_SATURATION_ATR)
    return PatternScore(
        geometry=geometry,
        trendline_fit=trendline_fit,
        volume_confirmation=volume_confirmation,
        duration=duration,
        follow_through=follow_through,
    )
