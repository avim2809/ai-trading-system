"""Pattern quality scoring.

The core empirical justification for this whole module: raw, unscored chart
patterns win at roughly a coin-flip rate, while quality-filtered patterns
(tight geometry, confirmed breakout volume, a clean trendline/neckline fit)
show materially higher historical win rates. Every :class:`~firm.patterns.
match.PatternMatch` must clear ``min_score`` before it becomes a tradeable
signal — see :mod:`firm.patterns.scanner`.

Score is 0-100, split across independently-computable components so a
pattern can be inspected component-by-component instead of as one opaque
number:

- ``geometry`` (0-30): how tightly the pivots satisfy the pattern's own
  tolerance rule (e.g. shoulder symmetry, peak/trough closeness).
- ``trendline_fit`` (0-15): R^2 of the neckline / trendline / poly fit.
- ``volume_confirmation`` (0-30): breakout-bar volume vs. its trailing
  20-day average — the single most-cited confirmation signal in classical
  TA, and (2026-09-27) the only one of the original components confirmed
  to actually discriminate noise from real patterns on the golden
  benchmark (``scripts/benchmark_pattern_detectors.py`` — real patterns'
  breakout bars show volume expansion; noise mostly doesn't). Absorbed
  ``duration``'s 10 points in the 2026-09-27 rebalance below.
- ``follow_through`` (0-10): genuine POST-confirmation drift — see the
  2026-09-27 fix note below.
- ``breakout_distance`` (0-10): how far the breakout-bar close cleared the
  pattern's own structural level (neckline/rectangle top/etc.), in units of
  the ATR *as of the confirmation bar itself* (contemporaneous volatility).
- ``pre_breakout_compression`` (0-5): whether ATR(14) was unusually tight
  (compressed) relative to its own trailing 20-bar average in the bar just
  before confirmation — compressed-volatility breakouts are documented as
  more credible/less-faded than breakouts occurring when volatility is
  already elevated.
- ``duration`` (reported, 0 weight as of 2026-09-27): see below — kept as a
  diagnostic field (still populated in ``score_breakdown``, still a stable
  ``PatternScore``/feature-vector field) but no longer contributes to
  ``total``.

**2026-09-27 false-positive-rate fix** (``docs/pattern_recognition_plan.md``
has the full investigation): the golden benchmark
(``scripts/benchmark_pattern_detectors.py``) measured a 70% false-positive
rate on pure random-walk noise at the live ``min_score=60`` threshold, with
a Brier score *worse* than an uninformative forecaster. Two of the seven
components were the biggest contributors and are fixed here:

1. **``duration`` never discriminated noise from signal.** A fixed 3%
   zigzag threshold on daily bars produces pivot spacing in the same
   15-90 bar band regardless of whether the underlying shape is real —
   noise scored this component ~10/10 (full marks) essentially always.
   Rather than keep a component with no discriminative power just for
   the sake of a round 7-part score, it's zeroed out of ``total`` and its
   10 points redistributed to ``volume_confirmation`` (20 -> 30), the one
   component that *did* separate noise from signal on the benchmark.
2. **``follow_through`` and ``breakout_distance`` were redundant, not
   independent evidence.** Both had reduced to the *same* underlying
   quantity — ``abs(close_at_confirm - entry)`` (``entry`` doubles as
   ``level_at_confirm``) — divided only by ATR measured at two different
   points in time (``current_atr`` vs. ``atr_at_confirm``), which are
   themselves highly autocorrelated for the short windows this scanner
   uses. On noise, this pair averaged 15.5 of their combined 20 points.
   ``follow_through_atr`` (computed by the caller, see
   ``scanner.py::_score_and_finalize``) now measures genuine
   POST-confirmation drift (``close_now`` vs. ``close_at_confirm``,
   signed by the pattern's own direction) instead of re-measuring the
   same breakout-bar distance ``breakout_distance`` already covers — a
   freshly-confirmed match correctly scores 0 here (no time yet to follow
   through), which is honest, not a bug, and a real, non-redundant signal
   once bars have actually elapsed since confirmation.

Weight total: 30 (geometry) + 15 (trendline_fit) + 30 (volume_confirmation)
+ 10 (follow_through) + 10 (breakout_distance) + 5 (pre_breakout_compression)
+ 0 (duration, diagnostic only) = 100. ``min_score`` thresholds tuned
against the prior scale (e.g. the 60.0 default in
``pattern_recognition.py``) are **not** automatically comparable across
this rebalance -- re-validate via the golden benchmark
(``scripts/benchmark_pattern_detectors.py``) rather than assuming 60 still
means the same thing; this is a real, intended shift in what "60" means,
not a compatibility-preserving reshuffle like the 2026-09 rebalance below.

Prior weight rebalance (2026-09, superseded by the above but kept for
history): originally geometry/trendline_fit/volume_confirmation/
duration/follow_through summed to 35+20+25+10+10=100. Adding
``breakout_distance``/``pre_breakout_compression`` proportionally scaled
the first three down (35->30, 20->15, 25->20) while leaving
duration/follow_through at their original 10 each, giving the two new
components 10 and 5 respectively.

Backward compatibility contract: ``breakout_distance`` and
``pre_breakout_compression`` are driven entirely by optional keyword args
(``close_at_confirm``, ``level_at_confirm``, ``atr_series``,
``confirm_index``). Any existing caller that doesn't pass them keeps
working unchanged and simply gets a **zero** contribution from both (never
a crash) — i.e. those callers are scored out of an effective max of 85 (30
+15+30+10+0+0, since duration no longer counts), not 100, until upgraded
to supply the new inputs.
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
        # duration excluded (2026-09-27): confirmed non-discriminative on
        # the golden benchmark (noise scores ~full marks), see module
        # docstring. Still populated/reported for diagnostics and as a
        # stable feature-vector field, just no longer counted.
        return (
            self.geometry
            + self.trendline_fit
            + self.volume_confirmation
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
    # 30 pts (was 20 pre-2026-09-27): absorbed duration's 10 points -- see
    # module docstring for why (the one component confirmed discriminative
    # on the golden benchmark).
    volume_confirmation = 30.0 * _clip01((volume_ratio - 1.0) / (_VOLUME_TARGET_MULTIPLE - 1.0))
    duration = _duration_score(duration_bars)  # diagnostic only, not in .total -- see docstring
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
