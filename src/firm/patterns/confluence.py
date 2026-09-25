"""Weekly-timeframe confluence: a bounded quality-score modifier, never a
second trading signal.

Multi-timeframe confirmation (checking that a higher-timeframe's structure
agrees with a daily-bar pattern's implied direction) is a well-supported
false-positive filter in practice. But it is trivially easy to get wrong: if
you ever let a higher-timeframe read depend on the *current, still-forming*
higher-timeframe bar, you leak a small amount of "the future" (today's own
close) into a feature that's supposed to represent independent, already-
known context — a 2025 multi-timeframe study found ~0.20 ROC-AUC inflation
from exactly this kind of leak. :func:`resample_to_weekly` exists
specifically to make that mistake structurally hard: it drops the trailing
week unconditionally (see its docstring), so nothing downstream in this
module can ever see a partially-formed week.

Deliberately three small, single-purpose functions with no cross-talk:

1. :func:`resample_to_weekly` — pure daily -> completed-weekly OHLCV
   resampling.
2. :func:`weekly_trend_direction` — one coarse trend read off the weekly
   closes (deliberately not over-parameterized: a single lookback, no extra
   tunable indicators, to avoid reintroducing the overfitting risk this
   whole feature is meant to guard against).
3. :func:`confluence_modifier` — maps (pattern direction, weekly trend) to a
   small, bounded additive adjustment. Like
   :mod:`firm.patterns.confirmation`'s retest modifier, this is a clean
   building block only — the integration owner (scanner.py /
   pattern_recognition.py) decides exactly how/whether to fold it into an
   overall quality_score. This module never emits an independent trading
   signal, by design: doing so would double the effective bet count on the
   same underlying move without having been separately validated.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WeeklyTrend = Literal["up", "down", "flat"]
PatternDirection = Literal["long", "short"]

# weekly_trend_direction: minimum |pct change| over `lookback_weeks` to call
# it "up"/"down" rather than "flat". A single module constant rather than a
# tunable parameter, deliberately -- see module docstring on avoiding
# over-parameterization here.
_FLAT_THRESHOLD_PCT = 0.01

# confluence_modifier: which weekly trend "agrees" / "contradicts" each
# pattern direction. Anything else (i.e. "flat" on either side) is neutral.
_AGREEING_TREND: dict[str, str] = {"long": "up", "short": "down"}
_CONTRADICTING_TREND: dict[str, str] = {"long": "down", "short": "up"}
_AGREE_MODIFIER = 1.0
_CONTRADICT_MODIFIER = -1.0
_NEUTRAL_MODIFIER = 0.0


def resample_to_weekly(
    dates: pd.Series | np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> pd.DataFrame:
    """Aggregate daily OHLCV bars into calendar-week (Monday-Sunday) OHLCV
    bars, indexed by each week's Sunday label (pandas ``"W-SUN"`` resample
    convention) — and, CRITICALLY, only ever returns fully-completed weeks.

    Look-ahead guard (the single most important correctness property of
    this function): the trailing resampled row is always dropped
    unconditionally. Why unconditional rather than conditional on some
    notion of "today": this function only receives ``dates`` (no separate
    wall-clock/as-of parameter), and given equities never trade on a
    Sunday (the ``"W-SUN"`` week-ending label), the week containing the
    single most recent date in ``dates`` can *never* be proven complete
    from the daily bars alone (Thu/Fri may simply not exist yet, whether
    because they're in the future relative to a point-in-time backtest's
    as-of date, or because today's live bar hasn't closed yet). Dropping it
    unconditionally is deliberately conservative: it may occasionally
    discard a week that happened to be genuinely complete (e.g. data ending
    exactly on a Friday), trading a small amount of extra staleness for
    never risking a look-ahead leak. Detecting "was Friday actually the
    last trading day of this week" would require a market-holiday calendar,
    which is out of scope and its own source of bugs.

    Parameters accept dates explicitly (rather than requiring a ``"date"``
    column already attached to a high/low/close/volume frame) so this
    function stays decoupled from any one caller's exact DataFrame shape —
    e.g. ``firm.strategies.pattern_recognition._adjusted_ohlc`` returns
    high/low/close/volume with no ``"date"`` column of its own, but the
    caller has ``sym_df["date"]`` available separately.

    Returns an empty DataFrame (columns ``high``, ``low``, ``close``,
    ``volume``, indexed by week-ending Sunday) if there isn't at least one
    fully-completed week in the input.
    """
    n = len(close)
    if n == 0:
        return pd.DataFrame(columns=["high", "low", "close", "volume"])

    daily = pd.DataFrame(
        {
            "high": np.asarray(high, dtype=float),
            "low": np.asarray(low, dtype=float),
            "close": np.asarray(close, dtype=float),
            "volume": np.asarray(volume, dtype=float),
        },
        index=pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).reset_index(drop=True))),
    ).sort_index()

    weekly = daily.resample("W-SUN").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    weekly = weekly.dropna(how="any")
    if weekly.empty:
        log.debug("resample_to_weekly: no complete daily bars to resample")
        return weekly

    # Drop the trailing (potentially still-forming) week -- see the
    # look-ahead guard explained above. This is unconditional, so it always
    # removes exactly one row.
    completed = weekly.iloc[:-1]
    log.debug(
        "resample_to_weekly: %d daily bars -> %d completed weekly bars (dropped 1 still-forming week)",
        n, len(completed),
    )
    return completed


def weekly_trend_direction(
    weekly_df: pd.DataFrame,
    *,
    lookback_weeks: int = 8,
) -> WeeklyTrend:
    """Coarse weekly trend read: compares the latest completed week's close
    to the close ``lookback_weeks`` weeks before it (a simple momentum read,
    chosen over e.g. a linear-regression slope of the trailing closes to
    avoid adding parameters/indicators here beyond the single lookback --
    see module docstring).

    ``"up"``/``"down"`` require the |pct change| over the lookback window to
    clear :data:`_FLAT_THRESHOLD_PCT`; anything smaller (including exactly
    flat, or when ``weekly_df`` doesn't have ``lookback_weeks + 1`` rows to
    compare) is ``"flat"`` -- the safe, neutral default when a genuine
    directional read isn't warranted.
    """
    if weekly_df.empty or len(weekly_df) <= lookback_weeks:
        log.debug(
            "weekly_trend_direction: only %d completed weekly bar(s), need > lookback_weeks=%d, returning flat",
            len(weekly_df), lookback_weeks,
        )
        return "flat"

    closes = weekly_df["close"].to_numpy(dtype=float)
    latest = closes[-1]
    past = closes[-1 - lookback_weeks]
    if past != past or latest != latest or past == 0:  # NaN guard
        return "flat"

    pct_change = (latest - past) / abs(past)
    if pct_change >= _FLAT_THRESHOLD_PCT:
        return "up"
    if pct_change <= -_FLAT_THRESHOLD_PCT:
        return "down"
    return "flat"


def confluence_modifier(
    pattern_direction: PatternDirection,
    weekly_trend: WeeklyTrend,
) -> float:
    """Small, bounded quality-score modifier from comparing a daily-bar
    pattern's implied direction against the weekly trend: ``+1.0`` when they
    agree (``long``+``up`` or ``short``+``down``), ``-1.0`` when they
    contradict (``long``+``down`` or ``short``+``up``), ``0.0`` when the
    weekly trend is ``"flat"`` (neutral -- no read either way).

    A pure building block: how (or whether) this gets combined with
    scorer.py's ``quality_score`` or confirmation.py's retest modifier is
    left to the integration owner. Never raises -- an unrecognized
    ``pattern_direction``/``weekly_trend`` combination logs a warning and
    returns the neutral ``0.0``.
    """
    if weekly_trend == "flat":
        return _NEUTRAL_MODIFIER
    if _AGREEING_TREND.get(pattern_direction) == weekly_trend:
        return _AGREE_MODIFIER
    if _CONTRADICTING_TREND.get(pattern_direction) == weekly_trend:
        return _CONTRADICT_MODIFIER
    log.warning(
        "confluence_modifier: unrecognized pattern_direction=%r / weekly_trend=%r, defaulting to neutral 0.0",
        pattern_direction, weekly_trend,
    )
    return _NEUTRAL_MODIFIER
