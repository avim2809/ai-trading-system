"""Breakout forecasts (credibility plan P3-03).

Source: plan P3-03 and Carver's breakout rule (*Advanced Futures Trading
Strategies*): the smoothed position of price inside its trailing range.

    mx_t  = rolling_max(price, N)      window includes t, min_periods = N
    mn_t  = rolling_min(price, N)
    raw_t = 40 * (price_t - (mx_t + mn_t) / 2) / (mx_t - mn_t)    in [-20, +20]
    raw   = ewm(raw, span = max(1, N // 4), adjust=False).mean()
    f     = clip(raw * scalar, floor, cap)

NOT related to ``firm.strategies.volatility_breakout`` (an ATR breakout on
single stocks). This module shares no code with it and must not import
``firm.strategies``.

The raw forecast needs no vol estimate (unlike EWMAC); vol enters only at
sizing (P3-06). Windows are trailing, so the forecast at close t is traded at
t+1 by P3-09. A zero range (flat or halted series) gives raw 0, never NaN/inf.
The scalar (mean |f| = 10) is a parameter here; it is estimated on real data
only in P3-11, on the uncapped, unfloored signed raw series. Per-rule forecasts
stay signed (``floor=-FORECAST_CAP``); only the combined forecast is floored
at 0 in P3-05.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Pre-registered lookbacks (source plan P3-03; Carver). Do not extend or select on performance.
BREAKOUT_LOOKBACKS: tuple[int, ...] = (20, 40, 80, 160, 320)
# Carver: 40 * (p - mid) / (max - min) spans [-20, +20].
BREAKOUT_RANGE_SCALE = 40.0
FORECAST_CAP = 20.0


def _span(n: int, smooth_span: int | None) -> int:
    return max(1, n // 4) if smooth_span is None else max(1, int(smooth_span))


def breakout_raw(price: pd.Series, n: int, smooth_span: int | None = None) -> pd.Series:
    """40 * (price - mid_N) / (max_N - min_N), then EWMA smoothed with span N/4 (default).

    ``smooth_span=1`` gives the unsmoothed series. NaN until N observations exist.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    price = price.astype(float)
    mx = price.rolling(n, min_periods=n).max()
    mn = price.rolling(n, min_periods=n).min()
    rng = mx - mn
    raw = BREAKOUT_RANGE_SCALE * (price - (mx + mn) / 2.0) / rng.where(rng > 0)
    flat = (rng == 0) & price.notna()
    if flat.any():
        logger.debug("breakout_raw: %d zero-range bars set to 0 (n=%d)", int(flat.sum()), n)
        raw = raw.mask(flat, 0.0)
    span = _span(n, smooth_span)
    if span > 1:
        raw = raw.ewm(span=span, adjust=False).mean()
    return raw


def breakout_forecast(
    price: pd.Series,
    n: int,
    scalar: float,
    cap: float = FORECAST_CAP,
    floor: float | None = -FORECAST_CAP,
    smooth_span: int | None = None,
) -> pd.Series:
    """raw * scalar, clipped to [floor, cap]. ``floor=None`` leaves the lower side uncapped."""
    f = breakout_raw(price, n, smooth_span) * scalar
    return f.clip(lower=floor, upper=cap)
