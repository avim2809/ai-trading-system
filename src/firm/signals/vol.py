"""Blended EWMA volatility estimator (credibility plan P3-01).

Source: plan P3-01 and Carver, *Systematic Trading* / *Advanced Futures Trading
Strategies*. A fast EWMA std (span 35) is blended 70/30 with the long-run
average of that same vol (2520 days) and floored at the expanding 5th
percentile of its own history, so vol does not collapse in quiet periods and
over-lever the book. The defaults below are the source plan's constants and are
not tuned; their robustness (+/-25%) is tested in G-RESEARCH 6.

    fast_t  = sqrt(EWMA_span(r^2)_t)           zero-mean (Carver), adjust=False
    long_t  = mean(fast over the last L valid obs); expanding mean while fewer than L
    blend_t = (1 - w) * fast_t + w * long_t    w = 0.3
    floor_t = expanding quantile_q(blend_..t)  q = 0.05, data up to and including t
    vol_t   = max(blend_t, floor_t) * sqrt(256)

Corrections to the source plan (each with a reason):

* The floor is an EXPANDING quantile, never a full-sample one (that would use
  future data).
* The vol at the close of day t uses return t. A consumer trading at t+1 must
  lag by one bar; nothing here shifts. P3-09 owns that test.
* 256 trading days per year (Carver), one exported constant; Sharpe reporting
  at 252 stays in ``firm.validation``.
* ``min_obs``, the long window and the floor all count VALID (non-NaN) return
  observations, not index rows. NaN returns (holidays, halts) are skipped, not
  forward-filled or zeroed (zeroing would bias vol down); the output is NaN on
  those rows.
* The EWMA is zero-mean (second moment, not demeaned variance). Fixed choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 256  # Carver convention; sqrt = 16. Imported by P3-02/03/06.

_WARMUPS = ("expanding", "nan")


def _valid(s: pd.Series) -> pd.Series:
    if np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("series contains inf")
    return s.dropna()


def ewma_std(returns: pd.Series, span: int) -> pd.Series:
    """Daily zero-mean EWMA std, ``sqrt(ewm(span, adjust=False)(r^2))``, unannualised.

    NaN returns are skipped (estimate computed over valid observations only); the
    output is NaN on NaN rows.
    """
    if span < 2:
        raise ValueError("span must be >= 2")
    r = _valid(returns.astype(float))
    out = np.sqrt(r.pow(2).ewm(span=span, adjust=False).mean())
    return out.reindex(returns.index)


def long_run_mean(vol: pd.Series, window: int, warmup: str) -> pd.Series:
    """Mean of ``vol`` over the last ``window`` valid obs.

    ``warmup="expanding"``: expanding mean until ``window`` obs exist (pre-registered
    fallback); ``"nan"``: NaN until then.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if warmup not in _WARMUPS:
        raise ValueError(f"warmup must be one of {_WARMUPS}")
    v = _valid(vol.astype(float))
    out = v.rolling(window, min_periods=1 if warmup == "expanding" else window).mean()
    return out.reindex(vol.index)


def expanding_percentile_floor(vol: pd.Series, q: float, min_obs: int) -> pd.Series:
    """Expanding q-quantile of ``vol`` using data up to and including t; NaN before ``min_obs`` valid obs."""
    if not 0.0 <= q < 1.0:
        raise ValueError("q must be in [0, 1)")
    if min_obs < 1:
        raise ValueError("min_obs must be >= 1")
    v = _valid(vol.astype(float))
    out = v.expanding(min_periods=min_obs).quantile(q)
    return out.reindex(vol.index)


def ewma_vol(
    returns: pd.Series,
    span: int = 35,
    blend_long_weight: float = 0.3,
    long_window_days: int = 2520,
    floor_percentile: float = 0.05,
    min_obs: int = 256,
    warmup: str = "expanding",
    annualise: bool = True,
) -> pd.Series:
    """Blended, floored EWMA vol of simple daily ``returns`` (fraction, 0.16 = 16%).

    NaN until ``min_obs`` valid returns exist (and, with ``warmup="nan"``, until
    ``long_window_days``). Known at the close of day t; consumers lag it themselves.
    ``floor_percentile=0`` disables the floor in effect (expanding minimum).
    """
    if not 0.0 <= blend_long_weight <= 1.0:
        raise ValueError("blend_long_weight must be in [0, 1]")
    if min_obs < 1:
        raise ValueError("min_obs must be >= 1")
    if warmup not in _WARMUPS:
        raise ValueError(f"warmup must be one of {_WARMUPS}")
    fast = ewma_std(returns, span)
    long_ = long_run_mean(fast, long_window_days, warmup)
    blend = (1.0 - blend_long_weight) * fast + blend_long_weight * long_
    # The floor itself is defined from the first valid blend; ``min_obs`` gates the output
    # below (so that warmup="nan", whose blend starts late, is not delayed a second time).
    floor = expanding_percentile_floor(blend, floor_percentile, 1)
    vol = np.maximum(blend, floor)
    count = returns.notna().cumsum()
    vol = vol.where(count >= min_obs)
    if annualise:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def price_unit_vol(price: pd.Series, vol_annual: pd.Series) -> pd.Series:
    """Daily vol in price units: ``price * vol_annual / sqrt(256)`` (EWMAC and sizing)."""
    return price * vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
