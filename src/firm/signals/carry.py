"""Carry forecast (credibility plan P3-04): pure function, fixture-tested.

RESEARCH USE BLOCKED until P2-07 and OD-01/OD-02; second-contract data availability unconfirmed
(OD-02 requirement). Built but unused in core_v1 (the ETF path omits carry). Never run this on a
real series before a futures family is pre-registered.

Source: plan P3-04 and A4; Koijen, Moskowitz, Pedersen and Vrugt (2018), "Carry"; Carver's
term-structure carry.

    raw_t = ((price_near - price_far) / years_between) / sigma_price_annual
    f_t   = clip(EWMA_span(raw_t) * scalar, floor, cap)       spans in CARRY_SMOOTH_SPANS, cap = 20

Sign: ``price_near > price_far`` is backwardation, positive carry for a long.

Preconditions the caller owns (not checkable here):

* ``price_near`` / ``price_far`` are the two contracts the roll rule would be holding at t (no
  look-ahead; P2-07 ``futures_roll`` in ``firm.data``, not created by this ticket). Back-adjusted
  continuous series are NOT a substitute.
* ``sigma_price_annual`` is in PRICE units per YEAR (``price * vol_annual``), so the ratio is a
  dimensionless annualised carry per unit of vol. Do not pass daily-unit sigma. No heuristic unit
  guard is applied.
* ``years_between`` comes from actual expiry dates (``years_between_contracts``).

The scalar is a parameter, estimated only in P3-11 if a futures family is pre-registered. NaN
inputs are skipped by the EWMA, never filled; the output is NaN on rows where raw carry is NaN.
No ETF-proxy carry (yield, roll) is provided or allowed without its own prereg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CARRY_SMOOTH_SPANS: tuple[int, ...] = (5, 20, 60, 120)
FORECAST_CAP = 20.0  # Carver: average |f| = 10, cap 20


def _check_years(years_between: pd.Series | float) -> None:
    arr = np.asarray(years_between, dtype=float)
    if np.isinf(arr).any() or (arr[~np.isnan(arr)] <= 0).any():
        raise ValueError("years_between must be positive and finite")


def raw_carry(
    price_near: pd.Series,
    price_far: pd.Series,
    years_between: pd.Series | float,
    sigma_price_annual: pd.Series,
) -> pd.Series:
    """(price_near - price_far) / years_between, divided by annualised sigma_price. Positive in backwardation."""
    _check_years(years_between)
    annual = (price_near.astype(float) - price_far.astype(float)) / years_between
    return annual / sigma_price_annual.astype(float).where(sigma_price_annual > 0)


def carry_forecast(
    price_near: pd.Series,
    price_far: pd.Series,
    years_between: pd.Series | float,
    sigma_price_annual: pd.Series,
    span: int,
    scalar: float,
    cap: float = FORECAST_CAP,
    floor: float | None = -FORECAST_CAP,
) -> pd.Series:
    """EWMA(span) of raw carry * scalar, clipped to [floor, cap]. ``floor=0.0`` for long/flat diagnostics."""
    if span < 2:
        raise ValueError("span must be >= 2")
    raw = raw_carry(price_near, price_far, years_between, sigma_price_annual)
    smooth = raw.dropna().ewm(span=span, adjust=False).mean().reindex(raw.index)
    return (smooth * scalar).clip(lower=floor, upper=cap)


def years_between_contracts(expiry_near: pd.Series, expiry_far: pd.Series) -> pd.Series:
    """(expiry_far - expiry_near).days / 365.25; raises ValueError if any gap is <= 0."""
    gap = (pd.to_datetime(expiry_far) - pd.to_datetime(expiry_near)).dt.days / 365.25
    _check_years(gap)
    return gap
