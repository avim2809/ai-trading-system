"""Position sizing, buffering and IDM (credibility plan P3-06).

Source: plan P3-06 and Carver's position-sizing and buffering rules (*Systematic Trading*,
*Advanced Futures Trading Strategies*). Formulas, kept from the source:

    N = forecast * capital * IDM * weight * tau / (10 * multiplier * price * fx * sigma%)
    B = 0.1 * capital * IDM * weight * tau / (multiplier * price * fx * sigma%)     (= N at forecast 10, x fraction)
    trade only if the current position is outside [N - B, N + B], trading to the nearest edge
    IDM = min(1 / sqrt(w' H w), 2.5)       H = sub-system return correlation, off-diagonal floored at 0

``sigma%`` is the annualised vol as a fraction (P3-01), ``tau`` a fraction (0.10), ``fx`` converts the
instrument currency to the account currency. tau is NOT chosen here: it comes from the ETF charter
(OD-17, set ex ante, no performance input) and has no default. IDM, weights and H are inputs; they are
estimated on real data only in P3-11 / P4-01.

The COMBINED forecast arriving here is floored at 0 by P3-05 (per-rule forecasts stay signed upstream),
so N >= 0 on the ETF path; ``buffered_trade`` clamps the lower edge at 0 (``long_only``). Consequence of the
source rule, not changed: at forecast 0 the position is cut to the upper edge B, not to 0.

Rounding happens AFTER buffering, on the trade target, so rounding cannot defeat the buffer. ETFs round
toward zero exactly like ``Allocator.plan`` (``_qty_toward_zero``: floor of |q| to 6 decimals when
fractional, ``floor(|q| + 1e-9)`` for whole shares) so shadow replay matches live order sizing; the
convention is mirrored here, not imported. ``nearest`` is for futures (multiplier > 1) only.

``gross_cap_scale`` is the pure pro-rata gross cap helper and bound flag (P3-09 records the share of
days True; P3-08 compares it to the 20% ceiling of OD-17). P4-03 owns the policy.

Research-only: not imported by ``firm.portfolio.__init__`` or any live module (P0-06 isolation test).
"""

from __future__ import annotations

import math

import numpy as np

from firm.portfolio.forecast_combine import fdm as _diversification_multiplier

IDM_CAP = 2.5  # Carver
DEFAULT_BUFFER_FRACTION = 0.10  # Carver: 10% of the average (forecast 10) position
_FRACTIONAL_DECIMALS = 6  # mirrors allocation.allocator._FRACTIONAL_DECIMALS
_EPS = 1e-9  # mirrors allocation.allocator._EPS
_MODES = ("none", "toward_zero", "nearest")


def _check(capital, idm_value, weight, tau, multiplier, price, fx, vol_pct) -> None:
    for name, v in (("price", price), ("vol_pct", vol_pct), ("multiplier", multiplier), ("fx", fx)):
        if not v > 0 or not math.isfinite(v):
            raise ValueError(f"{name} must be positive and finite, got {v!r}")
    for name, v in (("capital", capital), ("idm", idm_value), ("weight", weight), ("tau", tau)):
        if not v >= 0 or not math.isfinite(v):
            raise ValueError(f"{name} must be >= 0 and finite, got {v!r}")


def full_position(
    capital: float, idm: float, weight: float, tau: float, multiplier: float, price: float, fx: float, vol_pct: float
) -> float:
    """Position at forecast 10: capital*idm*weight*tau / (multiplier*price*fx*vol_pct)."""
    _check(capital, idm, weight, tau, multiplier, price, fx, vol_pct)
    return capital * idm * weight * tau / (multiplier * price * fx * vol_pct)


def target_position(
    forecast: float, capital: float, idm: float, weight: float, tau: float,
    multiplier: float, price: float, fx: float, vol_pct: float,
) -> float:
    """forecast / 10 * full_position (source N)."""
    return forecast / 10.0 * full_position(capital, idm, weight, tau, multiplier, price, fx, vol_pct)


def buffer_width(
    capital: float, idm: float, weight: float, tau: float, multiplier: float, price: float, fx: float,
    vol_pct: float, fraction: float = DEFAULT_BUFFER_FRACTION,
) -> float:
    """Source B: ``fraction`` times the full (forecast 10) position."""
    if not fraction >= 0:
        raise ValueError("fraction must be >= 0")
    return fraction * full_position(capital, idm, weight, tau, multiplier, price, fx, vol_pct)


def buffered_trade(current: float, target: float, buffer: float, long_only: bool = True) -> float:
    """New position: unchanged inside [target-B, target+B], else moved to the nearest edge.

    ``long_only`` (ETF path) clamps the lower edge at 0.
    """
    if not buffer >= 0:
        raise ValueError("buffer must be >= 0")
    lower, upper = target - buffer, target + buffer
    if long_only:
        lower = max(lower, 0.0)
        upper = max(upper, 0.0)
    if current < lower:
        return lower
    if current > upper:
        return upper
    return current


def round_position(position: float, fractional: bool, multiplier: float = 1.0, mode: str = "toward_zero") -> float:
    """Round a (buffered) position. Apply AFTER ``buffered_trade``.

    ``none``: unchanged. ``toward_zero``: floor of |q| (6 decimals if ``fractional`` else whole units), sign kept,
    as ``Allocator._qty_toward_zero``. ``nearest``: integer contracts, futures only (multiplier > 1).
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if fractional and multiplier > 1.0:
        raise ValueError("fractional sizing is for ETFs (multiplier 1) only")
    if mode == "none":
        return float(position)
    sign = -1.0 if position < 0 else 1.0
    if mode == "nearest":
        if multiplier <= 1.0:
            raise ValueError("nearest rounding is for futures (multiplier > 1) only")
        return float(round(position))
    if fractional:
        scale = 10**_FRACTIONAL_DECIMALS
        return sign * math.floor(abs(position) * scale) / scale + 0.0
    return sign * float(math.floor(abs(position) + _EPS)) + 0.0


def idm(weights: np.ndarray, H: np.ndarray, cap: float = IDM_CAP) -> float:
    """1 / sqrt(w' H w), H floored at 0 off-diagonal (same convention as the FDM), capped at ``cap``."""
    return _diversification_multiplier(weights, H, cap=cap, floor_rho_at_zero=True)


def gross_cap_scale(weights_by_instrument: dict[str, float], cap: float = 1.0) -> tuple[dict[str, float], bool]:
    """Pro-rata scale so sum(|w|) <= cap. Second value is the gross-cap-bound flag."""
    if not cap > 0:
        raise ValueError("cap must be positive")
    gross = sum(abs(w) for w in weights_by_instrument.values())
    if gross <= cap + _EPS:
        return dict(weights_by_instrument), False
    k = cap / gross
    return {s: w * k for s, w in weights_by_instrument.items()}, True
