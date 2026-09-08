"""Shared trendline fitting for triangle/wedge/flag/rectangle detection."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Trendline(NamedTuple):
    slope: float
    intercept: float
    r2: float

    def value_at(self, x: float) -> float:
        return self.slope * x + self.intercept


def fit_trendline(xs: list[float], ys: list[float]) -> Trendline | None:
    """OLS fit through the given points. ``None`` if fewer than 2 points."""
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return Trendline(float(slope), float(intercept), max(0.0, r2))


def line_through(x1: float, y1: float, x2: float, y2: float) -> Trendline:
    """Exact 2-point line (e.g. a neckline through two troughs) — always a
    perfect fit, so ``r2`` is trivially 1.0; callers score tightness
    separately via geometry tolerance, not this fit quality.
    """
    if x2 == x1:
        return Trendline(0.0, y1, 1.0)
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return Trendline(slope, intercept, 1.0)


def fit_poly2(ys: np.ndarray) -> tuple[float, float, float, float]:
    """Degree-2 OLS fit over ``ys`` indexed 0..len-1. Returns ``(a, b, c, r2)``
    for ``a*x^2 + b*x + c``. Used by Cup & Handle / Rounding Bottom to test
    for a concave-up ("U"-shaped, ``a > 0``) curve.
    """
    n = len(ys)
    x = np.arange(n, dtype=float)
    a, b, c = np.polyfit(x, ys, 2)
    pred = a * x**2 + b * x + c
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return float(a), float(b), float(c), max(0.0, r2)
