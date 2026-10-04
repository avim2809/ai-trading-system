"""Forecast combination and FDM (credibility plan P3-05).

Source: plan P3-05 and Carver. Several partially correlated forecasts for one instrument are
combined with FIXED handcrafted weights (never fitted on returns or Sharpe) and a forecast
diversification multiplier restoring the average |forecast| to about 10:

    combined = clip(sum_r w_r * f_r * FDM, floor, cap)      cap = 20
    FDM      = min(1 / sqrt(w' rho w), 2.5)                 rho = pooled forecast correlation,
                                                            off-diagonals floored at 0

NOT the same object as ``firm.agents.research._combine`` / ``firm.agents.analysts.combine_signals_hrp``,
which combine z-scored pipeline strategy scores (live-imported) and are neither reused nor edited.
This module is research-only: it must not be imported from ``firm.portfolio.__init__``, ``firm.live``,
``firm.api`` or ``firm.runtime`` (listed in the P0-06 isolation test).

Long/flat is applied HERE, once, on the combined forecast: ``combine_forecasts(..., floor=0.0)`` with
signed per-rule inputs (P3-02/03 keep floor=-20). Flooring each rule first would give rules at -10
and +10 a long position, combine-then-floor gives 0. Consequence: the average |combined| forecast of
a long/flat book is about 5, not 10, so realised vol sits well below tau.

Weights have no defaults (the charter fixes them, P5-02). A rule that is NaN on a day makes the combined
forecast NaN that day: there is no silent renormalisation, which would change the FDM implicitly.
Dropped speeds are handled by ``group_equal_weights`` explicitly. The FDM and rho are estimated on real
data only in P3-11; here they are inputs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FDM_CAP = 2.5  # Carver
FORECAST_CAP = 20.0


def validate_weights(weights: dict[str, float], tol: float = 1e-9) -> None:
    """Raise unless every weight is finite and >= 0 and they sum to 1 within ``tol``."""
    if not weights:
        raise ValueError("weights is empty")
    vals = np.array(list(weights.values()), dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any():
        raise ValueError("weights must be finite and >= 0")
    if abs(vals.sum() - 1.0) > tol:
        raise ValueError(f"weights must sum to 1, got {vals.sum()!r}")


def pooled_forecast_correlation(
    forecasts_by_instrument: dict[str, pd.DataFrame],
    start=None,
    end=None,
    min_overlap: int = 250,
) -> pd.DataFrame:
    """Average of per-instrument correlation matrices between rule columns (pairwise-complete).

    Only rows in [start, end] are used. Raises if any rule pair has fewer than ``min_overlap``
    jointly observed rows for any instrument.
    """
    if not forecasts_by_instrument:
        raise ValueError("no instruments")
    mats = []
    cols = None
    for name, df in forecasts_by_instrument.items():
        d = df.loc[start:end] if (start is not None or end is not None) else df
        if cols is None:
            cols = list(d.columns)
        elif list(d.columns) != cols:
            raise ValueError(f"rule columns of {name} differ from the first instrument")
        obs = d.notna().astype(int)
        overlap = obs.T.dot(obs)
        if (overlap.to_numpy() < min_overlap).any():
            raise ValueError(f"{name}: pairwise overlap below {min_overlap}")
        mats.append(d.corr(min_periods=min_overlap).to_numpy())
    avg = np.mean(mats, axis=0)
    return pd.DataFrame(avg, index=cols, columns=cols)


def fdm(
    weights: np.ndarray,
    rho: np.ndarray,
    cap: float = FDM_CAP,
    floor_rho_at_zero: bool = True,
) -> float:
    """1 / sqrt(w' rho w), off-diagonal rho floored at 0 (diagonal stays 1), result capped at ``cap``.

    Zero-weight rules are dropped first. ``w' rho w >= sum w_i^2 > 0`` after the floor, so no division by zero.
    """
    w = np.asarray(weights, dtype=float)
    r = np.array(rho, dtype=float)
    if r.shape != (w.size, w.size):
        raise ValueError("rho shape does not match weights")
    keep = w != 0.0
    w, r = w[keep], r[np.ix_(keep, keep)]
    if w.size == 0:
        raise ValueError("all weights are zero")
    if floor_rho_at_zero:
        r = np.maximum(r, 0.0)
    np.fill_diagonal(r, 1.0)
    q = float(w @ r @ w)
    if not q > 0:
        raise ValueError("w' rho w is not positive")
    return float(min(1.0 / np.sqrt(q), cap))


def group_equal_weights(
    group_weights: dict[str, float],
    surviving_by_group: dict[str, list[str]],
) -> dict[str, float]:
    """Split each group weight equally across its surviving rules; raises if a weighted group has none."""
    validate_weights(group_weights)
    out: dict[str, float] = {}
    for g, gw in group_weights.items():
        if gw == 0.0:
            continue
        rules = surviving_by_group.get(g)
        if not rules:
            raise ValueError(f"group {g!r} has no surviving rule")
        for r in rules:
            if r in out:
                raise ValueError(f"rule {r!r} appears in more than one group")
            out[r] = gw / len(rules)
    return out


def combine_forecasts(
    forecasts: pd.DataFrame,
    weights: dict[str, float],
    fdm_value: float,
    cap: float = FORECAST_CAP,
    floor: float | None = -FORECAST_CAP,
) -> pd.Series:
    """sum_r w_r f_r * FDM, clipped to [floor, cap]. The ETF path passes ``floor=0.0`` (combine, then floor).

    Zero-weight rules are dropped; any NaN among the weighted rules gives NaN for that day.
    """
    validate_weights(weights)
    active = {r: w for r, w in weights.items() if w != 0.0}
    missing = [r for r in active if r not in forecasts.columns]
    if missing:
        raise ValueError(f"forecast columns missing for weighted rules: {missing}")
    w = pd.Series(active)
    combined = forecasts[list(active)].mul(w, axis=1).sum(axis=1, skipna=False) * fdm_value
    return combined.clip(lower=floor, upper=cap)
