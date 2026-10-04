"""Unified cost model with sigma-scaled square-root impact, futures roll cost and a stress multiplier (P2-04).

All components are in account currency. ``qty`` is signed (negative = sell) in shares or contracts; ``adv`` is in the same unit as ``qty``;
``vol`` is the daily return sigma; ``spread`` is a FRACTION of price (``spread >= 1`` raises). There is deliberately no flat-bps entry point.

Relation to the legacy helpers in ``firm.agents._liquidity``: that module is intentionally NOT imported here (the live ExecutionAgent depends on it
and must stay untouched). Its impact is ``coefficient * sqrt(notional / ADV$)``; the source plan's is ``k * sigma * sqrt(qty / ADV)``, which also
scales with volatility. The new model owns the latter; ``adv`` is the same quantity as ``estimate_adv_dollars`` expressed in qty units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "costs.yaml"
_COMPONENTS = ("commission", "exchange_fees", "half_spread", "impact", "roll")


@dataclass(frozen=True)
class CostBreakdown:
    commission: float
    exchange_fees: float
    half_spread: float
    impact: float
    roll: float
    total: float

    def scaled(self, m: float) -> CostBreakdown:
        """Every component, and the total, times ``m`` (gate G-RESEARCH 4 stress)."""
        if m < 0 or not math.isfinite(m):
            raise ValueError(f"multiplier must be finite and >= 0, got {m}")
        return CostBreakdown(*(getattr(self, f.name) * m for f in fields(self)))


@dataclass(frozen=True)
class InstrumentCostSpec:
    """Loaded from config/costs.yaml ``specs``. ``multiplier`` is the CONTRACT multiplier (1.0 for ETFs), not the stress multiplier."""

    kind: Literal["etf", "future"]
    broker: Literal["ibkr", "alpaca"]
    multiplier: float = 1.0
    half_spread_bps: float | None = None
    impact_k: float = 1.0
    roll_spreads_per_roll: int = 2


def load_cost_config(path: Path = _DEFAULT_PATH) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or "schedules" not in cfg:
        raise ValueError(f"{path}: not a cost config")
    return cfg


def spec_from_config(name: str, cfg: dict | None = None) -> InstrumentCostSpec:
    cfg = cfg or load_cost_config()
    return InstrumentCostSpec(**cfg["specs"][name])


def _schedule(spec: InstrumentCostSpec, cfg: dict) -> dict:
    key = f"{spec.broker}_{spec.kind}"
    try:
        return cfg["schedules"][key]
    except KeyError:
        raise ValueError(f"no cost schedule {key!r} in config") from None


def _check_spread(spread: float) -> None:
    if not (0 <= spread < 1):
        raise ValueError(f"spread must be a fraction of price in [0, 1), got {spread}")


def _crossing_half_spread(spec: InstrumentCostSpec, qty: float, price: float, spread: float | None, cfg: dict) -> float:
    """Currency cost of crossing half the spread once on ``qty`` units."""
    if spread is None:
        spread = 2.0 * spec.half_spread_bps / 1e4 if spec.half_spread_bps is not None else float(cfg["default_spread_fraction"])
    _check_spread(spread)
    return 0.5 * spread * price * abs(qty) * spec.multiplier


def roll_cost(spec: InstrumentCostSpec, qty: float, price: float, spread: float | None = None, cfg: dict | None = None) -> float:
    """``roll_spreads_per_roll`` (default 2) spread crossings: close the expiring contract and open the next."""
    cfg = cfg or load_cost_config()
    return spec.roll_spreads_per_roll * _crossing_half_spread(spec, qty, price, spread, cfg)


def _commission_and_fees(spec: InstrumentCostSpec, qty: float, price: float, cfg: dict) -> tuple[float, float]:
    sch = _schedule(spec, cfg)
    n = abs(qty)
    value = n * price * spec.multiplier
    c, f = sch["commission"], sch["exchange_fees"]
    if spec.kind == "future":
        return n * c["per_contract"], n * f["per_contract"]
    commission = n * c["per_share"]
    commission = max(commission, c.get("min_per_order", 0.0))
    commission = min(commission, c.get("max_pct_of_value", 1.0) * value)
    fees = n * f["per_share"]
    if qty < 0:  # regulatory pass-throughs apply to sells only
        reg = sch.get("regulatory_on_sells") or {}
        fees += reg.get("sec_rate_of_value", 0.0) * value
        fees += min(reg.get("taf_per_share", 0.0) * n, reg.get("taf_max_per_order", math.inf))
    return commission, fees


def cost(
    spec: InstrumentCostSpec,
    qty: float,
    price: float,
    adv: float,
    spread: float | None,
    vol: float,
    *,
    multiplier: float = 1.0,
    is_roll: bool = False,
    cfg: dict | None = None,
) -> CostBreakdown:
    """Cost of trading ``qty`` units.

    impact = k * sigma * sqrt(|qty| / ADV) * notional. ``k * sigma * sqrt(...)`` is a fractional price move, so multiplying by notional
    (|qty| * price * contract multiplier) gives currency. ``multiplier`` is the stress multiplier and scales every component.
    ``qty == 0`` returns zeros; ``adv <= 0`` raises (no silent fallback).
    """
    cfg = cfg or load_cost_config()
    if adv <= 0 or not math.isfinite(adv):
        raise ValueError(f"adv must be positive, got {adv}")
    if price <= 0 or vol < 0:
        raise ValueError("price must be > 0 and vol >= 0")
    if spread is not None:
        _check_spread(spread)
    if qty == 0:
        return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0).scaled(multiplier)
    notional = abs(qty) * price * spec.multiplier
    commission, fees = _commission_and_fees(spec, qty, price, cfg)
    half = _crossing_half_spread(spec, qty, price, spread, cfg)
    impact = spec.impact_k * vol * math.sqrt(abs(qty) / adv) * notional
    roll = spec.roll_spreads_per_roll * half if is_roll else 0.0
    base = CostBreakdown(commission, fees, half, impact, roll, commission + fees + half + impact + roll)
    return base.scaled(multiplier)


def empirical_half_spread(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """Corwin-Schultz (2012) high-low spread estimator; returns the average HALF spread as a fraction of price.

    Two-day estimates: beta = sum of squared log(H/L) over days t, t+1; gamma = squared log of the 2-day high over the 2-day low;
    alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2)); S = 2 (e^alpha - 1) / (1 + e^alpha).
    Overnight gaps are removed by shifting day t+1's high/low when the prior close lies outside them. Negative estimates are set to zero
    (as in the paper) before averaging.

    Caveat: accurate when the spread is comparable to daily volatility; when daily vol is much larger than the spread (liquid ETFs) the
    clipped estimator is biased UP (conservative for costs). Treat it as an upper-ish bound there; do not use it to claim a tight spread.
    """
    h = high.to_numpy(float)
    l = low.to_numpy(float)
    c = close.to_numpy(float)
    if not (len(h) == len(l) == len(c)) or len(h) < 2:
        raise ValueError("need aligned high/low/close with at least 2 observations")
    h1, l1 = h[1:].copy(), l[1:].copy()
    prev_c = c[:-1]
    up = l1 > prev_c  # gap up: shift down
    h1[up] -= (l1 - prev_c)[up]
    l1[up] = prev_c[up]
    dn = h1 < prev_c  # gap down: shift up
    l1[dn] += (prev_c - h1)[dn]
    h1[dn] = prev_c[dn]
    h0, l0 = h[:-1], l[:-1]
    beta = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
    gamma = np.log(np.maximum(h0, h1) / np.minimum(l0, l1)) ** 2
    k = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isfinite(s), np.maximum(s, 0.0), np.nan)
    return float(np.nanmean(s)) / 2.0


def legacy_flat_bps(notional: float, bps_per_side: float, *, multiplier: float = 1.0) -> float:
    """Legacy research convention (flat bps per side), for COMPARISON only; never used by the gate (G-RESEARCH 4)."""
    return abs(notional) * bps_per_side / 1e4 * multiplier


def annual_cost_drag(trades: pd.DataFrame, nav: pd.Series, *, specs: dict[str, InstrumentCostSpec], multiplier: float = 1.0,
                     cfg: dict | None = None) -> float:
    """Total modelled cost / mean NAV, per year of the NAV index span.

    ``trades``: columns ``symbol, qty, price, adv, vol`` and optional ``spread``, ``is_roll``. ``specs`` maps symbol -> spec.
    """
    if len(nav) < 2:
        raise ValueError("nav needs at least 2 points")
    cfg = cfg or load_cost_config()
    total = 0.0
    for r in trades.itertuples(index=False):
        spread = getattr(r, "spread", None)
        spread = None if spread is None or pd.isna(spread) else float(spread)
        total += cost(specs[r.symbol], r.qty, r.price, r.adv, spread, r.vol, multiplier=multiplier,
                      is_roll=bool(getattr(r, "is_roll", False)), cfg=cfg).total
    years = (pd.Timestamp(nav.index[-1]) - pd.Timestamp(nav.index[0])).days / 365.25
    if years <= 0:
        raise ValueError("nav index spans no time")
    return total / float(nav.mean()) / years
