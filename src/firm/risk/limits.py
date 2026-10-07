"""Portfolio vol targeting, leverage and exposure limits (ticket P4-03; candidate-only, pure functions).

Sits conceptually before ``guard_order``; it is NOT wired into any live path and ``config/risk.yaml`` must not be loaded
from a live process. It does not reuse or edit ``Allocator``, ``RiskAgent`` or ``portfolio.optimizer``.

Pipeline (``apply_limits``): ``w1 = clip(tau / sigma_ewma, upper=max_vol_scale) * raw`` (source rule; ``raw`` is already
sized at tau by P3-06 so nothing here renormalises to tau), then ``apply_caps``: long-only, instrument risk-contribution
cap, asset-class risk-contribution cap, gross cap, futures margin cap.

Risk contribution ``RC_i = w_i (Sigma w)_i / (w' Sigma w)``. Caps are SHARES of portfolio risk, so scaling a group down
raises every other share; they are enforced by bisection on the group scale and iterated to a fixed point.

Implementation choices recorded for the OD-16 register (P0-08):

* ``vol_ewma_span`` and ``tau`` are required, no defaults; ``config/risk.yaml`` ships them as null and the loader raises.
* Risk-cap enforcement: bisection on the group scale in [0, 1] (200 steps), fixed point over instruments then classes
  (max 50 sweeps, tolerance 1e-10; non-convergence raises).
* A share cap that scaling cannot satisfy is logged and skipped, never raised and never used to drop an instrument:
  (a) the group is all of the risk (a share of 1 cannot be diluted), (b) the caps of the held groups sum to less than 1
  (e.g. fewer than 3 classes held at a 0.40 class cap).
* Futures margin use is ``sum_i |w_i| * margin_rate_i`` against ``margin_limit`` (default 0.30 of equity); the ticket does not
  say where the per-instrument margin rate comes from, so ``margin_rate`` is a required input for futures.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from firm.portfolio.sizing import gross_cap_scale

log = logging.getLogger(__name__)

__all__ = [
    "LimitResult",
    "RiskLimits",
    "apply_caps",
    "apply_limits",
    "ewma_realised_vol",
    "gross_cap_fraction",
    "load_limits",
    "risk_contributions",
    "vol_scale",
]

TRADING_DAYS = 252
MAX_GROSS_ETF = 1.0
MAX_GROSS_FUTURES = 4.0
DEFAULT_FUTURES_MARGIN_LIMIT = 0.30
_FIXED_POINT_ITERS = 50
_FIXED_POINT_TOL = 1e-10
_BISECT_STEPS = 200
_MIN_SCALE = 1e-9


@dataclass(frozen=True, kw_only=True)
class RiskLimits:
    tau: float  # annual target vol, from the charter (P5-02); no default
    vol_ewma_span: (
        int  # slow EWMA span (trading days); frozen in risk.yaml before P3-11; no default
    )
    instrument_type: Literal["etf", "futures"]
    max_gross: float  # validated: <= 1.0 (etf), <= 4.0 (futures)
    long_only: bool
    asset_class: Mapping[str, str]  # instrument -> class (universe_etf.yaml)
    handcraft_share: Mapping[str, float]  # P4-01 risk shares, used only for the instrument cap
    max_vol_scale: float = 1.5  # source leverage-scale cap; frozen, never tuned
    max_class_risk_share: float = 0.40
    max_instrument_risk_mult: float = 2.0
    margin_limit: float | None = None  # futures: defaults to 0.30 of equity; must stay None for ETF
    margin_rate: Mapping[str, float] | None = (
        None  # futures: initial margin as a fraction of notional
    )

    def __post_init__(self) -> None:
        if self.instrument_type not in ("etf", "futures"):
            raise ValueError(
                f"instrument_type must be 'etf' or 'futures', got {self.instrument_type!r}"
            )
        if not (isinstance(self.tau, (int, float)) and math.isfinite(self.tau) and self.tau > 0):
            raise ValueError(f"tau must be a positive finite number, got {self.tau!r}")
        if not (
            isinstance(self.vol_ewma_span, int)
            and not isinstance(self.vol_ewma_span, bool)
            and self.vol_ewma_span >= 2
        ):
            raise ValueError(f"vol_ewma_span must be an integer >= 2, got {self.vol_ewma_span!r}")
        source_cap = MAX_GROSS_ETF if self.instrument_type == "etf" else MAX_GROSS_FUTURES
        if not (0 < self.max_gross <= source_cap):
            raise ValueError(
                f"max_gross must be in (0, {source_cap}] for {self.instrument_type}, got {self.max_gross!r}"
            )
        if not self.max_vol_scale > 0:
            raise ValueError("max_vol_scale must be positive")
        if not (0 < self.max_class_risk_share <= 1.0):
            raise ValueError("max_class_risk_share must be in (0, 1]")
        if not self.max_instrument_risk_mult > 0:
            raise ValueError("max_instrument_risk_mult must be positive")
        if self.instrument_type == "etf":
            if self.margin_limit is not None or self.margin_rate is not None:
                raise ValueError("margin settings apply to futures only")
        else:
            if self.margin_limit is None:
                object.__setattr__(self, "margin_limit", DEFAULT_FUTURES_MARGIN_LIMIT)
            if not 0 < self.margin_limit <= 1.0:
                raise ValueError("margin_limit must be in (0, 1]")
            if not self.margin_rate:
                raise ValueError(
                    "futures limits require margin_rate (initial margin per unit notional, per instrument)"
                )


@dataclass(frozen=True)
class LimitResult:
    weights: pd.Series  # post-limit weights (fraction of NAV, signed)
    scale: float  # vol-target scalar applied (<= max_vol_scale); 1.0 from apply_caps alone
    gross_cap_bound: bool
    class_cap_bound: dict[str, bool]
    instrument_cap_bound: dict[str, bool]
    breaches_clipped: list[str]


def load_limits(
    path: str | os.PathLike = "config/risk.yaml",
    *,
    handcraft_share: Mapping[str, float] | None = None,
    asset_class: Mapping[str, str] | None = None,
) -> RiskLimits:
    """Load limits. ``tau`` and ``vol_ewma_span`` must be present and non-null (no placeholder value is ever invented).

    ``handcraft_share`` (P4-01 output) is a required argument. ``asset_class`` defaults to the ``asset_classes`` block of
    the YAML named by the ``universe`` key.
    """
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    for key in ("tau", "vol_ewma_span"):
        if cfg.get(key) is None:
            raise ValueError(
                f"{path}: {key!r} is missing; it is frozen elsewhere (charter / pre-P3-11 freeze), never defaulted"
            )
    if handcraft_share is None:
        raise ValueError("handcraft_share (P4-01 weights) is required")
    if asset_class is None:
        uni_path = cfg.get("universe")
        if not uni_path:
            raise ValueError(f"{path}: no 'universe' key and no asset_class mapping given")
        with open(uni_path, encoding="utf-8") as fh:
            uni = yaml.safe_load(fh)
        asset_class = {sym: cls for cls, syms in uni["asset_classes"].items() for sym in syms}
    kw: dict[str, Any] = {
        k: cfg[k]
        for k in (
            "max_vol_scale",
            "max_class_risk_share",
            "max_instrument_risk_mult",
            "margin_limit",
            "margin_rate",
        )
        if cfg.get(k) is not None
    }
    lim = RiskLimits(
        tau=cfg["tau"],
        vol_ewma_span=cfg["vol_ewma_span"],
        instrument_type=cfg["instrument_type"],
        max_gross=cfg["max_gross"],
        long_only=cfg["long_only"],
        asset_class=dict(asset_class),
        handcraft_share=dict(handcraft_share),
        **kw,
    )
    log.info(
        "loaded risk limits from %s (type=%s, max_gross=%s)",
        path,
        lim.instrument_type,
        lim.max_gross,
    )
    return lim


def ewma_realised_vol(portfolio_returns: pd.Series, span: int) -> float:
    """Annualised slow-EWMA realised vol of the forecast-sized portfolio's daily returns (zero-mean, RiskMetrics style).

    ``sqrt(252 * EWMA(r^2))`` with ``alpha = 2 / (span + 1)``. Needs at least ``span`` finite observations. The caller must
    pass a lagged series (no look-ahead).
    """
    r = pd.Series(portfolio_returns, dtype=float).dropna()
    if span < 2 or len(r) < span:
        raise ValueError(f"need span >= 2 and at least span={span} observations, got {len(r)}")
    ms = float((r**2).ewm(span=span, adjust=True).mean().iloc[-1])
    return math.sqrt(TRADING_DAYS * ms)


def _as_cov(weights: pd.Series, cov_annual: pd.DataFrame) -> np.ndarray:
    try:
        c = cov_annual.loc[weights.index, weights.index].to_numpy(dtype=float)
    except KeyError as exc:
        raise ValueError(f"covariance is missing instruments: {exc}") from exc
    if not np.isfinite(c).all():
        raise ValueError("covariance contains non-finite values")
    return (c + c.T) / 2.0


def risk_contributions(weights: pd.Series, cov_annual: pd.DataFrame) -> pd.Series:
    """``RC_i = w_i (Sigma w)_i / (w' Sigma w)``; sums to 1 (all zeros for an all-zero portfolio)."""
    w = weights.to_numpy(dtype=float)
    c = _as_cov(weights, cov_annual)
    return pd.Series(_rc(w, c), index=weights.index)


def _rc(w: np.ndarray, c: np.ndarray) -> np.ndarray:
    if not np.any(w):
        return np.zeros_like(w)
    mc = c @ w
    var = float(w @ mc)
    if not var > 0:
        raise ValueError("portfolio variance is not positive for a non-zero portfolio")
    return w * mc / var


def vol_scale(sigma_ewma: float, limits: RiskLimits) -> float:
    """``clip(tau / sigma_ewma, upper=max_vol_scale)``; 0 if ``sigma_ewma <= 0`` (logged)."""
    if sigma_ewma is None or not math.isfinite(sigma_ewma):
        raise ValueError(f"sigma_ewma must be finite, got {sigma_ewma!r}")
    if sigma_ewma <= 0:
        log.warning(
            "sigma_ewma=%s <= 0: vol scale set to 0 (flat) rather than dividing by zero", sigma_ewma
        )
        return 0.0
    return float(min(limits.tau / sigma_ewma, limits.max_vol_scale))


def _closed_form_scale(w: np.ndarray, c: np.ndarray, members: np.ndarray, cap: float) -> float | None:
    """Exact scale on ``members`` with risk share == cap (``solver="closed_form"``; ``None`` if the share cannot be diluted).

    With ``x = w`` scaled by k on the members: ``share(k) = (k^2 a + k b) / (k^2 a + 2 k b + d)`` where ``a = w_M' C_MM w_M``,
    ``b = w_M' C_M,rest w_rest``, ``d = w_rest' C_rest w_rest``; ``share(k) = cap`` is the quadratic
    ``a (1 - cap) k^2 + b (1 - 2 cap) k - cap d = 0`` whose positive root is returned (the stable form of the quadratic formula).
    """
    mask = np.zeros(len(w), dtype=bool)
    mask[members] = True
    wm = w[mask]
    u = c @ w
    var0 = float(w @ u)
    a = float(wm @ c[np.ix_(mask, mask)] @ wm)
    b = float(wm @ u[mask]) - a
    d = var0 - a - 2.0 * b
    if not d > 0 or not a > 0:
        return None
    A, B = a * (1.0 - cap), b * (1.0 - 2.0 * cap)
    disc = math.sqrt(max(B * B + 4.0 * A * cap * d, 0.0))
    k = 2.0 * cap * d / (B + disc) if B > 0 else (-B + disc) / (2.0 * A)
    return float(min(max(k, _MIN_SCALE), 1.0))


def _group_scale_to_cap(
    w: np.ndarray, c: np.ndarray, members: np.ndarray, cap: float, *, solver: str = "bisection", tol: float = _FIXED_POINT_TOL
) -> float | None:
    """Scale in (0, 1] on ``members`` so their risk share is <= cap after the rest is re-normalised.

    Returns 1.0 if already within the cap, ``None`` if no scale can satisfy it (share cannot be diluted).
    ``solver="bisection"`` (default, the P4-03 rule) or ``"closed_form"`` (the exact quadratic root, same fixed point, much faster).
    """

    def share(k: float) -> float:
        x = w.copy()
        x[members] *= k
        return float(_rc(x, c)[members].sum())

    if share(1.0) <= cap + tol:
        return 1.0
    if solver == "closed_form":
        return _closed_form_scale(w, c, members, cap)
    if share(_MIN_SCALE) > cap:
        return None
    lo, hi = _MIN_SCALE, 1.0  # share(lo) <= cap (feasible side), share(hi) > cap
    for _ in range(_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if share(mid) <= cap:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return lo


def apply_caps(
    weights: pd.Series, cov_annual: pd.DataFrame, limits: RiskLimits, *, solver: Literal["bisection", "closed_form"] = "bisection",
    max_sweeps: int = _FIXED_POINT_ITERS, tol: float = _FIXED_POINT_TOL, on_nonconvergence: Literal["raise", "return"] = "raise",
) -> LimitResult:
    """Long-only, instrument-RC cap, class-RC cap, gross cap, margin. Idempotent for fixed inputs.

    The keyword-only options default to the P4-03 rule (bisection, 50 sweeps, 1e-10, non-convergence raises) and exist for research
    backtests that call this once per day: ``solver="closed_form"`` solves each group scale exactly, ``max_sweeps`` / ``tol`` bound the
    fixed point, and ``on_nonconvergence="return"`` keeps the last (down-scaled) iterate and lists ``"nonconverged"`` in ``breaches_clipped``
    instead of raising.
    """
    if solver not in ("bisection", "closed_form") or on_nonconvergence not in ("raise", "return"):
        raise ValueError("solver must be 'bisection'|'closed_form' and on_nonconvergence 'raise'|'return'")
    nonconverged = False
    names = list(weights.index)
    w = weights.to_numpy(dtype=float).copy()
    if not np.isfinite(w).all():
        raise ValueError("weights contain non-finite values")
    c = _as_cov(weights, cov_annual)
    clipped: list[str] = []
    inst_bound = dict.fromkeys(names, False)
    classes = sorted({limits.asset_class[n] for n in names if n in limits.asset_class})
    missing_cls = [n for n in names if n not in limits.asset_class]
    if missing_cls:
        raise ValueError(f"no asset class for instruments: {missing_cls}")
    class_bound = dict.fromkeys(classes, False)

    if limits.long_only:
        neg = w < 0
        for i in np.where(neg)[0]:
            log.info("long-only: clipped %s weight %.6g to 0", names[i], w[i])
            clipped.append(f"long_only:{names[i]}")
        w[neg] = 0.0

    held = np.where(w != 0)[0]
    skip: set[str] = set()
    if held.size:
        inst_cap = {}
        for i in held:
            if names[i] not in limits.handcraft_share:
                raise ValueError(f"no handcraft share for held instrument {names[i]!r}")
            inst_cap[i] = limits.max_instrument_risk_mult * limits.handcraft_share[names[i]]
        if sum(inst_cap.values()) < 1.0 - 1e-12:
            log.warning(
                "instrument risk caps infeasible for the held set (caps sum %.4g < 1): skipped",
                sum(inst_cap.values()),
            )
            skip.update(f"i:{names[i]}" for i in held)
        held_classes = {limits.asset_class[names[i]] for i in held}
        if len(held_classes) * limits.max_class_risk_share < 1.0 - 1e-12:
            log.warning(
                "class risk cap infeasible: %d class(es) held at cap %.2f: skipped",
                len(held_classes),
                limits.max_class_risk_share,
            )
            skip.update(f"c:{k}" for k in held_classes)
        members_of = {
            k: np.array([i for i in held if limits.asset_class[names[i]] == k])
            for k in held_classes
        }

        for sweep in range(max_sweeps + 1):
            changed = False
            for i in held:
                key = f"i:{names[i]}"
                if key in skip or w[i] == 0:
                    continue
                k = _group_scale_to_cap(w, c, np.array([i]), inst_cap[i], solver=solver, tol=tol)
                if k is None:
                    log.warning("instrument cap infeasible for %s: skipped", names[i])
                    skip.add(key)
                elif k < 1.0:
                    log.info("instrument risk cap: scaled %s by %.6g", names[i], k)
                    w[i] *= k
                    inst_bound[names[i]] = True
                    changed = True
            for cls, mem in members_of.items():
                key = f"c:{cls}"
                if key in skip:
                    continue
                k = _group_scale_to_cap(w, c, mem, limits.max_class_risk_share, solver=solver, tol=tol)
                if k is None:
                    log.warning("class cap infeasible for %s: skipped", cls)
                    skip.add(key)
                elif k < 1.0:
                    log.info("class risk cap: scaled class %s by %.6g", cls, k)
                    w[mem] *= k
                    class_bound[cls] = True
                    changed = True
            if not changed:
                break
            if sweep == max_sweeps:
                if on_nonconvergence == "raise":
                    raise RuntimeError(
                        f"risk-contribution caps did not converge in {max_sweeps} sweeps"
                    )
                log.warning("risk-contribution caps did not converge in %d sweeps: last iterate kept", max_sweeps)
                nonconverged = True
    clipped += [f"instrument:{n}" for n, b in inst_bound.items() if b]
    clipped += [f"class:{k}" for k, b in class_bound.items() if b]
    clipped += [f"infeasible:{s}" for s in sorted(skip)]
    if nonconverged:
        clipped.append("nonconverged")

    # uniform scalings below leave every risk share unchanged, so the caps above still hold afterwards
    scaled, gross_bound = gross_cap_scale(dict(zip(names, w.tolist())), cap=limits.max_gross)
    if gross_bound:
        log.info("gross cap bound: scaled to %.4g", limits.max_gross)
        clipped.append("gross")
        w = np.array([scaled[n] for n in names])

    if limits.instrument_type == "futures":
        rates = np.array([limits.margin_rate.get(n, math.nan) for n in names])
        if np.isnan(rates[w != 0]).any():
            raise ValueError("margin_rate missing for a held futures instrument")
        use = float(np.nansum(np.abs(w) * np.nan_to_num(rates)))
        if use > limits.margin_limit + 1e-12:
            log.info("margin cap bound: use %.4g > %.4g, scaled down", use, limits.margin_limit)
            w = w * (limits.margin_limit / use)
            clipped.append("margin")

    return LimitResult(
        pd.Series(w, index=weights.index), 1.0, gross_bound, class_bound, inst_bound, clipped
    )


def apply_limits(
    raw_weights: pd.Series, cov_annual: pd.DataFrame, limits: RiskLimits, *, sigma_ewma: float
) -> LimitResult:
    """``scale = vol_scale(sigma_ewma)``; ``apply_caps(scale * raw)``. Not idempotent: ``scale`` depends on outside state."""
    s = vol_scale(sigma_ewma, limits)
    res = apply_caps(raw_weights * s, cov_annual, limits)
    port_vol = float(
        np.sqrt(
            max(
                res.weights.to_numpy() @ _as_cov(res.weights, cov_annual) @ res.weights.to_numpy(),
                0.0,
            )
        )
    )
    log.info(
        "apply_limits: scale=%.4g gross=%.4g portfolio_vol(diag)=%.4g",
        s,
        res.weights.abs().sum(),
        port_vol,
    )
    return LimitResult(
        res.weights,
        s,
        res.gross_cap_bound,
        res.class_cap_bound,
        res.instrument_cap_bound,
        res.breaches_clipped,
    )


def gross_cap_fraction(results: Sequence[LimitResult]) -> float:
    """Fraction of days (results) on which the gross cap was binding (OD-17 ceiling 0.20, reported by P3-08)."""
    if len(results) == 0:
        raise ValueError("no results")
    return float(sum(1 for r in results if r.gross_cap_bound) / len(results))
