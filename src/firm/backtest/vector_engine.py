"""Signed, multiplier-aware, cost-model-driven vector backtest engine (ticket P3-09).

Research-only. NOT imported by ``firm/backtest/__init__.py`` or any live module (P0-06 isolation test). This module does not import
``backtrader`` (the package ``__init__`` does; the guarantee here is a static AST check). It never touches files: callers hand it
arrays/frames (real runs go through ``firm.research.data_access``) and every non-exploratory run is recorded in the trial ledger.

Timing (the ONE place the fill lag is applied). ``target_positions`` are indexed by the date they were DECIDED (close t). With
``fill_lag_bars = L`` the position held after bar t is ``target[t - L]``; the trade ``held_t - held_{t-1}`` is costed at ``P_t`` and the
P&L of bar t is earned by the PRIOR position::

    pnl_t   = sum_i held_{t-1,i} * mult_i * (P_{t,i} - P_{t-1,i}) * fx_{t,i}
    cash_t  = rf_t * (E_{t-1} - sum_i held_{t-1,i} * mult_i * P_{t-1,i} * fx_{t-1,i})      (0 when rf is None)
    ret_t   = (pnl_t + cash_t - cost_t - borrow_t) / E_{t-1},   E_t = E_{t-1} * (1 + ret_t)

Costs are in account currency and charged on the post-return equity, so ``cost / E_{t-1}`` is larger than the legacy
``run_alt_premia_evaluation.simulate`` fraction-of-pre-return-equity by about ``(1 + rp_t)`` per rebalance day (documented, tested).
Roll costs: on a ``roll_flags`` date the carried position ``held_{t-1}`` is passed to ``cost_fn`` with ``is_roll=True`` and only the
``roll`` component of that second call is kept, so commission/fees/spread are not charged twice. A roll with no carried position is free.
If equity is exhausted (``E <= 0``) later returns are 0.

Sharpe convention: ``excess_returns = returns - rf`` (== ``returns`` when ``rf is None``); the ledger records the excess series.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ENGINE_VERSION = "1"
_COMPONENTS = ("commission", "exchange_fees", "half_spread", "impact", "roll", "total")
_DAYS = 252.0


class LedgerRequiredError(RuntimeError):
    """A non-exploratory run without a ledger context (AGENTS.md rule 2)."""


class CostFn(Protocol):
    def __call__(self, symbol: str, date, qty_delta: float, price: float, adv: float | None,
                 vol_pct: float | None, multiplier: float, is_roll: bool, stress: float) -> Any: ...


@dataclass(frozen=True)
class EngineConfig:
    initial_capital: float
    stress_multiplier: float = 1.0
    max_gross: float | None = None  # None for the engine; limits live in P4-03
    allow_short: bool = True
    fractional: dict[str, bool] | None = None  # symbol -> False truncates units toward zero; missing/None = fractional
    fill_lag_bars: int = 1  # signal at close t, filled at close t+1 (default)
    borrow_rate_annual: float = 0.0  # on |market value| of short positions, per 252 days


@dataclass(frozen=True)
class LedgerContext:
    """What ``run_vector_backtest`` needs to record a run via ``firm.research.ledger.run_trial`` (the machinery under ``backtest_logged``)."""

    family: str
    mode: str = "exploratory"  # "registered" requires prereg and an approved pre-registration
    prereg: str | None = None
    data_snapshot_id: str | None = None
    seed: int | None = None
    config: dict = field(default_factory=dict)  # strategy parameters / provenance recorded with the trial
    n_variants: int = 1


@dataclass
class EngineResult:
    returns: pd.Series
    excess_returns: pd.Series
    gross_returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame  # CostBreakdown components by day plus ``borrow`` (``total`` excludes borrow)
    gross_exposure: pd.Series
    gross_cap_bound_days: float  # share of days the supplied flags were True; NaN when no flags were supplied
    meta: dict


# ------------------------------------------------------------------ helpers

def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _aligned(name: str, frame: pd.DataFrame | None, like: pd.DataFrame, fill: float | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    out = frame.reindex(index=like.index, columns=like.columns)
    if fill is not None:
        out = out.fillna(fill)
    return out


def _cell(frame: pd.DataFrame | None, i: int, j: int) -> float | None:
    if frame is None:
        return None
    v = frame.iat[i, j]
    return None if pd.isna(v) else float(v)


def _validate(prices, targets, multipliers, config, rf):
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    if config.fill_lag_bars < 0:
        raise ValueError("fill_lag_bars must be >= 0")
    if not config.initial_capital > 0:
        raise ValueError("initial_capital must be positive")
    if config.stress_multiplier < 0 or not math.isfinite(config.stress_multiplier):
        raise ValueError("stress_multiplier must be finite and >= 0")
    missing = [c for c in prices.columns if c not in multipliers]
    if missing:
        raise ValueError(f"multiplier missing for {missing}")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("prices index must be strictly increasing")
    if prices.isna().any().any():
        raise ValueError("prices contain NaN; clean/forward-fill upstream (the engine never invents prices)")
    if (prices <= 0).any().any():
        raise ValueError("prices must be positive")
    tg = targets.reindex(index=prices.index, columns=prices.columns)
    if tg.isna().any().any():
        raise ValueError("target_positions contain NaN or do not cover prices; use 0 for flat")
    if not config.allow_short and (tg < 0).any().any():
        raise ValueError("allow_short=False but target_positions contain short (negative) units")
    if rf is not None:
        r = rf.reindex(prices.index)
        if r.isna().any():
            raise ValueError("rf has NaN or does not cover the price index")
    return tg


def _sharpe(x: pd.Series) -> float | None:
    s = x.std(ddof=1)
    return float(x.mean() / s) if len(x) > 2 and s > 0 else None


def _record(res: EngineResult, h, cfg: EngineConfig, rf: pd.Series | None, ctx: LedgerContext) -> None:
    ex = res.excess_returns
    gx = res.gross_returns - (rf if rf is not None else 0.0)
    kw: dict[str, Any] = {
        "start": str(ex.index[0].date()), "end": str(ex.index[-1].date()), "n_obs": len(ex), "periods_per_year": int(_DAYS),
        "sharpe_conversion": "per_period_excess_over_rf; annualise by sqrt(periods_per_year)",
    }
    if ctx.data_snapshot_id is not None:
        kw["data_snapshot_id"] = ctx.data_snapshot_id
    if ctx.seed is not None:
        kw["seed"] = ctx.seed
    ns, gs = _sharpe(ex), _sharpe(gx)
    if ns is not None:
        kw["net_sharpe"] = ns
    if gs is not None:
        kw["gross_sharpe"] = gs
    if len(ex) > 3 and ex.std(ddof=0) > 0:
        z = (ex - ex.mean()) / ex.std(ddof=0)
        kw["skew"], kw["kurt"] = float((z**3).mean()), float((z**4).mean())
    kw["n_variants"] = ctx.n_variants
    h.set(**kw)
    h.returns = ex


# ------------------------------------------------------------------ core

def _simulate(prices, tg, multipliers, cost_fn, config, adv, vol_pct, roll_flags, fx, rf, gross_cap_bound) -> EngineResult:
    dates, syms = prices.index, list(prices.columns)
    T, N = prices.shape
    mult = np.array([float(multipliers[s]) for s in syms])
    P = prices.to_numpy(float)
    FX = np.ones((T, N)) if fx is None else fx.reindex(index=dates, columns=syms).ffill().fillna(1.0).to_numpy(float)
    RF = np.zeros(T) if rf is None else rf.reindex(dates).to_numpy(float)
    units = tg.to_numpy(float).copy()
    for j, s in enumerate(syms):
        if config.fractional and config.fractional.get(s) is False:
            units[:, j] = np.trunc(units[:, j])
    lag = config.fill_lag_bars
    held = np.zeros((T, N))
    held[lag:] = units[: T - lag] if lag else units
    adv_f, vol_f = _aligned("adv", adv, prices, None), _aligned("vol_pct", vol_pct, prices, None)
    roll_f = None if roll_flags is None else roll_flags.reindex(index=dates, columns=syms).fillna(False).to_numpy(bool)

    ret = np.zeros(T)
    gross = np.zeros(T)
    comp = np.zeros((T, len(_COMPONENTS)))
    borrow = np.zeros(T)
    trades: list[dict] = []
    equity = float(config.initial_capital)
    prev = np.zeros(N)
    daily_borrow = config.borrow_rate_annual / _DAYS
    for t in range(T):
        if equity <= 0:
            held[t:] = 0.0
            break
        pnl = cash = 0.0
        if t > 0:
            pnl = float(np.sum(prev * mult * (P[t] - P[t - 1]) * FX[t]))
            mv_prev = prev * mult * P[t - 1] * FX[t - 1]
            cash = RF[t] * (equity - float(mv_prev.sum()))
            borrow[t] = daily_borrow * float(-mv_prev[mv_prev < 0].sum())
        elif RF[t] != 0.0:
            cash = RF[t] * equity
        dq = held[t] - prev
        for j in np.flatnonzero(dq != 0.0):
            b = cost_fn(syms[j], dates[t], float(dq[j]), float(P[t, j]), _cell(adv_f, t, j), _cell(vol_f, t, j),
                        float(mult[j]), False, config.stress_multiplier)
            vals = np.array([float(getattr(b, c)) for c in _COMPONENTS]) * FX[t, j]
            comp[t] += vals
            trades.append({"date": dates[t], "symbol": syms[j], "qty": float(dq[j]), "price": float(P[t, j]),
                           **dict(zip(_COMPONENTS, vals.tolist(), strict=True))})
        if roll_f is not None:
            for j in np.flatnonzero(roll_f[t] & (prev != 0.0)):
                b = cost_fn(syms[j], dates[t], float(prev[j]), float(P[t, j]), _cell(adv_f, t, j), _cell(vol_f, t, j),
                            float(mult[j]), True, config.stress_multiplier)
                r = float(b.roll) * FX[t, j]
                comp[t, _COMPONENTS.index("roll")] += r
                comp[t, _COMPONENTS.index("total")] += r
        total_cost = comp[t, -1]
        gross[t] = (pnl + cash) / equity
        ret[t] = max((pnl + cash - total_cost - borrow[t]) / equity, -1.0)
        equity *= 1.0 + ret[t]
        prev = held[t]

    idx = dates
    returns = pd.Series(ret, index=idx, name="returns")
    rf_s = pd.Series(RF, index=idx)
    mv = held * mult * P * FX
    eq_path = float(config.initial_capital) * np.cumprod(1.0 + ret)
    costs = pd.DataFrame(comp, index=idx, columns=list(_COMPONENTS))
    costs["borrow"] = borrow
    trades_df = pd.DataFrame(trades, columns=["date", "symbol", "qty", "price", *_COMPONENTS])
    flags_share = float("nan")
    if gross_cap_bound is not None:
        flags_share = float(gross_cap_bound.reindex(idx).fillna(False).astype(bool).mean())
    return EngineResult(
        returns=returns, excess_returns=returns - rf_s, gross_returns=pd.Series(gross, index=idx, name="gross_returns"),
        positions=pd.DataFrame(held, index=idx, columns=syms), trades=trades_df, costs=costs,
        gross_exposure=pd.Series(np.abs(mv).sum(axis=1) / np.where(eq_path > 0, eq_path, np.nan), index=idx, name="gross_exposure"),
        gross_cap_bound_days=flags_share,
        meta={"engine_version": ENGINE_VERSION, "rf_used": rf is not None, "stress_multiplier": config.stress_multiplier,
              "n_trades": len(trades_df)},
    )


def run_vector_backtest(
    prices: pd.DataFrame, target_positions: pd.DataFrame, multipliers: dict[str, float], cost_fn: CostFn, config: EngineConfig,
    *, adv: pd.DataFrame | None = None, vol_pct: pd.DataFrame | None = None, roll_flags: pd.DataFrame | None = None,
    fx: pd.DataFrame | None = None, rf: pd.Series | None = None, gross_cap_bound: pd.Series | None = None,
    ledger_ctx: LedgerContext | None = None, exploratory: bool = False,
) -> EngineResult:
    """Run the engine. Refuses (``LedgerRequiredError``) unless ``ledger_ctx`` is given or ``exploratory=True``."""
    if ledger_ctx is None and not exploratory:
        raise LedgerRequiredError("pass ledger_ctx (recorded via the trial ledger) or exploratory=True (recorded as mode='exploratory')")
    if ledger_ctx is not None and exploratory and ledger_ctx.mode != "exploratory":
        raise ValueError("exploratory=True cannot be combined with a non-exploratory ledger_ctx")
    ctx = ledger_ctx or LedgerContext(family="vector_engine_exploratory", mode="exploratory")

    from firm.research import ledger as L  # lazy: keeps this module's import light

    cfg_dict = dataclasses.asdict(config)
    meta_cfg = {"engine_config": cfg_dict, "engine_version": ENGINE_VERSION, "multipliers": dict(multipliers),
                "symbols": list(map(str, prices.columns)), "rf_used": rf is not None, "ctx": ctx.config}
    chash = L.config_hash(json.loads(L.canonical_json(meta_cfg)))
    with L.run_trial(ctx.family, meta_cfg, mode=ctx.mode, prereg=ctx.prereg) as h:  # type: ignore[arg-type]
        tg = _validate(prices, target_positions, multipliers, config, rf)
        res = _simulate(prices, tg, multipliers, cost_fn, config, adv, vol_pct, roll_flags, fx, rf, gross_cap_bound)
        res.meta.update(config_hash=chash, data_snapshot_id=ctx.data_snapshot_id)
        _record(res, h, config, None if rf is None else rf.reindex(prices.index), ctx)
    # run_trial appends in ``finally``; the id is the newest row for this config hash
    tr = L.trials()
    res.meta["trial_id"] = tr[tr["config_hash"] == chash].iloc[-1]["trial_id"] if len(tr) else None
    log.info("vector engine: %d bars, %d trades, mode=%s", len(prices), res.meta["n_trades"], ctx.mode)
    return res


# ------------------------------------------------------------------ optional P4-03 helpers (research-only, additive)

def ewma_covariance(returns: pd.DataFrame, span: int, periods_per_year: float = 256.0) -> np.ndarray:
    """Annualised zero-mean EWMA covariance per date, shape (T, N, N): ``sum_k (1-a)^k r r' / sum_k (1-a)^k``, ``a = 2 / (span + 1)``.

    The bias-corrected form of the same ``adjust=True`` EWMA that ``firm.risk.limits.ewma_realised_vol`` uses. Each matrix is a convex
    combination of outer products, hence symmetric positive semi-definite. A NaN return (instrument not yet listed) counts as no information
    (0). Row t uses returns up to and including t (known at the close on which the target is decided).
    """
    if span < 2:
        raise ValueError("span must be >= 2")
    x = returns.fillna(0.0).to_numpy(float)
    T, N = x.shape
    a = 2.0 / (span + 1.0)
    out = np.empty((T, N, N))
    num, den = np.zeros((N, N)), 0.0
    for t in range(T):
        num = (1.0 - a) * num + np.outer(x[t], x[t])
        den = (1.0 - a) * den + 1.0
        out[t] = num / den * periods_per_year
    return out


def portfolio_vol_scale(raw_weights: pd.DataFrame, returns: pd.DataFrame, *, tau: float, span: int, max_scale: float,
                        periods_per_year: float = 256.0) -> pd.Series:
    """P4-03 vol-target scalar ``clip(tau / sigma_ewma, upper=max_scale)`` per decision date.

    ``sigma_ewma`` is the slow EWMA (``span``) realised vol of the FORECAST-SIZED portfolio: the return of day t is
    ``sum_i raw_weights[t-1, i] * returns[t, i]`` (the weight known one close earlier, so no look-ahead), where ``raw_weights`` are the
    unscaled, unbuffered, uncapped forecast-sized weights (NaN = no forecast yet). The series starts at the first date with a lagged weight
    and sigma needs ``span`` observations; before that the result is NaN (the sizing wrapper treats NaN as scale 1). ``sigma <= 0`` gives 0,
    as ``firm.risk.limits.vol_scale``.
    """
    w_lag = raw_weights.reindex(index=returns.index, columns=returns.columns).shift(1)
    rp = (w_lag.fillna(0.0) * returns.fillna(0.0)).sum(axis=1).where(w_lag.notna().any(axis=1))
    ms = (rp**2).ewm(span=span, adjust=True, min_periods=span).mean()
    sigma = np.sqrt(periods_per_year * ms)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(sigma > 0, np.minimum(tau / sigma, max_scale), 0.0)
    return pd.Series(np.where(sigma.isna(), np.nan, scale), index=returns.index, name="vol_scale")


# ------------------------------------------------------------------ optional P3-06 sizing wrapper

def targets_from_forecasts(
    forecasts: pd.DataFrame, prices: pd.DataFrame, vol_pct: pd.DataFrame, *, weights: dict[str, float], idm: float, tau: float,
    capital: float | pd.Series, multipliers: dict[str, float], fx: pd.DataFrame | None = None, buffer_fraction: float = 0.10,
    long_only: bool = True, gross_cap: float | None = 1.0, rounding: str = "toward_zero", fractional: dict[str, bool] | None = None,
    vol_scale: pd.Series | None = None, risk_limits: Any = None, cov_annual: np.ndarray | None = None,
    diagnostics: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Unit targets (indexed by decision date, NOT shifted) from combined forecasts via ``firm.portfolio.sizing`` (P3-06).

    Per day and instrument: source N, buffer around N (``buffered_trade``), pro-rata gross cap (``gross_cap_scale``), then rounding
    (after buffering, as P3-06). A NaN forecast/price/vol means no information: the instrument is flat that day. ``capital`` is a fixed
    number or a per-date series (the engine does not feed back its own equity). Returns (targets, gross_cap_bound flags).

    Optional P4-03 layer (all default ``None``: the result is then exactly the P3-06 path above, covered by tests):

    * ``vol_scale``: per decision date, the vol-target scalar ``s`` (``portfolio_vol_scale``). ``w1 = s * raw`` is implemented by sizing N and
      the buffer width on ``capital * s`` (NaN = 1, no estimate yet).
    * ``risk_limits`` (``firm.risk.limits.RiskLimits``) with ``cov_annual`` (T, N, N; ``ewma_covariance``): after the buffer step the held weights
      go through ``apply_caps`` (long-only, instrument risk-contribution cap, class cap if below 1, gross cap). The gross cap is ``gross_cap`` of
      this function, so a perturbed value above the limits' own ETF ceiling still works: risk shares are scale-invariant, so capping
      ``w * max_gross / gross_cap`` and scaling back is identical to capping at ``gross_cap``. ``diagnostics`` (a dict) receives the number of days
      on which an instrument cap or a class cap bound and the number of days with an infeasible-cap skip.
    """
    from firm.portfolio import sizing as S

    syms = list(prices.columns)
    fcs = forecasts.reindex(index=prices.index, columns=syms)
    vol = vol_pct.reindex(index=prices.index, columns=syms)
    fxf = None if fx is None else fx.reindex(index=prices.index, columns=syms).ffill().fillna(1.0)
    scale = None if vol_scale is None else vol_scale.reindex(prices.index).fillna(1.0)
    if risk_limits is not None:
        if cov_annual is None or np.shape(cov_annual) != (len(prices), len(syms), len(syms)):
            raise ValueError("risk_limits needs cov_annual of shape (len(prices), n_instruments, n_instruments)")
        if gross_cap is None or not gross_cap > 0:
            raise ValueError("risk_limits needs a positive gross_cap")
    cur = dict.fromkeys(syms, 0.0)
    out = np.zeros(prices.shape)
    bound = np.zeros(len(prices), dtype=bool)
    n_inst_bound = n_class_bound = n_infeasible = 0
    for i, d in enumerate(prices.index):
        cap = float(capital.loc[d]) if isinstance(capital, pd.Series) else float(capital)
        size_cap = cap if scale is None else cap * float(scale.iat[i])
        new = {}
        for j, s in enumerate(syms):
            f, p, v = fcs.iat[i, j], prices.iat[i, j], vol.iat[i, j]
            if pd.isna(f) or pd.isna(p) or pd.isna(v) or v <= 0:
                new[s] = 0.0
                continue
            x = 1.0 if fxf is None else float(fxf.iat[i, j])
            args = (size_cap, idm, weights.get(s, 0.0), tau, multipliers[s], float(p), x, float(v))
            n = S.target_position(float(f), *args)
            b = S.buffer_width(*args, fraction=buffer_fraction)
            new[s] = S.buffered_trade(cur[s], n, b, long_only=long_only)
        if risk_limits is not None and cap > 0:
            unit_w = {s: multipliers[s] * float(prices.iat[i, j]) * (1.0 if fxf is None else float(fxf.iat[i, j])) / cap
                      for j, s in enumerate(syms)}
            held = [j for j, s in enumerate(syms) if new[s] != 0.0]
            if held:
                names = [syms[j] for j in held]
                k = gross_cap / risk_limits.max_gross
                wts = pd.Series({s: new[s] * unit_w[s] for s in names})
                cov = pd.DataFrame(np.asarray(cov_annual[i])[np.ix_(held, held)], index=names, columns=names)
                with _quiet("firm.risk.limits"):
                    res = _apply_caps(wts / k, cov, risk_limits)
                bound[i] = bool(res.gross_cap_bound)
                n_inst_bound += any(res.instrument_cap_bound.values())
                n_class_bound += any(res.class_cap_bound.values())
                n_infeasible += any(c.startswith("infeasible:") for c in res.breaches_clipped)
                for s in names:
                    new[s] = float(res.weights[s]) * k / unit_w[s]
        elif gross_cap is not None and cap > 0:
            wts = {s: new[s] * multipliers[s] * float(prices.iat[i, j]) * (1.0 if fxf is None else float(fxf.iat[i, j])) / cap
                   for j, s in enumerate(syms)}
            _, bound[i] = S.gross_cap_scale(wts, cap=gross_cap)
            if bound[i]:
                k = gross_cap / sum(abs(w) for w in wts.values())
                new = {s: u * k for s, u in new.items()}
        for j, s in enumerate(syms):
            frac = (multipliers[s] == 1.0) if fractional is None or s not in fractional else fractional[s]
            cur[s] = S.round_position(new[s], frac, multipliers[s], mode=rounding)
            out[i, j] = cur[s]
    if diagnostics is not None:
        diagnostics.update(instrument_cap_bound_days=int(n_inst_bound), class_cap_bound_days=int(n_class_bound),
                           infeasible_cap_days=int(n_infeasible), n_days=len(prices))
    return pd.DataFrame(out, index=prices.index, columns=syms), pd.Series(bound, index=prices.index, name="gross_cap_bound")


def _apply_caps(weights: pd.Series, cov: pd.DataFrame, limits: Any):
    from firm.risk.limits import apply_caps  # lazy: only the P4-03 path needs it

    return apply_caps(weights, cov, limits)


@contextlib.contextmanager
def _quiet(logger_name: str):
    """Raise a logger to ERROR for the block (apply_caps logs a line per binding cap per day; the counts go to ``diagnostics``)."""
    lg = logging.getLogger(logger_name)
    old = lg.level
    lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        lg.setLevel(old)
