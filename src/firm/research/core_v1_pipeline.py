"""Shared numerical pipeline of the core_v1 research run (tickets P3-11 constants and P3-08 G-RESEARCH run).

signals (``firm.signals``) -> per-rule signed forecasts -> combined long/flat forecast -> sizing (``firm.portfolio.sizing`` through
``firm.backtest.vector_engine.targets_from_forecasts``) -> vector engine with the ``firm.costs`` model. Research-only: never imported by
a live module (``tests/test_live_import_isolation.py`` forbids the ``firm.research`` prefix there).

No file is read here except through ``firm.data.etf_loader`` (``firm.research.data_access``, pre-seal, fail-closed); every
parameter default is read from ``config/gates.yaml`` (``robustness_parameters``), never typed here. Choices that the tickets leave open are
module constants below, each documented and surfaced in the reports:

* ``GROUP_WEIGHTS``: the two rule families (EWMAC, breakout) carry equal group weights; within a group the weight is split equally over the
  surviving speeds (charter ``speed_weights_rule: equal_within_group_over_surviving_speeds``). No data input.
* ``INITIAL_CAPITAL`` / ``REFERENCE_TRADE_FRACTION``: a fixed 100,000 USD account (the P2-05 BM2 convention) and a reference trade of 2% of it
  for the speed-filter cost per trade. No data input.
* Engine prices are ``adjusted_close`` (total return), the Sharpe is on engine returns versus zero (``rf=None``, the P2-05 convention), costs use
  the ``etf_alpaca`` schedule (owner decision 2026-10-05) with ADV expressed in adjusted-share units so that ``qty / ADV`` is consistent.
* Robustness re-estimates every pooled scalar on the same window with the perturbed definitions; the speed filter, FDM and IDM are held fixed
  unless the perturbed parameter is one of their own inputs (``speed_cost_max_fraction``, ``fdm_cap``, ``idm_cap``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from firm.backtest import vector_engine as VE
from firm.costs import model as CM
from firm.portfolio import forecast_combine as FC
from firm.portfolio import sizing as SZ
from firm.research import ledger as L
from firm.signals import breakout as BO
from firm.signals import ewmac as EW
from firm.signals import vol as VOL
from firm.validation import bootstrap as BS

log = logging.getLogger(__name__)

ENTRY_GATE_DAYS = 256
INITIAL_CAPITAL = 100_000.0
REFERENCE_TRADE_FRACTION = 0.02
GROUP_WEIGHTS = {"ewmac": 0.5, "breakout": 0.5}
COST_SPEC = "etf_alpaca"
SCALAR_TARGET_ABS = 10.0
FLAT_BPS_PER_SIDE = 5.0        # alt_premia legacy convention (scripts/alt_premia_preregistered_bars.py COSTS["etf_bps_per_side"]); comparison only
CORE_ONLY_BAND_ABS = 0.02      # docs/allocation_forward_test_plan.md: off-schedule trade when the split drifts more than 2 points
NOT_IN_PIPELINE = frozenset({"max_vol_scale", "instrument_risk_cap_multiple"})   # P4-03 limits layer is not wired into this sizing path
_TRADING_DAYS = 256


# ---------------------------------------------------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CoreParams:
    vol_span: int
    vol_blend_long_weight: float
    vol_long_window_days: int
    vol_floor_percentile: float
    ewmac_fast_spans: tuple[int, ...]
    ewmac_slow_to_fast_ratio: int
    forecast_cap: float
    breakout_lookbacks: tuple[int, ...]
    breakout_smoothing_fraction: float
    fdm_cap: float
    idm_cap: float
    buffer_fraction: float
    tau: float
    gross_cap: float
    speed_cost_max_fraction: float


_GATES_NAMES = {
    "vol_span": "vol_span", "vol_blend_long_weight": "vol_blend_long_weight", "vol_long_window_days": "vol_long_window_days",
    "vol_floor_percentile": "vol_floor_percentile", "ewmac_fast_spans": "ewmac_fast_spans",
    "ewmac_slow_to_fast_ratio": "ewmac_slow_to_fast_ratio", "forecast_cap": "forecast_cap", "breakout_lookbacks": "breakout_lookbacks_N",
    "breakout_smoothing_fraction": "breakout_smoothing_fraction_of_N", "fdm_cap": "fdm_cap", "idm_cap": "idm_cap",
    "buffer_fraction": "buffer_fraction", "gross_cap": "gross_cap", "speed_cost_max_fraction": "speed_cost_max_fraction",
}


def params_from_gates(gates: Mapping, tau: float) -> CoreParams:
    """Default parameters straight from ``gates['robustness_parameters']``; ``tau`` comes from the charter (never chosen here)."""
    rp = gates["robustness_parameters"]
    kw: dict[str, Any] = {}
    for field_name, gname in _GATES_NAMES.items():
        v = rp[gname]["value"]
        if v is None:
            raise ValueError(f"gates robustness_parameters.{gname} has no value")
        kw[field_name] = tuple(v) if isinstance(v, list) else v
    return CoreParams(tau=float(tau), **kw)


def with_config(params: CoreParams, config: Mapping, speed_subsets: Mapping) -> tuple[CoreParams, list[str]]:
    """Apply a grid config (forecast cap, buffer fraction, speed subset). Returns the params and the active base rule ids."""
    p = dataclasses.replace(params, forecast_cap=float(config["forecast_cap"]), buffer_fraction=float(config["buffer_fraction"]))
    return p, subset_ids(params, speed_subsets[config["speed_subset"]])


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def perturb(params: CoreParams, name: str, factor: float) -> CoreParams:
    """Perturb one parameter by ``factor`` (0.75 or 1.25). Integers round half up with minimum 2 (gates ``integer_rounding``)."""
    if "[" in name:
        base, idx = name[:-1].split("[")
        field_name = {v: k for k, v in _GATES_NAMES.items()}[base]
        seq = list(getattr(params, field_name))
        seq[int(idx)] = max(2, _half_up(seq[int(idx)] * factor))
        return dataclasses.replace(params, **{field_name: tuple(seq)})
    field_name = {v: k for k, v in _GATES_NAMES.items()}.get(name, name)
    if field_name not in {f.name for f in dataclasses.fields(params)} or field_name in ("ewmac_fast_spans", "breakout_lookbacks"):
        raise KeyError(name)
    cur = getattr(params, field_name)
    new = max(2, _half_up(cur * factor)) if isinstance(cur, int) and not isinstance(cur, bool) else float(cur) * factor
    return dataclasses.replace(params, **{field_name: new})


def perturbable_names(gates: Mapping) -> dict[str, list[str]]:
    """Parameter names to perturb, from ``gates['robustness_parameters']``: ``assessed`` and ``unassessed`` (with the reason implied by
    membership: a null value in gates, or a P4-03 limits-layer parameter that this sizing path does not use)."""
    rp = gates["robustness_parameters"]
    assessed: list[str] = []
    unassessed: list[str] = []
    for name, spec in rp.items():
        if name in NOT_IN_PIPELINE or (spec["value"] is None and name != "tau"):
            unassessed.append(name)
        elif isinstance(spec["value"], list):
            assessed.extend(f"{name}[{i}]" for i in range(len(spec["value"])))
        else:
            assessed.append(name)
    return {"assessed": assessed, "unassessed": unassessed}


# ---------------------------------------------------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RuleDef:
    rule_id: str       # stable id from the BASE parameters (ewmac_8, breakout_80)
    family: str        # "ewmac" | "breakout"
    base_speed: int    # fast span / lookback in the base parameters
    span: int          # fast span / lookback under the parameters in force
    slow: int | None
    smooth: int | None


def rule_defs(base: CoreParams, params: CoreParams) -> list[RuleDef]:
    out = []
    for b, s in zip(base.ewmac_fast_spans, params.ewmac_fast_spans, strict=True):
        out.append(RuleDef(f"ewmac_{b}", "ewmac", b, s, s * params.ewmac_slow_to_fast_ratio, None))
    for b, n in zip(base.breakout_lookbacks, params.breakout_lookbacks, strict=True):
        out.append(RuleDef(f"breakout_{b}", "breakout", b, n, None, max(1, _half_up(n * params.breakout_smoothing_fraction))))
    return out


def subset_ids(base: CoreParams, subset: Mapping) -> list[str]:
    """Base rule ids of a pre-registered speed subset (``{"ewmac_fast_spans": [...], "breakout_lookbacks": [...]}``)."""
    ids = [f"ewmac_{s}" for s in base.ewmac_fast_spans if s in subset["ewmac_fast_spans"]]
    ids += [f"breakout_{n}" for n in base.breakout_lookbacks if n in subset["breakout_lookbacks"]]
    return ids


def family_speeds(rule_ids: Sequence[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"ewmac": [], "breakout": []}
    for r in rule_ids:
        fam, sp = r.rsplit("_", 1)
        out[fam].append(int(sp))
    return out


def rule_weights(survivors: Mapping[str, Sequence[int]], group_weights: Mapping[str, float]) -> dict[str, float]:
    """Group weights split equally over the surviving speeds (``group_equal_weights``). A group with no survivor hands its weight to the
    other groups pro rata; no survivor at all gives no weights (the instrument is flat)."""
    live = {g: w for g, w in group_weights.items() if survivors.get(g)}
    if not live:
        return {}
    tot = sum(live.values())
    gw = {g: w / tot for g, w in live.items()}
    ids = FC.group_equal_weights(gw, {g: [f"{g}_{s}" for s in survivors[g]] for g in gw})
    return ids


# ---------------------------------------------------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Panel:
    symbols: list[str]
    close: pd.DataFrame            # adjusted_close (total return), filled so that the engine never sees a NaN price
    raw_close: pd.DataFrame        # unadjusted close, filled (after-tax model)
    ret: pd.DataFrame              # simple total return; NaN before the first bar of each instrument
    adv: pd.DataFrame              # trailing median dollar volume / adjusted close (shares in adjusted units), lagged one bar, filled
    snapshot_id: str = "synthetic"
    dividends: pd.DataFrame | None = None   # date, symbol, amount (per raw share)
    fx: pd.Series | None = None             # USD/ILS
    source: str = "synthetic"
    extra: dict = field(default_factory=dict)


def panel_from_series(series: Mapping[str, Any], *, snapshot_id: str, dividends: pd.DataFrame | None, fx: pd.Series | None,
                      source: str) -> Panel:
    """Build a Panel from ``{symbol: EtfSeries}`` (``firm.data.etf_loader``). Pre-inception prices are back-filled with the first price
    (zero P&L, flat target) so the engine sees no NaN; the entry gate keeps those dates flat."""
    from firm.data.etf_loader import total_return

    syms = list(series)
    idx = pd.DatetimeIndex(sorted(set().union(*[s.bars.index for s in series.values()])))
    adj = pd.DataFrame({k: series[k].bars["adjusted_close"] for k in syms}).reindex(idx)
    raw = pd.DataFrame({k: series[k].bars["close"] for k in syms}).reindex(idx)
    vol_sh = pd.DataFrame({k: series[k].bars["volume"] for k in syms}).reindex(idx)
    ret = pd.DataFrame({k: total_return(series[k]) for k in syms}).reindex(idx)
    return _finish_panel(syms, adj, raw, vol_sh, ret, snapshot_id, dividends, fx, source)


def _finish_panel(syms, adj, raw, volume, ret, snapshot_id, dividends, fx, source) -> Panel:
    dollar = (volume * raw).rolling(20, min_periods=5).median().shift(1)
    adv = (dollar / adj).bfill().ffill()
    close = adj.ffill().bfill()
    rawc = raw.ffill().bfill()
    return Panel(symbols=list(syms), close=close, raw_close=rawc, ret=ret, adv=adv, snapshot_id=snapshot_id, dividends=dividends, fx=fx,
                 source=source)


def synthetic_panel(seed: int = 0, n_days: int = 1500, symbols: Sequence[str] = ("AAA", "BBB", "CCC", "DDD"),
                    start: str = "2010-01-04") -> Panel:
    """Random trending/mean-reverting regimes, consistent adjusted/raw prices and quarterly dividends. For tests and dry runs only."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    k = len(symbols)
    regime = np.repeat(rng.choice([-1.0, 1.0], size=n_days // 120 + 1), 120)[:n_days]
    common = rng.normal(0, 0.006, n_days)
    rets = np.empty((n_days, k))
    for j in range(k):
        drift = (0.0004 * regime) * (1.0 + 0.3 * j / k) + 0.0001
        rets[:, j] = drift + common * (0.6 + 0.2 * j) + rng.normal(0, 0.007 + 0.002 * j, n_days)
    rets[0] = np.nan
    ret = pd.DataFrame(rets, index=idx, columns=list(symbols))
    adj = 100.0 * (1.0 + ret.fillna(0.0)).cumprod() * (1 + 0.1 * np.arange(k))
    div_rows, f = [], pd.DataFrame(1.0, index=idx, columns=list(symbols))
    raw = adj.copy()
    ex = idx[(idx.month % 3 == 0) & (idx.day >= 15) & (idx.day <= 19)]
    ex = ex[~pd.Series(ex.to_period("M")).duplicated().to_numpy()]
    for s in symbols:
        yld = 0.004
        for d in ex:
            if d == idx[0]:
                continue
            div_rows.append({"date": d, "symbol": s, "amount": float(raw.loc[d, s] * yld)})
    divs = pd.DataFrame(div_rows)
    # adjusted = raw * f, f falls by (1 - yld) at each ex-date going backwards in time
    for s in symbols:
        fac = pd.Series(1.0, index=idx)
        for d in ex:
            fac.loc[idx < d] *= (1.0 - 0.004)
        f[s] = fac
        raw[s] = adj[s] / fac
    vol_sh = pd.DataFrame(rng.integers(5e5, 2e6, (n_days, k)).astype(float), index=idx, columns=list(symbols))
    fx = pd.Series(3.5 * np.cumprod(1 + rng.normal(0, 0.002, n_days)), index=idx)
    p = _finish_panel(list(symbols), adj, raw, vol_sh, ret, "synthetic", divs, fx, "synthetic")
    return p


# ---------------------------------------------------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------------------------------------------------
def vol_annual(panel: Panel, params: CoreParams) -> pd.DataFrame:
    """Blended, floored EWMA vol (annualised, fraction) per instrument; NaN until the 256-return entry gate."""
    cols = {s: VOL.ewma_vol(panel.ret[s], span=params.vol_span, blend_long_weight=params.vol_blend_long_weight,
                            long_window_days=params.vol_long_window_days, floor_percentile=params.vol_floor_percentile,
                            min_obs=ENTRY_GATE_DAYS) for s in panel.symbols}
    return pd.DataFrame(cols).reindex(panel.close.index)


def raw_forecasts(panel: Panel, params: CoreParams, defs: Sequence[RuleDef], vol: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Uncapped, unfloored signed raw forecast per rule id: DataFrame (date x instrument), NaN before the entry gate."""
    v = vol_annual(panel, params) if vol is None else vol
    out: dict[str, pd.DataFrame] = {}
    for d in defs:
        cols = {}
        for s in panel.symbols:
            price = panel.close[s]
            if d.family == "ewmac":
                sig = EW.ewmac_raw(price, VOL.price_unit_vol(price, v[s]), d.span, d.slow)
            else:
                sig = BO.breakout_raw(price, d.span, d.smooth)
            cols[s] = sig.where(v[s].notna())
        out[d.rule_id] = pd.DataFrame(cols)
    return out


def estimate_pooled_scalars(raw_by_rule: Mapping[str, Mapping[str, pd.Series]], window: tuple[str, str],
                            max_research_date=None) -> dict[str, float]:
    """One pooled scalar per rule (never per instrument) so that mean |raw * scalar| = 10 on the UNCAPPED, UNFLOORED signed raw forecast."""
    return {rule: float(EW.estimate_pooled_scalar(dict(by_inst), SCALAR_TARGET_ABS, window[0], window[1], max_research_date))
            for rule, by_inst in raw_by_rule.items()}


def scalar_diagnostics(raw: Mapping[str, pd.DataFrame], scalars: Mapping[str, float], window: tuple[str, str], cap: float) -> dict[str, dict]:
    """Per rule: the pooled mean |raw*scalar| uncapped (= 10) and after the cap (reported separately)."""
    out = {}
    for r, df in raw.items():
        v = df.loc[window[0]:window[1]].stack().dropna().abs().to_numpy() * scalars[r]
        out[r] = {"mean_abs_uncapped": float(v.mean()), "mean_abs_capped": float(np.minimum(v, cap).mean()), "n_obs": int(len(v))}
    return out


def signed_forecasts(raw: Mapping[str, pd.DataFrame], scalars: Mapping[str, float], params: CoreParams) -> dict[str, pd.DataFrame]:
    """Per instrument a DataFrame (date x rule id) of SIGNED capped scaled forecasts (rules are not floored; only the combination is)."""
    syms = next(iter(raw.values())).columns
    return {s: pd.DataFrame({r: (raw[r][s] * scalars[r]).clip(-params.forecast_cap, params.forecast_cap) for r in raw}) for s in syms}


# ---------------------------------------------------------------------------------------------------------------------
# cost per trade and speed filter
# ---------------------------------------------------------------------------------------------------------------------
def cost_per_trade_fraction(spec: CM.InstrumentCostSpec, *, price: float, adv_units: float, vol_daily: float, notional: float,
                            cfg: dict | None = None) -> float:
    """Average one-way cost of a reference trade as a FRACTION of its notional (buy and sell averaged), from ``firm.costs``."""
    qty = notional / price
    buy = CM.cost(spec, qty, price, adv_units, None, vol_daily, cfg=cfg).total
    sell = CM.cost(spec, -qty, price, adv_units, None, vol_daily, cfg=cfg).total
    return float((buy + sell) / 2.0 / notional)


def apply_cost_speed_filter(turnovers: Mapping[str, Mapping[str, float]], cost_per_trade: Mapping[str, float],
                            sigma_pct: Mapping[str, float], expected_rule_sharpe: float,
                            max_cost_fraction: float = EW.DEFAULT_MAX_COST_FRACTION) -> dict[str, dict[str, list[int]]]:
    """Per instrument, keep a speed iff turnover * cost_per_trade / sigma_pct <= max_cost_fraction * expected_rule_sharpe.

    ``turnovers[instrument][rule_id]``. Uses cost and turnover only; there is no return/P&L argument. Output per instrument:
    ``{"ewmac": [surviving fast spans], "breakout": [surviving lookbacks]}``.
    """
    cands = {}
    for inst, by_rule in turnovers.items():
        cands[inst] = {r: {"turnover": t, "cost_per_trade": cost_per_trade[inst], "sigma_pct": sigma_pct[inst]} for r, t in by_rule.items()}
    kept = EW.select_speeds(cands, expected_rule_sharpe, max_cost_fraction)
    return {inst: family_speeds(rules) for inst, rules in kept.items()}


# ---------------------------------------------------------------------------------------------------------------------
# FDM / IDM
# ---------------------------------------------------------------------------------------------------------------------
def pooled_rho(forecasts: Mapping[str, pd.DataFrame], window: tuple[str, str], symbols: Sequence[str], min_overlap: int = 250) -> pd.DataFrame:
    """Pooled correlation between SIGNED rules: the average of the per-instrument correlation matrices (``pooled_forecast_correlation``),
    over instruments with enough overlap (the others are skipped and counted in the caller's report)."""
    ok = {s: forecasts[s] for s in symbols if (forecasts[s].loc[window[0]:window[1]].notna().all(axis=1).sum() >= min_overlap)}
    if not ok:
        raise ValueError("no instrument has enough joint forecast observations in the window")
    return FC.pooled_forecast_correlation(ok, window[0], window[1], min_overlap=min_overlap)


def fdm_from_rho(weights: np.ndarray, rho: np.ndarray, cap: float) -> float:
    """1 / sqrt(w' rho w) with off-diagonals floored at 0, capped at ``cap``."""
    return FC.fdm(np.asarray(weights, float), np.asarray(rho, float), cap=cap)


def estimate_fdm_rho(forecasts: Mapping[str, pd.DataFrame], window: tuple[str, str], weights: Mapping[str, float],
                     cap: float = FC.FDM_CAP) -> tuple[np.ndarray, float]:
    """(pooled rho between the weighted rules, FDM). ``forecasts[instrument]`` is a DataFrame of signed rule forecasts."""
    rules = list(next(iter(forecasts.values())).columns)
    rho = pooled_rho(forecasts, window, list(forecasts)).loc[rules, rules].to_numpy()
    w = np.array([weights.get(r, 0.0) for r in rules])
    return rho, fdm_from_rho(w, rho, cap)


def estimate_idm_H(subsystem_returns: pd.DataFrame, weights: np.ndarray, window: tuple[str, str], cap: float = SZ.IDM_CAP,
                   min_overlap: int = 250) -> tuple[np.ndarray, float]:
    """(pooled sub-system return correlation H, IDM = 1/sqrt(w' H w) capped). H is floored at 0 off-diagonal inside ``idm``."""
    r = subsystem_returns.loc[window[0]:window[1]] if window[0] else subsystem_returns
    H = r.corr(min_periods=min(min_overlap, max(len(r) // 2, 2))).to_numpy()
    H = np.where(np.isnan(H), 0.0, H)
    np.fill_diagonal(H, 1.0)
    return H, float(SZ.idm(np.asarray(weights, float), H, cap=cap))


# ---------------------------------------------------------------------------------------------------------------------
# constants bundle
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class ConstantsBundle:
    scalars: dict[str, float]
    survivors: dict[str, dict[str, list[int]]]       # instrument -> family -> surviving speeds (base ids)
    rho: pd.DataFrame                                # pooled rule correlation (full rule set)
    idm: float
    instrument_weights: dict[str, float]
    group_weights: dict[str, float]
    window: tuple[str, str]
    fdm_override: dict[str, float] | None = None     # per-instrument FDM when held from constants.json


def fdm_for(constants: ConstantsBundle, symbol: str, active_ids: Sequence[str] | None, params: CoreParams,
            capped: bool = True) -> tuple[dict[str, float], float]:
    """(rule weights, FDM) for one instrument: surviving speeds intersected with the config's speed subset, FDM from the stored rho."""
    surv = constants.survivors[symbol]
    if active_ids is not None:
        keep = family_speeds(list(active_ids))
        surv = {fam: [s for s in surv[fam] if s in keep[fam]] for fam in surv}
    w = rule_weights(surv, constants.group_weights)
    if not w:
        return {}, 1.0
    ids = list(w)
    sub = constants.rho.loc[ids, ids].to_numpy()
    fdm = fdm_from_rho(np.array([w[i] for i in ids]), sub, params.fdm_cap if capped else math.inf)
    return w, fdm


# ---------------------------------------------------------------------------------------------------------------------
# engine run
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class RunResult:
    engine: VE.EngineResult
    gross_cap_bound_share: float
    gross_cap_flags: pd.Series
    targets: pd.DataFrame
    combined: pd.DataFrame
    fdm: dict[str, float]
    weights: dict[str, dict[str, float]]
    params: CoreParams
    trial_id: str | None = None


def make_cost_fn(panel: Panel, vol_daily: pd.DataFrame, *, flat_bps: float | None = None):
    """CostFn for the engine: ``firm.costs.model.cost`` (etf_alpaca) or, for the comparison-only run, flat bps per side."""
    cfg = CM.load_cost_config()
    spec = CM.spec_from_config(COST_SPEC, cfg)
    vol_floor = 1e-4

    def fn(symbol, date, qty_delta, price, adv, vol_pct, multiplier, is_roll, stress):
        if flat_bps is not None:
            c = abs(qty_delta) * price * flat_bps / 1e4 * stress
            return CM.CostBreakdown(0.0, 0.0, c, 0.0, 0.0, c)
        if adv is None or not adv > 0:
            raise ValueError(f"no ADV for {symbol} on {date}")
        v = vol_pct if (vol_pct is not None and vol_pct >= 0) else vol_floor
        return CM.cost(spec, qty_delta, price, adv, None, v, multiplier=stress, is_roll=is_roll, cfg=cfg)

    return fn


def build_targets(panel: Panel, constants: ConstantsBundle, params: CoreParams, subset: Sequence[str] | None, *,
                  raw: Mapping[str, pd.DataFrame] | None = None, vol: pd.DataFrame | None = None,
                  scalars: Mapping[str, float] | None = None, fdm_fixed: Mapping[str, float] | None = None,
                  base: CoreParams | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict, dict, pd.DataFrame]:
    """Unit targets and gross-cap flags for one configuration. Returns (targets, flags, combined forecasts, fdm, weights, vol)."""
    base = base or params
    v = vol_annual(panel, params) if vol is None else vol
    defs = rule_defs(base, params)
    r = raw_forecasts(panel, params, defs, v) if raw is None else raw
    sc = dict(constants.scalars if scalars is None else scalars)
    fcs = signed_forecasts(r, sc, params)
    combined = pd.DataFrame(0.0, index=panel.close.index, columns=panel.symbols)
    fdms, wts = {}, {}
    for s in panel.symbols:
        w, fdm = fdm_for(constants, s, subset, params)
        if fdm_fixed is not None and s in fdm_fixed and w:
            fdm = min(float(fdm_fixed[s]), params.fdm_cap)   # held UNCAPPED value; only the cap may differ (robustness of fdm_cap)
        fdms[s], wts[s] = fdm, w
        if not w:
            combined[s] = np.nan
            continue
        combined[s] = FC.combine_forecasts(fcs[s][list(w)], w, fdm, cap=params.forecast_cap, floor=0.0)
    targets, flags = VE.targets_from_forecasts(
        combined, panel.close, v, weights=constants.instrument_weights, idm=constants.idm, tau=params.tau, capital=INITIAL_CAPITAL,
        multipliers=dict.fromkeys(panel.symbols, 1.0), buffer_fraction=params.buffer_fraction, long_only=True,
        gross_cap=params.gross_cap, rounding="toward_zero")
    return targets, flags, combined, fdms, wts, v


def run_config(panel: Panel, constants: ConstantsBundle, params: CoreParams, *, subset_ids: Sequence[str] | None, ledger_mode: str,
               label: str, stress: float = 1.0, flat_bps: float | None = None, base: CoreParams | None = None,
               scalars: Mapping[str, float] | None = None, fdm_fixed: Mapping[str, float] | None = None,
               family: str = "core_v1_engine", prereg: str | None = None, snapshot_id: str | None = None,
               seed: int | None = None) -> RunResult:
    """One engine run of one configuration at ``stress`` x cost. The engine row is written under ``family`` (a separate family from the
    registered core_v1 trial rows: every engine run is a ledger row, ``run_vector_backtest`` refuses to run unlogged)."""
    targets, flags, combined, fdms, wts, v = build_targets(panel, constants, params, subset_ids, scalars=scalars, fdm_fixed=fdm_fixed, base=base)
    cfg = VE.EngineConfig(initial_capital=INITIAL_CAPITAL, stress_multiplier=float(stress), allow_short=False, fill_lag_bars=1)
    ctx = VE.LedgerContext(family=family, mode="exploratory", data_snapshot_id=snapshot_id or panel.snapshot_id, seed=seed,
                           config={"label": label, "stress": stress, "flat_bps": flat_bps, "outer_mode": ledger_mode, "prereg": prereg})
    res = VE.run_vector_backtest(
        panel.close, targets, dict.fromkeys(panel.symbols, 1.0), make_cost_fn(panel, v / math.sqrt(_TRADING_DAYS), flat_bps=flat_bps), cfg,
        adv=panel.adv, vol_pct=v / math.sqrt(_TRADING_DAYS), gross_cap_bound=flags, ledger_ctx=ctx)
    return RunResult(res, float(flags.mean()), flags, targets, combined, fdms, wts, params, res.meta.get("trial_id"))


def core_only_100_targets(prices: pd.DataFrame, capital: float, band_abs: float = CORE_ONLY_BAND_ABS) -> pd.DataFrame:
    """60/40 SPY/IEF held at 100% of capital: rebalanced to target on the first bar of the series, on the first trading day of each month and on
    any day the SPY weight drifts more than ``band_abs`` from 0.60 (the allocation forward-test rule)."""
    w_t = {"SPY": 0.6, "IEF": 0.4}
    idx = prices.index
    out = np.zeros((len(idx), 2))
    units = np.zeros(2)
    cols = ["SPY", "IEF"]
    first_of_month = pd.Series(idx.to_period("M"), index=idx).ne(pd.Series(idx.to_period("M"), index=idx).shift())
    nav = float(capital)
    for i, d in enumerate(idx):
        px = prices.loc[d, cols].to_numpy(float)
        if i > 0:
            nav = float(units @ px) if units.any() else nav
        wspy = float(units[0] * px[0] / nav) if units.any() and nav > 0 else 0.0
        if i == 0 or bool(first_of_month.iloc[i]) or abs(wspy - w_t["SPY"]) > band_abs:
            units = np.array([w_t[c] * nav / px[j] for j, c in enumerate(cols)])
        out[i] = units
    return pd.DataFrame(out, index=idx, columns=cols)


# ---------------------------------------------------------------------------------------------------------------------
# reference drawdown (charter max_dd_procedure) and tax-model trades
# ---------------------------------------------------------------------------------------------------------------------
def bootstrap_max_dd_p95(returns, path_len: int, draws: int, seed: int, percentile: float = 95.0, block_len: float | None = None,
                         chunk: int = 500) -> float:
    """95th percentile of the max drawdown of stationary-bootstrap paths of ``path_len`` days (Politis-White block length)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    bl = BS.politis_white_block_length(r) if block_len is None else float(block_len)
    seeds = np.random.SeedSequence(seed).spawn(math.ceil(draws / chunk))
    dds = []
    for ci, k in enumerate(range(0, draws, chunk)):
        m = min(chunk, draws - k)
        paths = BS.block_bootstrap_returns(r, bl, m, int(seeds[ci].generate_state(1)[0]), n_periods=path_len)
        nav = np.concatenate([np.ones((m, 1)), np.cumprod(1.0 + paths, axis=1)], axis=1)
        dds.append((1.0 - nav / np.maximum.accumulate(nav, axis=1)).max(axis=1))
    return float(np.percentile(np.concatenate(dds), percentile))


def tax_model_trades(positions_adj: pd.DataFrame, adj: pd.DataFrame, raw: pd.DataFrame, dividends: pd.DataFrame | None,
                     symbols: Sequence[str], cost_usd: pd.DataFrame | None = None) -> pd.DataFrame:
    """Discretionary trades in RAW shares for ``firm.reporting.after_tax.apply_tax``.

    The engine holds positions in adjusted-price units (total-return convention); the tax model holds raw shares, reinvests every dividend at
    the ex-date close and taxes it. Target raw shares = adjusted units * (adjusted / raw close); the model's own reinvestment is subtracted so
    the trade list contains only the engine's real decisions: ``trade_t = target_t - q_{t-1} * (1 + d_t / P_t)``.
    """
    idx = positions_adj.index
    d = pd.DataFrame(0.0, index=idx, columns=list(symbols))
    if dividends is not None and len(dividends):
        dd = dividends.assign(date=pd.to_datetime(dividends["date"])).groupby(["date", "symbol"])["amount"].sum().unstack("symbol")
        d = dd.reindex(idx).reindex(columns=list(symbols)).fillna(0.0)
    rows = []
    for s in symbols:
        target = (positions_adj[s] * adj[s] / raw[s]).to_numpy(float)
        q_prev = 0.0
        for i, t in enumerate(idx):
            drift = q_prev * (1.0 + d[s].iat[i] / raw[s].iat[i])
            qty = target[i] - drift
            if abs(qty) > 1e-9:
                rows.append({"date": t, "symbol": s, "qty": float(qty), "price": float(raw[s].iat[i]),
                             "cost_usd": float(cost_usd[s].iat[i]) if cost_usd is not None else 0.0})
            q_prev = target[i] if abs(qty) > 1e-9 else drift
    return pd.DataFrame(rows, columns=["date", "symbol", "qty", "price", "cost_usd"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def jsonable(o: Any) -> Any:
    """Recursively convert numpy / pandas scalars and frames to plain JSON types (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, pd.DataFrame):
        return {"index": [str(i) for i in o.index], "columns": [str(c) for c in o.columns], "values": jsonable(o.to_numpy())}
    if isinstance(o, pd.Series):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def dump_json(obj: Any, path: Path) -> None:
    Path(path).write_text(json.dumps(jsonable(obj), indent=1, sort_keys=True) + "\n", encoding="utf-8")



# ---------------------------------------------------------------------------------------------------------------------
# ledger logging of every step (config['kind'] carries the row kind enumerated in the frozen module)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class LedgerCtx:
    """How a driver records its steps. ``mode='registered'`` needs ``prereg`` and a clean tree (the real run); tests use ``exploratory``."""

    mode: str = "exploratory"
    prereg: str | None = None
    family: str = "core_v1"
    snapshot_id: str | None = None
    seed: int | None = None
    default_params: dict = field(default_factory=lambda: {"forecast_cap": 20.0, "buffer_fraction": 0.1, "speed_subset": "all"})
    trial_ids: dict[str, str] = field(default_factory=dict)
    fingerprint: str | None = None


def returns_fields(r: pd.Series, periods_per_year: int = 252) -> dict:
    """Ledger fields of a per-period excess return series (same conventions as the vector engine's own record)."""
    x = r.dropna()
    out: dict[str, Any] = {"start": str(x.index[0].date()), "end": str(x.index[-1].date()), "n_obs": len(x),
                           "periods_per_year": periods_per_year,
                           "sharpe_conversion": "per_period_excess_over_zero; annualise by sqrt(periods_per_year)"}
    sd = float(x.std(ddof=1)) if len(x) > 2 else 0.0
    if sd > 0:
        out["net_sharpe"] = float(x.mean() / sd)
    s0 = float(x.std(ddof=0)) if len(x) > 3 else 0.0
    if s0 > 0:
        z = (x - x.mean()) / s0
        out["skew"], out["kurt"] = float((z**3).mean()), float((z**4).mean())
    return out


@contextlib.contextmanager
def logged_trial(ctx: LedgerCtx, kind: str, step: str, key: str, *, params: Mapping | None = None, detail: Mapping | None = None,
                 returns: pd.Series | None = None, extra_fields: Mapping | None = None) -> Iterator[Any]:
    """Record one step as a ledger trial: ``config['kind']`` = ``kind``; mode/prereg from ``ctx``; the pre-registration must cover ``params``.

    Yields the trial handle. ``returns`` may be set on the handle inside the block (``h.returns = ...``) or passed up front. The new row's
    trial id is stored in ``ctx.trial_ids[f"{kind}:{step}:{key}"]`` after the block.
    """
    gp = dict(params or ctx.default_params)
    config = {"kind": kind, "step": step, "key": key, "params": gp, "detail": jsonable(dict(detail or {})),
              "prereg_fingerprint": ctx.fingerprint}
    chash = L.config_hash(json.loads(L.canonical_json(config)))
    prereg = ctx.prereg if ctx.mode == "registered" else None
    with L.run_trial(ctx.family, config, mode=ctx.mode, prereg=prereg, prereg_config=gp) as h:  # type: ignore[arg-type]
        fields: dict[str, Any] = {"seed": ctx.seed}
        if ctx.snapshot_id:
            fields["data_snapshot_id"] = ctx.snapshot_id
        h.set(**{k: v for k, v in fields.items() if v is not None})
        if returns is not None:
            h.returns = returns
        yield h
        if h.returns is not None:
            h.set(**returns_fields(h.returns))
        if extra_fields:
            h.set(**dict(extra_fields))
    tr = L.trials(family=ctx.family)
    hit = tr[tr["config_hash"] == chash]
    ctx.trial_ids[f"{kind}:{step}:{key}"] = str(hit.iloc[-1]["trial_id"]) if len(hit) else ""
