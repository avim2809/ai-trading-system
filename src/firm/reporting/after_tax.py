"""After-tax, after-cost benchmark reporting for an Israeli resident (ticket P2-05).

INFORMATION ONLY. NOT TAX ADVICE and NOT the owner's liability: every rule is a switchable, documented assumption in
``config/tax_il.yaml`` (values tagged by source status; the statutory treatment is unverified). Nothing here is imported
by a live module.

Conventions (fixed ex ante in the config, applied identically to the system and to BM2 because both go through ``_simulate``):

* ``deferred_mtm``: tax on realised real gains accrues daily (annual surtax recomputed on the year-to-date amounts) and a
  deferred-tax liability (DTL = ``rate_real_gain`` x net unrealised real gain, floored at zero after offsetting realised
  losses) is deducted from NAV, so a buy-and-hold NAV is shown net of the tax on its unrealised gain. ``terminal_liquidation``
  (sensitivity only) deducts the DTL on the last day only.
* Real gain per FIFO lot: ``proceeds_ils - basis_ils * (cpi_sell / cpi_buy if inflation_adjust else 1)`` with ILS translation at
  the trade date. Losses offset same-year gains; carried forward only if ``loss_carryforward``.
* Surtax: ``(surtax_rate_1 + surtax_rate_2) * max(0, capital income - threshold)``; both layers are applied to the same
  capital-income base (realised net real gain after loss offsets plus gross dividends). The source plan is ambiguous here: verify.
* Dividends: withholding by fund domicile leaves the fund at source (reinvested net); Israeli tax on the gross dividend is
  ``dividend_rate`` (default ``rate_real_gain``, an assumption) minus the withholding only when ``foreign_tax_credit``.
* ``nav_pre_tax_ils`` is the marked portfolio value (cash + positions) in ILS, i.e. before Israeli tax but AFTER source withholding;
  ``taxes=False`` gives a state with no tax at all (no withholding either).

Prices must be split-adjusted, NON-dividend-adjusted close; dividends come from the dividend table only (never pass
``adjusted_close`` here: that double counts).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from firm.costs.model import cost as _cost
from firm.costs.model import spec_from_config
from firm.validation.bootstrap import politis_white_block_length, stationary_bootstrap_indices

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "tax_il.yaml"
STATES = ("gross", "after_cost", "after_tax")
PPY = 252
DISCLAIMER = "INFORMATION ONLY - not tax advice and not the owner's actual liability."


@dataclass(frozen=True)
class TaxConfig:
    rate_real_gain: float
    surtax_rate_1: float
    surtax_rate_2: float
    surtax_threshold_ils: float
    inflation_adjust: bool
    loss_carryforward: bool
    withholding: dict[str, float]
    assumptions: dict[str, str] = field(default_factory=dict)
    dividend_rate: float | None = None
    foreign_tax_credit: bool = False
    convention: str = "deferred_mtm"


def load_tax_config(path: Path = _DEFAULT_PATH) -> TaxConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    status = raw.get("source_status", {})
    assumptions: dict[str, str] = {}
    withholding: dict[str, float] = {}
    for dom, val in (raw.get("withholding") or {}).items():
        if val is None:
            withholding[dom] = 0.0
            assumptions[f"withholding.{dom}"] = "UNSET (adviser Q10): modelled as 0.0 placeholder, not a tax fact"
        else:
            withholding[dom] = float(val)
            assumptions[f"withholding.{dom}"] = f"{val} [{status.get('withholding', 'unset')}]"
    for key in ("rate_real_gain", "surtax_rate_1", "surtax_rate_2", "surtax_threshold_ils", "inflation_adjust", "loss_carryforward",
                "foreign_tax_credit", "dividend_rate", "fx_translation", "lot_method", "convention"):
        val = raw.get(key)
        assumptions[key] = f"{'UNSET (reuses rate_real_gain)' if val is None else val} [{status.get(key, 'see config')}]"
    return TaxConfig(
        rate_real_gain=float(raw["rate_real_gain"]), surtax_rate_1=float(raw["surtax_rate_1"]), surtax_rate_2=float(raw["surtax_rate_2"]),
        surtax_threshold_ils=float(raw["surtax_threshold_ils"]), inflation_adjust=bool(raw["inflation_adjust"]),
        loss_carryforward=bool(raw["loss_carryforward"]), withholding=withholding, assumptions=assumptions,
        dividend_rate=None if raw.get("dividend_rate") is None else float(raw["dividend_rate"]),
        foreign_tax_credit=bool(raw.get("foreign_tax_credit", False)), convention=str(raw.get("convention", "deferred_mtm")),
    )


def describe_assumptions(cfg: TaxConfig) -> str:
    lines = [DISCLAIMER, "Active assumptions:"] + [f"  {k}: {v}" for k, v in sorted(cfg.assumptions.items())]
    return "\n".join(lines)


@dataclass(frozen=True)
class PerfSummary:
    cagr: float
    vol: float
    sharpe: float
    max_dd: float  # positive drawdown depth


def _asof(s: pd.Series, dates: pd.DatetimeIndex, name: str) -> pd.Series:
    s = s.sort_index()
    out = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    if out.isna().any():
        raise ValueError(f"{name} has no value on or before {out.index[out.isna()][0].date()}")
    return out


def _year_tax(net_gain: float, div_income: float, carry_in: float, cfg: TaxConfig) -> tuple[float, float, float, float]:
    """(tax_gain, surtax, loss_used, carry_out) for one year's realised amounts."""
    if net_gain < 0:
        taxable, used = 0.0, 0.0
        carry_out = carry_in - net_gain if cfg.loss_carryforward else 0.0
    else:
        used = min(carry_in, net_gain) if cfg.loss_carryforward else 0.0
        taxable = net_gain - used
        carry_out = carry_in - used if cfg.loss_carryforward else 0.0
    surtax = (cfg.surtax_rate_1 + cfg.surtax_rate_2) * max(0.0, taxable + div_income - cfg.surtax_threshold_ils)
    return cfg.rate_real_gain * taxable, surtax, used, carry_out


Policy = Callable[[pd.Timestamp, dict[str, float], float, pd.Series], list[dict]]


def _simulate(trades: pd.DataFrame | None, prices: pd.DataFrame, fx: pd.Series, cpi: pd.Series, dividends: pd.DataFrame,
              domicile: dict[str, str], cfg: TaxConfig, *, initial_cash_usd: float, taxes: bool, convention: str,
              policy: Policy | None = None) -> pd.DataFrame:
    prices = prices.sort_index()
    dates = pd.DatetimeIndex(prices.index)
    fx_d = _asof(fx, dates, "fx")
    cpi_d = _asof(cpi, dates, "cpi")
    by_day: dict[pd.Timestamp, list[dict]] = {}
    if trades is not None and len(trades):
        t = trades.copy()
        t["date"] = pd.to_datetime(t["date"])
        for rec in t.sort_values("date", kind="stable").to_dict("records"):
            by_day.setdefault(rec["date"], []).append(rec)
    divs: dict[pd.Timestamp, list[tuple[str, float]]] = {}
    if len(dividends):
        for d, s, a in zip(pd.to_datetime(dividends["date"]), dividends["symbol"], dividends["amount"], strict=True):
            divs.setdefault(d, []).append((s, float(a)))
    div_rate = cfg.rate_real_gain if cfg.dividend_rate is None else cfg.dividend_rate

    lots: dict[str, list[list[float]]] = {}  # sym -> [[qty, basis_ils, cpi_buy], ...] FIFO
    cash = float(initial_cash_usd)
    executed: list[dict] = []
    realised: dict[int, float] = {}
    div_income: dict[int, float] = {}
    div_tax_by_year: dict[int, float] = {}
    years = sorted(set(dates.year))
    for y in years:
        realised[y] = div_income[y] = div_tax_by_year[y] = 0.0
    carry = 0.0
    paid_prior = 0.0  # tax of closed years (gain tax + surtax + dividend tax)
    annual: dict[int, dict[str, float]] = {}
    rows = []

    def qty_now() -> dict[str, float]:
        return {s: sum(lot[0] for lot in ls) for s, ls in lots.items() if ls}

    def buy(sym: str, qty: float, price: float, cost_usd: float, day, fxv: float, cpiv: float) -> None:
        nonlocal cash
        spend = qty * price + cost_usd
        cash -= spend
        lots.setdefault(sym, []).append([qty, spend * fxv, cpiv])

    def sell(sym: str, qty: float, price: float, cost_usd: float, day, fxv: float, cpiv: float) -> None:
        nonlocal cash
        have = sum(lot[0] for lot in lots.get(sym, []))
        if qty > have * (1 + 1e-9) + 1e-9:
            raise ValueError(f"{day.date()}: selling {qty} {sym} but only {have} held (long-only engine)")
        proceeds_usd = qty * price - cost_usd
        cash += proceeds_usd
        left = qty
        gain = 0.0
        while left > 1e-12 and lots.get(sym):
            lot = lots[sym][0]
            take = min(left, lot[0])
            frac = take / lot[0]
            basis = lot[1] * frac * ((cpiv / lot[2]) if cfg.inflation_adjust else 1.0)
            gain += proceeds_usd * fxv * (take / qty) - basis
            lot[0] -= take
            lot[1] *= 1.0 - frac
            if lot[0] <= 1e-12:
                lots[sym].pop(0)
            left -= take
        realised[day.year] += gain

    for k, day in enumerate(dates):
        fxv, cpiv, row = float(fx_d.iloc[k]), float(cpi_d.iloc[k]), prices.iloc[k]
        for sym, amt in divs.get(day, []):
            held = sum(lot[0] for lot in lots.get(sym, []))
            if held <= 0:
                continue
            gross = held * amt
            wh = cfg.withholding[domicile[sym]] if taxes else 0.0
            net = gross * (1.0 - wh)
            if taxes:
                tax = max(0.0, div_rate * gross * fxv - (gross * wh * fxv if cfg.foreign_tax_credit else 0.0))
                div_tax_by_year[day.year] += tax
                div_income[day.year] += gross * fxv
            buy(sym, net / float(row[sym]), float(row[sym]), 0.0, day, fxv, cpiv)  # reinvest net dividend at the close
            cash += net  # the reinvestment spent exactly the net cash received
        todo = list(by_day.get(day, []))
        if policy is not None:
            todo = policy(day, qty_now(), cash, row) + todo
        for rec in todo:
            sym, q = rec["symbol"], float(rec["qty"])
            price = float(rec["price"]) if rec.get("price") is not None and not pd.isna(rec.get("price")) else float(row[sym])
            c = float(rec.get("cost_usd", 0.0) or 0.0)
            (buy if q > 0 else sell)(sym, abs(q), price, c, day, fxv, cpiv)
            executed.append({"date": day, "symbol": sym, "qty": q, "price": price, "cost_usd": c})
        pos_usd = sum(sum(lot[0] for lot in ls) * float(row[s]) for s, ls in lots.items())
        nav_pre = (cash + pos_usd) * fxv
        if taxes:
            ytd = realised[day.year]
            tg, sur, _, _ = _year_tax(ytd, div_income[day.year], carry, cfg)
            accrued = paid_prior + tg + sur + div_tax_by_year[day.year]
            unreal = 0.0
            for s, ls in lots.items():
                for lot in ls:
                    unreal += lot[0] * float(row[s]) * fxv - lot[1] * ((cpiv / lot[2]) if cfg.inflation_adjust else 1.0)
            loss_avail = (carry if cfg.loss_carryforward else 0.0) + max(0.0, -ytd)
            dtl = cfg.rate_real_gain * max(0.0, unreal - loss_avail)
        else:
            accrued = dtl = 0.0
        last = k == len(dates) - 1
        charge_dtl = dtl if (convention == "deferred_mtm" or last) else 0.0
        rows.append((nav_pre, accrued, dtl, nav_pre - accrued - charge_dtl))
        if last or dates[k + 1].year != day.year:  # close the tax year
            tg, sur, used, carry_out = _year_tax(realised[day.year], div_income[day.year], carry, cfg) if taxes else (0.0, 0.0, 0.0, carry)
            dt_ = div_tax_by_year[day.year] if taxes else 0.0
            annual[day.year] = {"net_real_gain": realised[day.year], "loss_used": used, "tax_gain": tg, "surtax": sur,
                                "tax_dividends": dt_, "total": tg + sur + dt_}
            paid_prior += tg + sur + dt_
            carry = carry_out
    out = pd.DataFrame(rows, index=dates, columns=["nav_pre_tax_ils", "tax_accrued_ils", "dtl_ils", "nav_after_tax_ils"])
    out.attrs["annual_tax"] = pd.DataFrame.from_dict(annual, orient="index").rename_axis("year")
    out.attrs["trades"] = pd.DataFrame(executed, columns=["date", "symbol", "qty", "price", "cost_usd"])
    out.attrs["initial_usd"] = float(initial_cash_usd)
    return out


def apply_tax(trades: pd.DataFrame, prices_usd: pd.DataFrame, fx_usdils: pd.Series, cpi: pd.Series, dividends: pd.DataFrame,
              domicile: dict[str, str], cfg: TaxConfig, *, initial_cash_usd: float = 0.0, taxes: bool = True,
              convention: str | None = None) -> pd.DataFrame:
    """Daily NAV in ILS. ``trades``: date, symbol, qty (signed shares), optional price and cost_usd (costs enter before tax).

    Columns: ``nav_pre_tax_ils``, ``tax_accrued_ils``, ``dtl_ils``, ``nav_after_tax_ils``; ``attrs['annual_tax']`` has the
    per-year tax table. ``taxes=False`` returns the no-tax state. Dividends are reinvested at the ex-date close.
    """
    return _simulate(trades, prices_usd, fx_usdils, cpi, dividends, domicile, cfg, initial_cash_usd=initial_cash_usd, taxes=taxes,
                     convention=convention or cfg.convention)


def _rebalance_days(dates: pd.DatetimeIndex, mode: str) -> set[pd.Timestamp]:
    if mode not in ("annual", "monthly"):
        raise ValueError(f"rebalance must be 'annual' or 'monthly', not {mode!r}")
    key = dates.year if mode == "annual" else dates.year * 100 + dates.month
    first = pd.Series(dates, index=dates).groupby(key).first()
    return set(first) | {dates[0]}


def benchmark_bm2(prices_usd: pd.DataFrame, fx: pd.Series, cpi: pd.Series, dividends: pd.DataFrame, cfg: TaxConfig, cost_cfg: dict,
                  *, rebalance: Literal["annual", "monthly"] = "annual", initial_usd: float = 100_000.0,
                  weights: dict[str, float] | None = None, domicile: dict[str, str] | None = None, cost_spec: str = "etf_ibkr",
                  adv_shares: float = 1e12) -> pd.DataFrame:
    """BM2 60/40 SPY/IEF through the SAME engine and tax config as the system. Columns gross / after_cost / after_tax (ILS NAV).

    Costs (``firm.costs.model.cost``: commission, fees, half spread) are charged before tax; market impact is omitted
    (``vol=0`` and a huge ``adv_shares``: a small passive account), a flagged simplification. ``attrs['trades']`` holds the
    after-tax run's executed trades, so the system path (``apply_tax``) can reproduce the after-tax column exactly.
    """
    w = weights or {"SPY": 0.6, "IEF": 0.4}
    dom = domicile or {s: "US" for s in w}
    prices = prices_usd[list(w)].sort_index()
    days = _rebalance_days(pd.DatetimeIndex(prices.index), rebalance)
    spec = spec_from_config(cost_spec, cost_cfg)

    def make_policy(with_costs: bool) -> Policy:
        def policy(day, qty, cash, row):
            if day not in days:
                return []
            nav = cash + sum(qty.get(s, 0.0) * float(row[s]) for s in w)
            deltas = {s: w[s] * nav / float(row[s]) - qty.get(s, 0.0) for s in w}
            out = []
            for s in sorted(deltas, key=lambda s: deltas[s]):  # sells first
                q = deltas[s]
                if abs(q * float(row[s])) < 1e-9:
                    continue
                c = _cost(spec, q, float(row[s]), adv_shares, None, 0.0, cfg=cost_cfg).total if with_costs else 0.0
                out.append({"symbol": s, "qty": q, "price": float(row[s]), "cost_usd": c})
            return out
        return policy

    common = {"prices": prices, "fx": fx, "cpi": cpi, "dividends": dividends, "domicile": dom, "cfg": cfg,
              "initial_cash_usd": initial_usd, "convention": cfg.convention}
    gross = _simulate(None, taxes=False, policy=make_policy(False), **common)
    after_cost = _simulate(None, taxes=False, policy=make_policy(True), **common)
    after_tax = _simulate(None, taxes=True, policy=make_policy(True), **common)
    out = pd.DataFrame({"gross": gross["nav_after_tax_ils"], "after_cost": after_cost["nav_after_tax_ils"],
                        "after_tax": after_tax["nav_after_tax_ils"]})
    out.attrs.update(trades=after_tax.attrs["trades"], initial_usd=initial_usd, annual_tax=after_tax.attrs["annual_tax"],
                     rebalance=rebalance, assumptions=cfg.assumptions)
    return out


def gate_benchmark_bm2(prices_usd, fx, cpi, dividends, cfg, cost_cfg, **kw) -> tuple[str, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Gate-7 benchmark: the rebalancing variant (annual / monthly) with the higher AFTER-TAX Sharpe, plus both variants."""
    both = {m: benchmark_bm2(prices_usd, fx, cpi, dividends, cfg, cost_cfg, rebalance=m, **kw) for m in ("annual", "monthly")}
    sharpe = {m: summarise(df["after_tax"]).sharpe for m, df in both.items()}
    name = max(sharpe, key=sharpe.get)
    log.info("gate benchmark %s (after-tax Sharpe %s)", name, sharpe)
    return name, both[name], both


def _perf_arrays(r: np.ndarray, rf: np.ndarray | float = 0.0) -> PerfSummary:
    n = len(r)
    sd = r.std(ddof=1)
    nav = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    dd = 1.0 - nav / np.maximum.accumulate(nav)
    return PerfSummary(cagr=float(nav[-1] ** (PPY / n) - 1.0), vol=float(sd * math.sqrt(PPY)),
                       sharpe=float(np.mean(r - rf) / sd * math.sqrt(PPY)), max_dd=float(dd.max()))


def summarise(nav: pd.Series, rf: pd.Series | None = None) -> PerfSummary:
    """CAGR on the 252-day convention, annualised vol and Sharpe (``rf`` = daily rate series), max drawdown depth (positive)."""
    r = nav.pct_change().dropna()
    rf_v = 0.0 if rf is None else rf.reindex(r.index).ffill().fillna(0.0).to_numpy()
    return _perf_arrays(r.to_numpy(dtype=float), rf_v)


def _metrics(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``sample`` (m, n, k) of returns -> Sharpe, CAGR, max drawdown, each (m, k)."""
    _m, n, _ = sample.shape
    sharpe = sample.mean(1) / sample.std(1, ddof=1) * math.sqrt(PPY)
    cagr = np.exp(np.log1p(sample).sum(1) * (PPY / n)) - 1.0
    nav = np.cumprod(1.0 + sample, axis=1)
    peak = np.maximum(np.maximum.accumulate(nav, axis=1), 1.0)
    return sharpe, cagr, (1.0 - nav / peak).max(1)


def compare(system: dict[str, pd.Series], bench: dict[str, pd.Series], *, n_boot: int = 10_000, seed: int,
            block: str | float = "politis_white") -> dict:
    """Six PerfSummary (system and bench x gross/after_cost/after_tax) and per state the differences system minus bench in
    Sharpe, CAGR and max drawdown, each with a 95% PAIRED stationary-bootstrap CI.

    One set of Politis-White block indices per replicate is applied to all aligned return columns (system and benchmark, every
    state). The CI is on SR(system) - SR(bench), not the Sharpe of the daily difference (an information ratio). Sharpe is vs 0.
    """
    if set(system) != set(STATES) or set(bench) != set(STATES):
        raise ValueError(f"system and bench must be keyed by {STATES}")
    if not (isinstance(block, str) and block == "politis_white") and not (isinstance(block, (int, float)) and block > 1.0):
        raise ValueError("block must be 'politis_white' or a mean block length > 1 (iid resampling is not allowed)")
    names = [("system", s) for s in STATES] + [("bench", s) for s in STATES]
    aligned = pd.concat({f"{w}/{s}": (system if w == "system" else bench)[s] for w, s in names}, axis=1, join="inner")
    rets = aligned.pct_change().dropna()
    mat = rets.to_numpy(dtype=float)
    n = len(mat)
    bl = politis_white_block_length(mat) if block == "politis_white" else float(block)
    chunk = max(1, min(n_boot, 6_000_000 // (n * mat.shape[1])))
    seeds = np.random.SeedSequence(seed).spawn(math.ceil(n_boot / chunk))
    gaps: dict[str, list[np.ndarray]] = {k: [] for k in ("sharpe", "cagr", "max_dd")}
    for ci, start in enumerate(range(0, n_boot, chunk)):
        m = min(chunk, n_boot - start)
        idx = stationary_bootstrap_indices(n, m, bl, int(seeds[ci].generate_state(1)[0]))  # ONE index set for every column
        sh, cg, dd = _metrics(mat[idx])
        for key, arr in (("sharpe", sh), ("cagr", cg), ("max_dd", dd)):
            gaps[key].append(arr[:, :3] - arr[:, 3:])
    summ = {f"{w}/{s}": _perf_arrays(mat[:, j]) for j, (w, s) in enumerate(names)}
    out: dict = {"system": {s: summ[f"system/{s}"] for s in STATES}, "bench": {s: summ[f"bench/{s}"] for s in STATES},
                 "diff": {}, "seed": seed, "n_boot": n_boot, "block_length": bl, "n_obs": n}
    for si, s in enumerate(STATES):
        out["diff"][s] = {}
        for key in ("sharpe", "cagr", "max_dd"):
            est = getattr(summ[f"system/{s}"], key) - getattr(summ[f"bench/{s}"], key)
            col = np.concatenate(gaps[key])[:, si]
            lo, hi = np.percentile(col, [2.5, 97.5])
            out["diff"][s][key] = {"estimate": float(est), "ci_low": float(lo), "ci_high": float(hi)}
    return out
