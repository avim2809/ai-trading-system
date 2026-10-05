"""Owner-authorised pre-seal dry run of the P2-05 after-tax BM2 benchmark (60/40 SPY/IEF, US-listed) vs ``core_only_100``.

  nice -n 10 python scripts/run_p2_05_dry_run.py [--asof 2026-09-30] [--out research/reports]
  python scripts/run_p2_05_dry_run.py --synthetic --out <tmp>     # plumbing check on random data, temp ledger, no real data

INFORMATION ONLY: not tax advice and not the owner's liability. Nothing here is chosen or tuned from results: every convention
(annual rebalance for the gate, deferred-tax MTM, cost basis ``etf_alpaca``, seed, windows) is fixed ex ante in
``config/tax_il.yaml`` / ``config/costs.yaml`` / the P2-05 ticket / OD decisions, and the sensitivities are only reported.

Data is read ONLY through ``firm.data.etf_loader`` (``firm.research.data_access``, pre-seal, fail-closed). The analysis runs once
through ``firm.research.backtest_logged`` as an exploratory, counted trial (family ``p2_05_bm2_dry_run``).

Documented substitutions (all flagged in the report):
* Israeli CPI is not available (``il_macro`` is not in the allow-list): the basis inflation uplift is switched OFF (the config's
  documented sensitivity ``inflation_adjust: false``) and CPI is a flat 1.0. This OVERSTATES the modelled tax.
* USD/ILS comes from the EODHD forex file (the Bank of Israel series is not available).
* ``corporate_actions/dividends`` exists on disk for SPY but not for IEF, so cash dividends are IMPLIED for both from the ratio
  ``adjusted_close/close`` on ex-dates (``D = P_{t-1} (1 - f_{t-1}/f_t)``, ``f = adjusted_close/close``). Noise (< 1e-5) and
  dividends (> 1e-3) are separated by a clean gap; the derivation is validated against the 87 real SPY dividends in the report.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import firm.research  # noqa: F401  (seal / wrapped entry-point check)
from firm.costs.model import load_cost_config
from firm.data.etf_loader import load_dividends, load_etf_universe, usd_ils
from firm.reporting import after_tax as at
from firm.research import seal
from firm.research.ledger import backtest_logged

log = logging.getLogger("run_p2_05_dry_run")

FAMILY = "p2_05_bm2_dry_run"
SEED = 13  # recorded; fixed ex ante (batch id), not chosen from results
N_BOOT = 10_000
DIVIDEND_FLOOR = 1e-4  # implied ex-date yield floor; the observed noise (<1e-5) and dividends (>1e-3) are separated by two decades
WINDOWS = {  # name -> (start, end); "core" aligns with docs/allocation_replay_2026_09.json core_only_100
    "core_window": ("2015-02-02", "2026-09-28"),
    "long_window": ("2005-01-03", "2026-09-29"),  # USD/ILS starts 2005-01-03; SPY's last bar is 2026-09-29
}
_RESULT: dict = {}


def implied_dividends(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Per-share cash dividends implied by the adjusted/raw close ratio on ex-dates (see module docstring)."""
    f = bars["adjusted_close"] / bars["close"]
    prev_close = bars["close"].shift(1)
    d = prev_close * (1.0 - f.shift(1) / f)
    keep = (d / prev_close).abs() > DIVIDEND_FLOOR
    out = pd.DataFrame({"date": bars.index[keep.to_numpy()], "symbol": symbol, "amount": d[keep].to_numpy()})
    return out.reset_index(drop=True)


def load_inputs(asof: dt.date) -> dict:
    """Real inputs, pre-seal, through the typed loader only."""
    s = load_etf_universe(ROOT / "config" / "universe_etf.yaml", asof=asof, include_delisted=False)
    bars = {k: s[k].bars for k in ("SPY", "IEF")}
    prices = pd.concat({k: bars[k]["close"] for k in bars}, axis=1).dropna()
    adj = pd.concat({k: bars[k]["adjusted_close"] for k in bars}, axis=1).reindex(prices.index)
    divs = pd.concat([implied_dividends(bars[k], k) for k in bars], ignore_index=True)
    real = load_dividends(["SPY"], asof=asof)
    return {"prices": prices, "adjusted": adj, "dividends": divs, "fx": usd_ils(asof, source="eodhd"), "real_spy_dividends": real,
            "bars": bars, "source": "eodhd etfs_full via firm.data.etf_loader; fx eodhd forex/USDILS"}


def synthetic_inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-12-01", "2026-09-29")
    px = pd.DataFrame({"SPY": 100 * np.cumprod(1 + rng.normal(4e-4, 0.01, len(idx))),
                       "IEF": 100 * np.cumprod(1 + rng.normal(1e-4, 0.004, len(idx)))}, index=idx)
    q = idx[idx.month.isin([3, 6, 9, 12]) & (idx.day == 15)]
    divs = pd.DataFrame({"date": q, "symbol": "SPY", "amount": 0.5})
    fx = pd.Series(3.5 * np.cumprod(1 + rng.normal(0, 0.003, len(idx))), index=idx)
    return {"prices": px, "adjusted": px, "dividends": divs, "fx": fx, "real_spy_dividends": divs, "bars": {}, "source": "SYNTHETIC"}


def _summ(nav: pd.Series) -> dict:
    return dataclasses.asdict(at.summarise(nav))


def _states(df: pd.DataFrame) -> dict[str, pd.Series]:
    # benchmark_bm2 frames carry DataFrame-valued ``attrs`` that pandas cannot compare inside ``pd.concat``: strip them on the columns
    return {s: pd.Series(df[s].to_numpy(), index=df.index, name=s) for s in at.STATES}


def no_cpi_config(cfg_base: at.TaxConfig) -> at.TaxConfig:
    """No Israeli CPI available: the config's documented sensitivity ``inflation_adjust: false`` (printed in the assumptions)."""
    assumptions = {**cfg_base.assumptions, "inflation_adjust": "False (FORCED: Israeli CPI unavailable; documented sensitivity; overstates tax)"}
    return dataclasses.replace(cfg_base, inflation_adjust=False, assumptions=assumptions)


def analyse(inp: dict, cfg_base: at.TaxConfig, cost_cfg: dict, replay: dict | None) -> dict:
    cfg = no_cpi_config(cfg_base)
    out: dict = {"disclaimer": at.DISCLAIMER, "no_post_seal_data_was_read": True, "seed": SEED, "n_boot": N_BOOT,
                 "data_source": inp["source"], "tax_assumptions_active": at.describe_assumptions(cfg).splitlines(),
                 "substitutions": ["Israeli CPI unavailable: inflation_adjust forced False, CPI flat 1.0 (overstates modelled tax)",
                                   "USD/ILS from EODHD forex (Bank of Israel series not available)",
                                   "dividends implied from adjusted_close/close ratio for SPY and IEF (no IEF dividend file on disk)",
                                   "withholding US/IE unset (adviser Q10): modelled 0.0 placeholder"], "windows": {}}
    prices, divs, fx = inp["prices"], inp["dividends"], inp["fx"]
    cpi = pd.Series(1.0, index=prices.index)
    usd = pd.Series(1.0, index=prices.index)
    for name, (a, b) in WINDOWS.items():
        px = prices.loc[a:b]
        dv = divs[(pd.to_datetime(divs["date"]) >= px.index[0]) & (pd.to_datetime(divs["date"]) <= px.index[-1])]
        w: dict = {"start": px.index[0].date().isoformat(), "end": px.index[-1].date().isoformat(), "n_days": len(px)}
        runs = {
            "annual_alpaca": at.benchmark_bm2(px, fx, cpi, dv, cfg, cost_cfg, rebalance="annual", cost_spec="etf_alpaca"),
            "monthly_alpaca": at.benchmark_bm2(px, fx, cpi, dv, cfg, cost_cfg, rebalance="monthly", cost_spec="etf_alpaca"),
            "annual_ibkr": at.benchmark_bm2(px, fx, cpi, dv, cfg, cost_cfg, rebalance="annual", cost_spec="etf_ibkr"),
            "annual_alpaca_terminal_liquidation": at.benchmark_bm2(
                px, fx, cpi, dv, dataclasses.replace(cfg, convention="terminal_liquidation"), cost_cfg, rebalance="annual",
                cost_spec="etf_alpaca"),
        }
        w["ils_nav_summaries"] = {k: {s: _summ(df[s]) for s in at.STATES} for k, df in runs.items()}
        w["gate_benchmark_variant_by_after_tax_sharpe"] = max(
            ("annual_alpaca", "monthly_alpaca"), key=lambda k: w["ils_nav_summaries"][k]["after_tax"]["sharpe"])
        w["annual_tax_annual_alpaca"] = json.loads(runs["annual_alpaca"].attrs["annual_tax"].round(2).to_json(orient="index"))
        w["n_trades_annual_alpaca"] = len(runs["annual_alpaca"].attrs["trades"])
        w["paired_bootstrap_monthly_minus_annual"] = json.loads(json.dumps(
            at.compare(_states(runs["monthly_alpaca"]), _states(runs["annual_alpaca"]), n_boot=N_BOOT, seed=SEED), default=lambda o: dataclasses.asdict(o)))
        usd_runs = {m: at.benchmark_bm2(px, usd, cpi, dv, cfg, cost_cfg, rebalance=m, cost_spec="etf_alpaca") for m in ("annual", "monthly")}
        w["usd_pre_tax_summaries"] = {m: {s: _summ(df[s]) for s in ("gross", "after_cost")} for m, df in usd_runs.items()}
        adj = inp["adjusted"].loc[a:b].pct_change().dropna()
        daily_rb = (1 + 0.6 * adj["SPY"] + 0.4 * adj["IEF"]).cumprod()
        w["usd_daily_rebalanced_adjusted_close_reference"] = _summ(daily_rb)
        out["windows"][name] = w
    if replay is not None:
        ref = replay["core_only_100"]
        c = out["windows"]["core_window"]
        daily = c["usd_daily_rebalanced_adjusted_close_reference"]["cagr"]
        out["core_only_100_reference"] = {
            "published": {k: ref[k] for k in ("start", "end", "n_days", "cagr", "vol", "sharpe_ex_tbill", "max_drawdown", "orders")},
            "note": ("published summary only (no daily series on disk): core_only_100 is a monthly drift-band rebalanced 60/40 SPY/IEF; its "
                     "Sharpe is ex-T-bill, ours is vs zero, so Sharpe is not comparable; CAGR/vol/maxDD are"),
            "bm2_usd_pre_tax_core_window": c["usd_pre_tax_summaries"],
            "daily_rebalanced_cagr": daily, "abs_cagr_gap_to_published_pp": abs(daily - ref["cagr"]) * 100,
            "real_data_sanity_test_equivalent_passes_within_1pp": bool(abs(daily - ref["cagr"]) < 0.01)}
    dv = inp["real_spy_dividends"]
    imp = divs[divs["symbol"] == "SPY"]
    real = dv.assign(date=pd.to_datetime(dv["date"])).set_index("date")["amount"]
    impl = imp.assign(date=pd.to_datetime(imp["date"])).set_index("date")["amount"]
    both = pd.concat({"real": real, "implied": impl}, axis=1, sort=True).dropna()
    rel = ((both["implied"] / both["real"]) - 1).abs() if len(both) else pd.Series(dtype=float)
    out["implied_dividend_validation_spy"] = {"n_real": len(real), "n_implied_total": len(impl), "n_matched_dates": len(both),
                                              "median_rel_err": float(rel.median()) if len(rel) else None,
                                              "max_rel_err": float(rel.max()) if len(rel) else None,
                                              "implied_n_ief": int((divs["symbol"] == "IEF").sum())}
    return out


def _ci(d: dict) -> str:
    return f"{d['estimate']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"


def to_markdown(r: dict) -> str:
    L = ["# P2-05 after-tax BM2 dry run (60/40 SPY/IEF, US-listed)", "", f"**{r['disclaimer']}**", "",
         (f"No post-seal data was read (asof 2026-09-30, pre-seal, through `firm.research.data_access`). Exploratory ledger trial "
          f"family `{FAMILY}` (run once, counted). Seed {r['seed']}, {r['n_boot']} paired stationary-bootstrap draws (Politis-White)."), "",
         "## Substitutions and assumptions (read first)", "", *[f"- {s}" for s in r["substitutions"]], "",
         "Tax assumptions printed by the model:", "", "```", *r["tax_assumptions_active"], "```", ""]
    for name, w in r["windows"].items():
        L += [f"## Window {name}: {w['start']}..{w['end']} ({w['n_days']} days); ILS NAV", "",
              f"Gate variant by higher after-tax Sharpe: **{w['gate_benchmark_variant_by_after_tax_sharpe']}**", "",
              "| variant | state | CAGR | vol | Sharpe (vs 0) | maxDD |", "|---|---|---|---|---|---|"]
        for k, st in w["ils_nav_summaries"].items():
            for s, m in st.items():
                L.append(f"| {k} | {s} | {m['cagr']:.2%} | {m['vol']:.2%} | {m['sharpe']:.3f} | {m['max_dd']:.2%} |")
        L += ["", "USD pre-tax BM2 (no FX, no tax):", "", "| rebalance | state | CAGR | vol | Sharpe (vs 0) | maxDD |", "|---|---|---|---|---|---|"]
        for k, st in w["usd_pre_tax_summaries"].items():
            for s, m in st.items():
                L.append(f"| {k} | {s} | {m['cagr']:.2%} | {m['vol']:.2%} | {m['sharpe']:.3f} | {m['max_dd']:.2%} |")
        d = w["usd_daily_rebalanced_adjusted_close_reference"]
        L += ["", (f"Daily-rebalanced adjusted_close total-return reference (USD): CAGR {d['cagr']:.2%}, vol {d['vol']:.2%}, "
               f"Sharpe {d['sharpe']:.3f}, maxDD {d['max_dd']:.2%}."), "",
              (f"Paired bootstrap, monthly (as 'system') minus annual (gate bench), ILS NAV, block length "
               f"{w['paired_bootstrap_monthly_minus_annual']['block_length']:.1f}:"), "",
              "| state | dSharpe [95% CI] | dCAGR [95% CI] | dMaxDD [95% CI] |", "|---|---|---|---|"]
        for s, dd in w["paired_bootstrap_monthly_minus_annual"]["diff"].items():
            L.append(f"| {s} | {_ci(dd['sharpe'])} | {_ci(dd['cagr'])} | {_ci(dd['max_dd'])} |")
        L += ["", f"Annual-rebalance trades: {w['n_trades_annual_alpaca']}. Modelled tax per year (ILS, annual gate run):", "",
              "| year | net real gain | tax gain | surtax | tax dividends | total |", "|---|---|---|---|---|---|"]
        for y, t in w["annual_tax_annual_alpaca"].items():
            L.append(f"| {y} | {t['net_real_gain']:,.0f} | {t['tax_gain']:,.0f} | {t['surtax']:,.0f} | {t['tax_dividends']:,.0f} | {t['total']:,.0f} |")
        L.append("")
    c = r.get("core_only_100_reference")
    if c:
        p = c["published"]
        L += ["## Versus `core_only_100` (docs/allocation_replay_2026_09.json, published summary)", "", c["note"] + ".", "",
              (f"core_only_100 ({p['start']}..{p['end']}, {p['n_days']} days): CAGR {p['cagr']:.2%}, vol {p['vol']:.2%}, Sharpe ex-T-bill "
               f"{p['sharpe_ex_tbill']:.3f}, maxDD {p['max_drawdown']:.2%}, {p['orders']} orders."), "",
              (f"Daily-rebalanced SPY/IEF adjusted-close reference CAGR {c['daily_rebalanced_cagr']:.2%}; gap to published "
               f"{c['abs_cagr_gap_to_published_pp']:.2f} pp; within the 1 pp bound of the opt-in `real_data` sanity test: "
               f"**{c['real_data_sanity_test_equivalent_passes_within_1pp']}** (computed here, equivalent to that test)."), ""]
    v = r["implied_dividend_validation_spy"]
    L += ["## Implied-dividend validation (SPY real dividend file vs implied)", "",
          (f"real {v['n_real']}, implied (SPY, all history) {v['n_implied_total']}, matched on date {v['n_matched_dates']}; "
           f"median relative error {v['median_rel_err']:.2e}, max {v['max_rel_err']:.2e}; implied IEF events {v['implied_n_ief']}."), ""]
    return "\n".join(L)


@backtest_logged(FAMILY, mode="exploratory", fixed_args=("asof", "start_end_windows", "cost_basis", "seed", "n_boot", "synthetic"))
def p2_05_bm2_dry_run(asof: str, start_end_windows: dict, cost_basis: str, seed: int, n_boot: int, synthetic: bool) -> pd.Series:
    """Logged unit of analysis. Returns the gate BM2 (annual, etf_alpaca) daily after-tax ILS return series, core window."""
    cfg = at.load_tax_config(ROOT / "config" / "tax_il.yaml")
    cost_cfg = load_cost_config()
    inp = synthetic_inputs() if synthetic else load_inputs(dt.date.fromisoformat(asof))
    replay = None if synthetic else json.loads((ROOT / "docs" / "allocation_replay_2026_09.json").read_text(encoding="utf-8"))
    res = analyse(inp, cfg, cost_cfg, replay)
    _RESULT.update(res)
    a, b = WINDOWS["core_window"]
    px = inp["prices"].loc[a:b]
    cfg2 = no_cpi_config(cfg)
    dv = inp["dividends"]
    dv = dv[(pd.to_datetime(dv["date"]) >= px.index[0]) & (pd.to_datetime(dv["date"]) <= px.index[-1])]
    nav = at.benchmark_bm2(px, inp["fx"], pd.Series(1.0, index=px.index), dv, cfg2, cost_cfg, rebalance="annual", cost_spec="etf_alpaca")
    ret = nav["after_tax"].pct_change().dropna()
    return pd.Series(ret.to_numpy(), index=ret.index, name="bm2_annual_after_tax_ils")  # plain Series: frame attrs break parquet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", type=dt.date.fromisoformat, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "reports")
    ap.add_argument("--synthetic", action="store_true", help="random fixture + temp ledger (plumbing check only)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    asof = a.asof or seal.max_research_date()
    seal.check_asof(asof, what="run_p2_05_dry_run")
    if a.synthetic:
        os.environ.setdefault("FIRM_RESEARCH_LEDGER_ROOT", tempfile.mkdtemp(prefix="p205_synth_ledger_"))
        os.makedirs(os.path.join(os.environ["FIRM_RESEARCH_LEDGER_ROOT"], "returns"), exist_ok=True)
    p2_05_bm2_dry_run(asof.isoformat(), WINDOWS, "etf_alpaca", SEED, N_BOOT, bool(a.synthetic))
    a.out.mkdir(parents=True, exist_ok=True)
    stem = "p2_05_bm2_dry_run" + ("_SYNTHETIC" if a.synthetic else "")
    (a.out / f"{stem}.json").write_text(json.dumps(_RESULT, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (a.out / f"{stem}.md").write_text(to_markdown(_RESULT), encoding="utf-8")
    log.info("report written to %s", a.out / f"{stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
