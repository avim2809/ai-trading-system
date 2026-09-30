"""Replay the REAL allocator (src/firm/allocation) day by day over history.

This is the reference simulation for the deployed allocation portfolio: the
same Allocator / StaticSleeve / BtcTrendSleeve classes the live engine runs,
with whole-share rounding for ETFs, fractional BTC, the NAV drift band, the
BTC sleeve's weekly review and within-sleeve band, and per-side costs.

Timing (same convention as scripts/run_alt_premia_evaluation.py): the plan on
US trading day t uses only completed data (ETF closes <= t-1, BTC UTC bars
dated < t) and fills at day t's close. Cash earns the T-bill rate.

    python scripts/allocation_replay.py --data <dir> --config config/live_alpaca_allocation.example.yaml \
        --start 2015-02-01 --out docs/allocation_replay_2026_09.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_alt_premia_evaluation as ev  # noqa: E402  (data loading only)
from firm.allocation.allocator import build_allocator  # noqa: E402

log = logging.getLogger(__name__)

COST_BPS = {"etf": 5.0, "crypto": 25.0}


def replay(allocation_cfg: dict, data_dir: Path, start: str, end: str | None = None,
           initial_nav: float = 100_000.0) -> dict:
    inp = ev.load_inputs(data_dir)
    alloc = build_allocator(allocation_cfg)
    etf_syms = [s for s in alloc.symbols() if "/" not in s]
    px = inp.px[etf_syms]
    btc = inp.btc.rename("BTC/USD")
    dates = inp.dates[(inp.dates >= pd.Timestamp(start)) & (inp.dates <= pd.Timestamp(end or ev.prereg.DATA_END))]

    qty: dict[str, float] = {}
    cash = initial_nav
    last_rebalance: dict[str, datetime | None] = {s.name: None for s in alloc.sleeves}
    navs, n_orders, turnover, costs = [], 0, 0.0, 0.0
    prev_date = None
    for t in dates:
        # Accrue cash interest from the previous close.
        if prev_date is not None:
            cash *= 1.0 + float(inp.rf.loc[t])
        close_t = {s: float(px.at[t, s]) for s in etf_syms if np.isfinite(px.at[t, s])}
        btc_hist_t = btc[btc.index < t]                       # completed UTC bars only
        close_t["BTC/USD"] = float(btc[btc.index <= t].iloc[-1])  # fill at the latest BTC mark
        prev_close = {s: float(px[s][px.index < t].iloc[-1]) for s in etf_syms}
        prev_close["BTC/USD"] = float(btc_hist_t.iloc[-1])
        nav_prev = cash + sum(q * prev_close[s] for s, q in qty.items())
        positions = {s: q * prev_close[s] for s, q in qty.items() if q}
        history = {s: px[s][px.index < t].dropna() for s in etf_syms}
        history["BTC/USD"] = btc_hist_t
        asof = datetime.combine(t.date(), time(15, 0))
        plan = alloc.plan(asof, nav_prev, positions, prev_close, history, last_rebalance,
                          quantities=dict(qty))
        # Same rule as LiveTradingEngine._finish: record every rebalanced
        # (due and fully plannable) sleeve, even when it placed no orders.
        for name in plan.rebalanced_sleeves:
            last_rebalance[name] = asof
        for o in plan.orders:
            s, q = o["symbol"], float(o["quantity"])
            fill = close_t[s]
            signed = q if o["side"] == "buy" else -q
            notional = q * fill
            c = notional * COST_BPS["crypto" if "/" in s else "etf"] / 1e4
            qty[s] = qty.get(s, 0.0) + signed
            cash -= signed * fill + c
            n_orders += 1
            turnover += notional
            costs += c
        nav = cash + sum(q * close_t[s] for s, q in qty.items())
        navs.append((t, nav))
        prev_date = t
    nav_s = pd.Series(dict(navs))
    ret = nav_s.pct_change().fillna(nav_s.iloc[0] / initial_nav - 1)
    rf = inp.rf.reindex(ret.index)
    ex = ret - rf
    peak = nav_s.cummax()
    dd = 1 - nav_s / peak
    years = len(ret) / 252
    by_year = {int(y): float((1 + g).prod() - 1) for y, g in ret.groupby(ret.index.year)}
    return {
        "start": str(dates[0].date()), "end": str(dates[-1].date()), "n_days": int(len(ret)),
        "final_nav": float(nav_s.iloc[-1]),
        "cagr": float((nav_s.iloc[-1] / initial_nav) ** (1 / years) - 1),
        "vol": float(ret.std(ddof=1) * math.sqrt(252)),
        "sharpe_ex_tbill": float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)),
        "max_drawdown": float(dd.max()),
        "max_drawdown_date": str(dd.idxmax().date()),
        "worst_month": float(ret.groupby(ret.index.to_period("M")).apply(lambda g: (1 + g).prod() - 1).min()),
        "orders": n_orders, "annual_turnover_x_nav": float(turnover / nav_s.mean() / years),
        "annual_cost_pct_nav": float(costs / nav_s.mean() / years),
        "calendar_year_returns": by_year,
        "final_weights": {s: round(q * close_t[s] / nav_s.iloc[-1], 4) for s, q in qty.items() if q},
        "_nav": nav_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=str(_ROOT / "config" / "live_alpaca_allocation.example.yaml"))
    ap.add_argument("--start", default="2015-02-01")
    ap.add_argument("--out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())["allocation"]
    variants = {
        "deployed_92core_8btc": cfg,
        "core_only_100": {**cfg, "sleeves": [{**cfg["sleeves"][0], "weight": 1.0}]},
    }
    out = {}
    for name, c in variants.items():
        r = replay(c, Path(args.data), args.start)
        r.pop("_nav")
        out[name] = r
        log.warning("%s: CAGR %.1f%% vol %.1f%% Sharpe %.2f maxDD %.1f%% (%s) orders %d turnover %.2fx/yr cost %.3f%%/yr",
                    name, 100 * r["cagr"], 100 * r["vol"], r["sharpe_ex_tbill"], 100 * r["max_drawdown"],
                    r["max_drawdown_date"], r["orders"], r["annual_turnover_x_nav"], 100 * r["annual_cost_pct_nav"])
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
