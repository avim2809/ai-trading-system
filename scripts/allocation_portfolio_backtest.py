"""Historical sanity backtest of the proposed 92/8 core+satellite allocation portfolio.

This is NOT a new edge claim and is not pre-registered as one: the 60/40 core and
the C1 BTC-trend satellite were already evaluated (and fingerprinted) in
``scripts/alt_premia_preregistered_bars.py`` / ``docs/edge_search_verdict_2026_09.md``.
This script only asks "what would the exact deployed portfolio have returned",
reusing that frozen rule set and the same cached data and cost model, for
sizing a kill-switch and writing up the forward-test power calculation
(``scripts/allocation_forward_test_preregistered.py``).

Portfolio being deployed to one paper instance (Alpaca, $100k):
- core: 92% of NAV in 60% SPY / 40% IEF, rebalanced monthly (close of the first
  US trading day of the month) plus a 2%-absolute daily drift band;
- satellite: 8% of NAV in the pre-registered BTC 4-week trend rule (C1 in
  ``alt_premia_preregistered_bars.CANDIDATES["C1_btc_trend"]``): long BTC if the
  28-day return > 0 at the Sunday UTC close, sized min(1, 0.40/vol63 x sqrt(365))
  *within the sleeve*, band 0.10, 1-day lag, 25 bps/side.

Capital treatment: the two sleeves are modelled as independent capital (the
same "sleeved" convention already live on Alpaca, see docs/capital_sleeves_plan.md)
-- 92% and 8% of NAV are fixed at inception and each sleeve compounds on its
own returns; they are never rebalanced against each other. This is a modelling
choice (documented, not hidden): over 11+ years a large BTC rally would let the
satellite's *actual* share of NAV drift above 8%, which the historical results
below make visible (see ``satellite_actual_share_end``).

Usage::

    python scripts/allocation_portfolio_backtest.py --data <dir> \
        --report docs/allocation_portfolio_backtest_2026_09.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import alt_premia_preregistered_bars as prereg  # noqa: E402
import run_alt_premia_evaluation as ape  # noqa: E402

log = logging.getLogger(__name__)

WINDOW_START = prereg.CANDIDATES["C1_btc_trend"]["eval_start"]  # "2015-02-01": BTC data availability
WINDOW_END = prereg.DATA_END  # "2026-09-28"
CORE_WEIGHT = 0.92
SATELLITE_WEIGHT = 0.08
CORE_TARGETS = {"SPY": 0.6, "IEF": 0.4}
CORE_DRIFT_BAND_ABS = 0.02
KILL_SWITCH_LIVE = 0.08  # config/live.yaml & config/live_alpaca.yaml: kill_switch_drawdown

LEGACY_RETURNS_PARQUET = Path(
    "/tmp/claude-0/-local-store-git-ai-trading-system/"
    "c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/runs/step1/C0_legacy_optimal.returns.parquet"
)


# ---------------------------------------------------------------------------
# Core sleeve: 60/40 SPY/IEF, monthly + 2%-absolute daily drift band
# ---------------------------------------------------------------------------

def core_6040_drift_band(inp: "ape.Inputs", stress: bool = False) -> pd.Series:
    """60/40 SPY/IEF, target-weight rebalance at the close of the first US
    trading day of each month, plus a same-day check any other day: if either
    sleeve-internal weight has drifted more than ``CORE_DRIFT_BAND_ABS`` from
    its 0.6/0.4 target, rebalance immediately. Same cost model as the 60/40
    benchmark (``etf_bps_per_side``, no leverage)."""
    r = inp.px[["SPY", "IEF"]].pct_change().to_numpy()
    target = np.array([CORE_TARGETS["SPY"], CORE_TARGETS["IEF"]])
    me = ape.month_end_flags(inp.dates)
    ok = inp.px[["SPY", "IEF"]].notna().all(axis=1).to_numpy()

    def decide(i: int, w: np.ndarray):
        if i >= 1 and me[i - 1] and ok[i - 1]:
            return target
        if ok[i] and np.any(np.abs(w - target) > CORE_DRIFT_BAND_ABS):
            return target
        return None

    net, held = ape.simulate(
        r, inp.rf.to_numpy(), decide,
        np.full(2, prereg.COSTS["etf_bps_per_side"] * ape._cost_mult(stress)),
    )
    s = pd.Series(net, inp.dates, name="core_6040_band")
    s.attrs["held"] = held
    return s


def core_rebalance_events(inp: "ape.Inputs") -> dict:
    """Count how often the drift band, not the monthly clock, forces a trade."""
    r = inp.px[["SPY", "IEF"]].pct_change().to_numpy()
    target = np.array([CORE_TARGETS["SPY"], CORE_TARGETS["IEF"]])
    me = ape.month_end_flags(inp.dates)
    ok = inp.px[["SPY", "IEF"]].notna().all(axis=1).to_numpy()
    monthly, band = 0, 0

    def decide(i: int, w: np.ndarray):
        nonlocal monthly, band
        if i >= 1 and me[i - 1] and ok[i - 1]:
            monthly += 1
            return target
        if ok[i] and np.any(np.abs(w - target) > CORE_DRIFT_BAND_ABS):
            band += 1
            return target
        return None

    ape.simulate(r, inp.rf.to_numpy(), decide, np.full(2, prereg.COSTS["etf_bps_per_side"]))
    return {"monthly_trades": monthly, "drift_band_trades": band}


# ---------------------------------------------------------------------------
# Sleeved (non-cross-rebalanced) combination
# ---------------------------------------------------------------------------

def sleeved_portfolio(core: pd.Series, satellite: pd.Series, core_w: float, sat_w: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Combine two independently-compounding capital sleeves (fixed split at
    inception, never rebalanced against each other -- matches the live
    ``capital_allocation_mode: sleeved`` convention). Returns (portfolio daily
    return, core NAV share of total, satellite NAV share of total)."""
    core_growth = np.cumprod(1.0 + core.to_numpy())
    sat_growth = np.cumprod(1.0 + satellite.to_numpy())
    total = core_w * core_growth + sat_w * sat_growth
    prev = np.r_[1.0, total[:-1]]
    port_ret = pd.Series(total / prev - 1.0, core.index, name="portfolio")
    core_share = pd.Series(core_w * core_growth / total, core.index, name="core_share")
    sat_share = pd.Series(sat_w * sat_growth / total, core.index, name="satellite_share")
    return port_ret, core_share, sat_share


def sleeved_portfolio_periodic_reset(core: pd.Series, satellite: pd.Series, core_w: float, sat_w: float,
                                      reset: np.ndarray) -> pd.Series:
    """Sensitivity variant: like ``sleeved_portfolio``, but the 92/8 split
    between sleeves is also reset back to target (not just each sleeve's own
    internal rule) on days where ``reset[i]`` is True. The pre-registered
    portfolio definition does not specify a cross-sleeve rebalance, so the
    headline result uses ``sleeved_portfolio`` (no reset, matches the live
    ``capital_allocation_mode: sleeved`` convention); this is reported alongside
    it because letting the sleeves run un-rebalanced for a decade lets a small
    satellite dominate NAV (see ``satellite_actual_share`` in the report)."""
    core_r, sat_r = core.to_numpy(), satellite.to_numpy()
    n = len(core_r)
    total = np.empty(n)
    core_val, sat_val = core_w, sat_w
    for i in range(n):
        core_val *= 1.0 + core_r[i]
        sat_val *= 1.0 + sat_r[i]
        t = core_val + sat_val
        total[i] = t
        if reset[i]:
            core_val, sat_val = core_w * t, sat_w * t
    prev = np.r_[1.0, total[:-1]]
    return pd.Series(total / prev - 1.0, core.index, name="portfolio_monthly_sleeve_reset")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _nav(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def _drawdown_series(r: pd.Series) -> pd.Series:
    nav = _nav(r)
    peak = nav.cummax()
    return 1 - nav / peak  # positive number, e.g. 0.08 = 8% drawdown


def _year_max_dd(dd: pd.Series, year: int) -> float:
    sub = dd[dd.index.year == year]
    return float(sub.max()) if len(sub) else float("nan")


def _worst_month(r: pd.Series) -> tuple[str, float]:
    m = (1 + r).resample("ME").prod() - 1
    idx = m.idxmin()
    return str(idx.date()), float(m.loc[idx])


def _drawdown_episodes(r: pd.Series, min_depth: float = 0.03) -> list[dict]:
    eps = ape.worst_episodes(r.to_numpy(), k=10_000)
    out = []
    for p, t, depth in eps:
        if -depth < min_depth:
            continue
        out.append({
            "peak": str(r.index[p].date()), "trough": str(r.index[t].date()),
            "depth": float(depth), "days_to_trough": int(t - p),
        })
    return sorted(out, key=lambda e: e["depth"])


def metrics(r: pd.Series, rf: pd.Series) -> dict:
    j = pd.concat([r, rf], axis=1, keys=["r", "rf"]).dropna()
    ex = (j.r - j.rf).to_numpy()
    n_years = len(j) / prereg.TRADING_DAYS
    cagr = float(np.prod(1 + j.r.to_numpy()) ** (prereg.TRADING_DAYS / len(j)) - 1)
    vol = float(j.r.std(ddof=1) * ape.ANN)
    sh = ape.sharpe(ex)
    dd = _drawdown_series(j.r)
    worst_month_date, worst_month_ret = _worst_month(j.r)
    return {
        "n_days": int(len(j)), "start": str(j.index[0].date()), "end": str(j.index[-1].date()),
        "total_return": float(np.prod(1 + j.r.to_numpy()) - 1),
        "cagr": cagr, "annualized_vol": vol, "sharpe_above_tbills": sh,
        "max_drawdown": float(dd.max()),
        "worst_month": {"month": worst_month_date, "return": worst_month_ret},
        "drawdown_2020": _year_max_dd(dd, 2020),
        "drawdown_2022": _year_max_dd(dd, 2022),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    inp = ape.load_inputs(Path(args.data))
    rf = inp.rf

    log.info("building core (60/40 + 2%% drift band) and satellite (C1 BTC trend, unchanged) sleeves")
    # Compute each sleeve's full-history return series first (the drift-band and
    # C1 review logic need continuous history to decide correctly), THEN restrict
    # to the deployment window before combining capital -- sleeved capital must be
    # rebased at the portfolio's actual inception (2015-02, BTC availability), not
    # at whatever date the underlying SPY/IEF/BTC data happens to begin. Combining
    # over the full history first would fix the 92/8 split decades before the
    # window starts and report a bogus, already-drifted starting share.
    core_full = core_6040_drift_band(inp)
    core_events = core_rebalance_events(inp)
    satellite_full, _ = ape.c1(inp)  # exact pre-registered rule, unmodified
    bm2_full = ape.bm2(inp)          # (b) 60/40 alone, 100% of NAV
    bm1_full = ape.bm1(inp)          # (c) SPY alone

    core_w = ape.window(core_full, WINDOW_START, WINDOW_END)
    satellite_w = ape.window(satellite_full, WINDOW_START, WINDOW_END)
    rf_w = ape.window(rf, WINDOW_START, WINDOW_END)
    bm2_w = ape.window(bm2_full, WINDOW_START, WINDOW_END)
    bm1_w = ape.window(bm1_full, WINDOW_START, WINDOW_END)

    port_ret, core_share, sat_share = sleeved_portfolio(core_w, satellite_w, CORE_WEIGHT, SATELLITE_WEIGHT)
    # Isolate the satellite's marginal contribution: same core allocation (92%),
    # 8% left in cash instead of BTC. Same sleeved-capital combinator, cash sleeve
    # earns the SPY-calendar T-bill rate.
    core_only_92pct_ret, _, _ = sleeved_portfolio(core_w, rf_w.rename("cash"), CORE_WEIGHT, SATELLITE_WEIGHT)
    core_only_92pct_ret.name = "core_92pct_plus_cash"

    series = {
        "portfolio_92_8": port_ret,
        "bm2_60_40_100pct": bm2_w,
        "bm1_spy_100pct": bm1_w,
        "core_92pct_plus_8pct_cash": core_only_92pct_ret,
        "satellite_c1_btc_trend_alone": satellite_w,
    }

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "sanity backtest, not a new edge claim; reuses the frozen C1 rule from "
                "alt_premia_preregistered_bars.py and its cost model",
        "reused_prereg_fingerprint": prereg.bars_fingerprint(),
        "window": {"start": WINDOW_START, "end": WINDOW_END, "reason": "BTC data availability (C1's own eval_start)"},
        "portfolio_definition": {
            "core_weight": CORE_WEIGHT, "satellite_weight": SATELLITE_WEIGHT,
            "core_targets": CORE_TARGETS, "core_rebalance": "monthly (first US trading day) + 2%-absolute daily drift band",
            "satellite_rule": "C1_btc_trend, unmodified (see alt_premia_preregistered_bars.CANDIDATES['C1_btc_trend'])",
            "capital_treatment": "sleeved: 92/8 fixed at inception, sleeves never rebalanced against each other",
        },
        "core_rebalance_events_full_history": core_events,
        "kill_switch_live_config": KILL_SWITCH_LIVE,
        "results": {},
    }

    for name, s in series.items():
        w = ape.window(s, WINDOW_START, WINDOW_END)
        rfw = ape.window(rf, WINDOW_START, WINDOW_END)
        report["results"][name] = metrics(w, rfw)
        m = report["results"][name]
        log.info("%-30s CAGR %+.2f%%  vol %.2f%%  Sharpe(ex-Tbill) %+.2f  maxDD %.1f%%",
                 name, 100 * m["cagr"], 100 * m["annualized_vol"], m["sharpe_above_tbills"], 100 * m["max_drawdown"])

    # Satellite contribution: sharpe/cagr delta of portfolio vs core-only(92%+cash)
    p = report["results"]["portfolio_92_8"]
    c = report["results"]["core_92pct_plus_8pct_cash"]
    report["satellite_contribution"] = {
        "definition": "portfolio_92_8 minus core_92pct_plus_8pct_cash, same 92% core, isolates swapping the "
                       "unallocated 8%% between cash and the BTC sleeve",
        "delta_cagr": p["cagr"] - c["cagr"],
        "delta_sharpe_above_tbills": p["sharpe_above_tbills"] - c["sharpe_above_tbills"],
        "delta_vol": p["annualized_vol"] - c["annualized_vol"],
        "delta_max_drawdown": p["max_drawdown"] - c["max_drawdown"],
    }

    # Satellite actual NAV share drift (sleeved capital, no cross-rebalancing)
    sat_share_w = ape.window(sat_share, WINDOW_START, WINDOW_END)
    core_share_w = ape.window(core_share, WINDOW_START, WINDOW_END)
    report["satellite_actual_share"] = {
        "start": float(sat_share_w.iloc[0]), "end": float(sat_share_w.iloc[-1]),
        "max": float(sat_share_w.max()), "min": float(sat_share_w.min()),
        "note": "target is a fixed 8%; with no cross-sleeve rebalancing this drifts with relative sleeve performance",
    }
    report["core_actual_share_end"] = float(core_share_w.iloc[-1])

    # Sensitivity: same portfolio, but the 92/8 sleeve split is also restored at
    # each of the core's own monthly rebalance days (an alternative design choice,
    # not the pre-registered default -- see docstring on sleeved_portfolio_periodic_reset).
    me_w = ape.month_end_flags(core_w.index)
    reset_flags = np.r_[False, me_w[:-1]]  # trade at close of first trading day of month, same as core
    port_reset = sleeved_portfolio_periodic_reset(core_w, satellite_w, CORE_WEIGHT, SATELLITE_WEIGHT, reset_flags)
    report["sensitivity_monthly_sleeve_rebalance"] = metrics(port_reset, rf_w)
    report["sensitivity_monthly_sleeve_rebalance"]["note"] = (
        "alternative to the headline 'sleeved, never cross-rebalanced' result: also resets the 92/8 "
        "split monthly. Included because the un-rebalanced satellite share drifts to "
        f"{report['satellite_actual_share']['end']:.0%} of NAV by {WINDOW_END} (see satellite_actual_share)."
    )
    log.info("sensitivity (monthly sleeve reset): CAGR %+.2f%%  vol %.2f%%  Sharpe %+.2f  maxDD %.1f%%",
             100 * report["sensitivity_monthly_sleeve_rebalance"]["cagr"],
             100 * report["sensitivity_monthly_sleeve_rebalance"]["annualized_vol"],
             report["sensitivity_monthly_sleeve_rebalance"]["sharpe_above_tbills"],
             100 * report["sensitivity_monthly_sleeve_rebalance"]["max_drawdown"])

    # Drawdown distribution / kill-switch calibration on the full portfolio
    port_w = ape.window(port_ret, WINDOW_START, WINDOW_END)
    dd = _drawdown_series(port_w)
    episodes = _drawdown_episodes(port_w, min_depth=0.02)
    trips = [e for e in episodes if e["depth"] <= -KILL_SWITCH_LIVE]
    report["drawdown_distribution"] = {
        "episodes_ge_2pct": episodes,
        "kill_switch_threshold": KILL_SWITCH_LIVE,
        "episodes_that_would_have_tripped_8pct": trips,
        "n_trips": len(trips),
        "pct_of_days_in_ge_8pct_drawdown": float((dd >= KILL_SWITCH_LIVE).mean()),
        "drawdown_percentiles": {
            str(q): float(np.percentile(dd.to_numpy(), q)) for q in (50, 75, 90, 95, 99, 100)
        },
    }
    depths = sorted(-e["depth"] for e in episodes)
    if depths:
        # Where 8% sits among the realised drawdown episodes.
        n_deeper_than_8 = sum(1 for d in depths if d >= KILL_SWITCH_LIVE)
        report["drawdown_distribution"]["kill_switch_calibration_note"] = (
            f"{n_deeper_than_8} of {len(depths)} distinct drawdown episodes >=2% would have reached or "
            f"exceeded the live 8% kill-switch threshold; deepest episode {max(depths):.1%}."
        )

    # Legacy (old live IBKR system) control, overlapping window only
    if LEGACY_RETURNS_PARQUET.exists():
        legacy = pd.read_parquet(LEGACY_RETURNS_PARQUET)["ret"]
        legacy.index = pd.DatetimeIndex(legacy.index)
        lo, hi = legacy.index.min(), legacy.index.max()
        log.info("legacy C0 live-config control available %s -> %s", lo.date(), hi.date())
        legacy_metrics = {}
        for name, s in {**series, "legacy_C0_live_config": legacy}.items():
            w = ape.window(s, str(lo.date()), str(hi.date()))
            rfw = ape.window(rf, str(lo.date()), str(hi.date()))
            legacy_metrics[name] = metrics(w, rfw)
        report["overlapping_window_vs_legacy"] = {
            "start": str(lo.date()), "end": str(hi.date()), "results": legacy_metrics,
        }
    else:
        log.warning("legacy returns parquet not found at %s; skipping control comparison", LEGACY_RETURNS_PARQUET)

    out_path = Path(args.report) if args.report else _ROOT / "docs" / "allocation_portfolio_backtest_2026_09.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    log.info("wrote %s", out_path)
    log.info("kill-switch calibration: %s", report["drawdown_distribution"].get("kill_switch_calibration_note"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
