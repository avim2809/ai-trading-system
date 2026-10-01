#!/usr/bin/env python3
"""POST-HOC DIAGNOSTIC pass (not a change to the frozen verdict): rebuild the
independent recompute's own pipeline with five identified bad-data tickers
excluded entirely (ACRX, CERN: un-adjusted split boundary inside the hold;
SMLP: phantom zero-price Christmas-Day row; QPAC: untradeable zero-volume
shell with garbage quotes; XBKS: recurring ~250x scale-factor data
corruption throughout 2009-2015), and reports headline stats, calendar-time
Sharpe/DSR/PBO and placebo p95 with vs. without them.

Reuses recompute.py's own vetted functions (same process family, not the
forbidden harness) rather than re-deriving the pipeline a third time.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import recompute as rc  # noqa: E402

from firm.eval.overfitting import cscv_pbo, deflated_sharpe  # noqa: E402
import insider_cluster_preregistered_bars as prereg  # noqa: E402

OUT = rc.OUT
BAD_TICKERS = ["ACRX", "CERN", "SMLP", "QPAC", "XBKS"]


def winsorized_mean(v: np.ndarray, lo_pct=1.0, hi_pct=99.0) -> float:
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    return float(np.clip(v, lo, hi).mean())


def main() -> None:
    benches = {}
    for s in ("IWC", "IWM", "IJH"):
        d = pd.read_parquet(rc.EODHD / "etfs" / f"{s}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        for c in ("open", "close", "adjusted_close", "volume"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        ratio = (d["adjusted_close"] / d["close"]).where(d["close"] > 0)
        d["adj_open"] = d["open"] * ratio
        benches[s] = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    data_end = benches["IWM"]["date"].iloc[-1]

    print("building coverage (reloading full price universe)...")
    events, cache = rc.build_coverage()
    primary_start = pd.Timestamp(prereg.WINDOWS["primary_post_sample"]["start"])
    primary_end = pd.Timestamp(prereg.WINDOWS["primary_post_sample"]["end"])
    is_covered = events["status"] == "covered"
    in_window = (~is_covered) | ((events["entry_date"] >= primary_start) & (events["entry_date"] <= primary_end))
    events_windowed = events[in_window].copy()

    cal_dates = pd.DatetimeIndex(benches["IWM"]["date"][benches["IWM"]["date"] >= primary_start])
    variants: dict[str, pd.Series] = {}
    prep: dict = {}
    results = {"bad_tickers_excluded": BAD_TICKERS, "holds": {}}

    for hname, hold in rc.HOLDS.items():
        print(f"{hname}: computing event returns ...")
        df = rc.compute_event_returns(events_windowed, cache, benches, hold, data_end)
        strict = df[df["name_ok"] == True]  # noqa: E712
        clean_df = df[~df["ticker"].isin(BAD_TICKERS)]
        clean_strict = strict[~strict["ticker"].isin(BAD_TICKERS)]
        n_excluded = int(len(strict) - len(clean_strict))

        v = clean_strict["xs_net"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        months = clean_strict.loc[np.isfinite(clean_strict["xs_net"]), "entry_date"].dt.to_period("M").astype(str).to_numpy()
        boot = rc.month_cluster_bootstrap_mean(v, months, rc.N_BOOT, rc.SEED + hold + 777)

        hold_res = {
            "n_before_exclusion": int(len(strict)), "n_after_exclusion": int(len(clean_strict)),
            "n_excluded_events": n_excluded,
            "mean": float(v.mean()), "median": float(np.median(v)),
            "lb": float(np.quantile(boot, rc.ALPHA)), "ub": float(np.quantile(boot, 1 - rc.ALPHA)),
            "winsor_1_99_mean": winsorized_mean(v),
            "max_xs_net": float(v.max()) if len(v) else None,
            "min_xs_net": float(v.min()) if len(v) else None,
        }

        print(f"{hname}: placebo (excluding bad tickers) ...")
        pl = rc.run_placebo(clean_strict, events, cache, benches, hold, rc.N_PLACEBO,
                            rc.SEED + 1 + hold + 777, data_end)
        hold_res["placebo_p95"] = float(np.nanpercentile(pl, rc.PLACEBO_PCTL))
        hold_res["placebo_median"] = float(np.nanmedian(pl))
        hold_res["placebo_frac_above_0.5"] = float((pl > 0.5).mean())

        for set_name, sub in (("strict", clean_strict), ("covered", clean_df)):
            for bcol in ("bench_primary", "bench_iwm"):
                key = f"{hname}|{set_name}|{bcol}"
                variants[key] = rc.calendar_time_series(sub, cache, benches, bcol, cal_dates, prep)

        results["holds"][hname] = hold_res
        print(f"{hname}: mean={hold_res['mean']:.4f} median={hold_res['median']:.4f} "
              f"winsor={hold_res['winsor_1_99_mean']:.4f} placebo_p95={hold_res['placebo_p95']:.4f} "
              f"excluded {n_excluded} events")

    cal = pd.DataFrame(variants)
    trial_sr = (cal.mean() / cal.std(ddof=1)).to_numpy()
    pbo = float(cscv_pbo(cal.to_numpy(), n_partitions=prereg.PBO["n_partitions"]))
    results["pbo_excl"] = pbo
    results["calendar_time_excl"] = {}
    for hname in rc.HOLDS:
        key = f"{hname}|strict|bench_primary"
        x = cal[key].to_numpy()
        active = cal[key] != 0
        results["calendar_time_excl"][hname] = {
            "ann_mean_excess": float(np.mean(x) * 252),
            "ann_vol": float(np.std(x, ddof=1) * math.sqrt(252)),
            "sharpe": float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(252)),
            "dsr": float(deflated_sharpe(x, trial_sr, prior_trials=prereg.DSR["prior_trials"])),
            "days_with_positions": int(active.sum()),
        }

    (OUT / "posthoc_full.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
