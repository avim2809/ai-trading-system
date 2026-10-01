#!/usr/bin/env python3
"""Data-quality diagnostic for the insider-cluster evaluation.

Scans every ticker's daily adjusted_close for moves that look like bad data
(a huge one-day jump that reverts within a few sessions, or an
adjusted_close/close ratio that jumps discontinuously with no matching
corporate action), cross-references against the PRIMARY run's
events_evaluated.parquet to find which real events those spikes corrupted,
and recomputes headline stats after excluding the offending tickers --
labelled POST-HOC DIAGNOSTIC, not a change to the frozen verdict.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/local/store/git/ai-trading-system")
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fetch_eodhd_prices import price_path  # noqa: E402
from firm.eval.overfitting import cscv_pbo, deflated_sharpe  # noqa: E402
import insider_cluster_preregistered_bars as prereg  # noqa: E402

OUT = Path(
    "/tmp/claude-0/-local-store-git-ai-trading-system/"
    "c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/runs/insider_recompute"
)
PRIMARY = Path(
    "/tmp/claude-0/-local-store-git-ai-trading-system/"
    "c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/runs/insider"
)


def load_price(ticker: str) -> pd.DataFrame | None:
    path = price_path(ticker)
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    if d.empty:
        return None
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for c in ("open", "close", "adjusted_close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ratio = (d["adjusted_close"] / d["close"]).where(d["close"] > 0)
    d["adj_open"] = d["open"] * ratio
    d["ratio"] = ratio
    return d


def find_event_level_offenders(ev: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    ev = ev.copy()
    ev["abs_xs_net"] = ev["xs_net"].abs()
    return ev.sort_values("abs_xs_net", ascending=False).head(top_n)


def scan_ticker_for_spikes(d: pd.DataFrame, entry_date, exit_date) -> list[dict]:
    """Look for a single-day adjusted_close move > +200% / < -70% that mostly
    reverts within 5 sessions, inside [entry_date, exit_date]."""
    sub = d[(d["date"] >= entry_date - pd.Timedelta(days=10)) &
            (d["date"] <= exit_date + pd.Timedelta(days=10))].reset_index(drop=True)
    if len(sub) < 3:
        return []
    ret = sub["adjusted_close"].pct_change()
    hits = []
    for i in range(1, len(sub)):
        r = ret.iloc[i]
        if not np.isfinite(r):
            continue
        if r > 2.0 or r < -0.70:
            # reversion check: within next 5 sessions, does price come back
            # within 30% of pre-spike level?
            pre = sub["adjusted_close"].iloc[i - 1]
            fwd = sub["adjusted_close"].iloc[i:i + 6]
            reverts = bool((abs(fwd - pre) / pre < 0.30).any()) if pre and np.isfinite(pre) else False
            hits.append({"date": str(sub["date"].iloc[i].date()), "ret": float(r),
                        "prev_close": float(pre), "spike_close": float(sub["adjusted_close"].iloc[i]),
                        "ratio_prev": float(sub["ratio"].iloc[i - 1]) if np.isfinite(sub["ratio"].iloc[i - 1]) else None,
                        "ratio_spike": float(sub["ratio"].iloc[i]) if np.isfinite(sub["ratio"].iloc[i]) else None,
                        "reverts_within_5d": reverts})
    return hits


def main() -> None:
    ev_primary = pd.read_parquet(PRIMARY / "events_evaluated.parquet")
    top = find_event_level_offenders(ev_primary, top_n=30)

    cache: dict[str, pd.DataFrame] = {}
    offenders = []
    for r in top.itertuples(index=False):
        if r.ticker not in cache:
            cache[r.ticker] = load_price(r.ticker)
        d = cache[r.ticker]
        if d is None:
            continue
        hits = scan_ticker_for_spikes(d, r.entry_date, r.exit_date)
        offenders.append({
            "ticker": r.ticker, "hold": r.hold_name, "entry_date": str(r.entry_date.date()),
            "exit_date": str(r.exit_date.date()), "gross": r.gross, "net": r.net,
            "xs_net": r.xs_net, "bench_primary": r.bench_primary, "adv_bucket": r.adv_bucket,
            "early_exit": bool(r.early_exit), "spikes": hits,
        })

    ratio_offenders_count = 0
    for o in offenders:
        for h in o["spikes"]:
            rp, rs = h.get("ratio_prev"), h.get("ratio_spike")
            if rp and rs and rp > 0:
                jump = abs(rs - rp) / rp
                h["adj_ratio_jump_pct"] = float(jump)
                if jump > 0.5:
                    ratio_offenders_count += 1

    (OUT / "offenders_raw.json").write_text(json.dumps(offenders, indent=2, default=str))

    # classify each offender heuristically
    classified = []
    for o in offenders:
        likely_bad = False
        reason = "no single-day spike >200%/-70% found near this event window"
        if o["spikes"]:
            spike_reverting = [h for h in o["spikes"] if h["reverts_within_5d"]]
            ratio_jumpy = [h for h in o["spikes"] if h.get("adj_ratio_jump_pct", 0) and h["adj_ratio_jump_pct"] > 0.5]
            if spike_reverting:
                likely_bad = True
                reason = f"{len(spike_reverting)} spike(s) revert within 5 sessions -- transient, not a real re-rating"
            elif ratio_jumpy:
                likely_bad = True
                reason = f"{len(ratio_jumpy)} spike(s) coincide with an adjusted_close/close ratio jump (>50%) -- looks like a split/scale artifact, not a real move"
            else:
                reason = "large move present but does not revert and no ratio jump -- plausibly real (e.g. buyout/halt)"
        classified.append({**o, "likely_bad_data": likely_bad, "reason": reason})

    (OUT / "offenders_classified.json").write_text(json.dumps(classified, indent=2, default=str))

    bad_tickers = sorted({o["ticker"] for o in classified if o["likely_bad_data"]})
    print("top offenders scanned:", len(classified))
    print("flagged likely-bad-data tickers:", bad_tickers)

    # -----------------------------------------------------------------
    # POST-HOC: recompute headline stats on the PRIMARY run's own event
    # table, excluding events on flagged tickers, plus winsorized stats.
    # -----------------------------------------------------------------
    def month_cluster_bootstrap_mean(values, month_keys, n_boot, seed):
        rng = np.random.default_rng(seed)
        uniq, inv = np.unique(month_keys, return_inverse=True)
        sums = np.bincount(inv, weights=values, minlength=len(uniq))
        counts = np.bincount(inv, minlength=len(uniq))
        draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)

    ALPHA = prereg.BOOTSTRAP["alpha_one_sided"]
    N_BOOT = prereg.BOOTSTRAP["n_boot"]
    SEED = prereg.SEED

    posthoc = {}
    ev_clean_all = ev_primary[~ev_primary["ticker"].isin(bad_tickers)].copy()
    for hname in ("3_month", "6_month"):
        strict = ev_primary[(ev_primary["hold_name"] == hname) & (ev_primary["name_ok"] == True)]  # noqa: E712
        clean = strict[~strict["ticker"].isin(bad_tickers)]
        v = clean["xs_net"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        months = clean.loc[np.isfinite(clean["xs_net"]), "entry_date"].dt.to_period("M").astype(str).to_numpy()
        boot = month_cluster_bootstrap_mean(v, months, N_BOOT, SEED + hash(hname) % 1000)
        winsor_lo, winsor_hi = np.percentile(v, [1, 99])
        v_wins = np.clip(v, winsor_lo, winsor_hi)
        posthoc[hname] = {
            "n_excluding_flagged": int(len(v)),
            "n_excluded_events": int(len(strict) - len(clean)),
            "mean": float(v.mean()), "median": float(np.median(v)),
            "lb": float(np.quantile(boot, ALPHA)), "ub": float(np.quantile(boot, 1 - ALPHA)),
            "winsor_1_99_mean": float(v_wins.mean()),
            "max_xs_net": float(v.max()) if len(v) else None,
            "min_xs_net": float(v.min()) if len(v) else None,
        }

    # calendar-time / DSR / PBO / placebo recompute is expensive (needs full
    # per-day rebuild); approximate the "post-hoc calendar-time" quickly by
    # rebuilding only the strict|bench_primary series per hold, excluding
    # flagged tickers, from the RECOMPUTE's own calendar_time.parquet inputs
    # if available (built independently), else note as not rebuilt here.
    cal_path = OUT / "calendar_time.parquet"
    if cal_path.exists():
        cal = pd.read_parquet(cal_path)
        posthoc["calendar_time_note"] = "see main recompute.json calendar_time; flagged-ticker exclusion not re-run through the daily calendar-time builder (event-level only) -- see report for reasoning"
    (OUT / "posthoc_diagnostic.json").write_text(json.dumps(posthoc, indent=2, default=float))
    print(json.dumps(posthoc, indent=2, default=float))


if __name__ == "__main__":
    main()
