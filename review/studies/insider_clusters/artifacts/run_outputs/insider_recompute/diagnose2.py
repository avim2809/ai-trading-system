#!/usr/bin/env python3
"""Data-quality diagnostic v2: distinguishes a correctly-handled split
(raw close jumps, adjusted_close stays continuous because the ratio steps
in sync) from a BAD adjustment (raw close jumps, adjusted_close ALSO jumps
by about the same factor because the ratio failed to step -- i.e. a split
was applied to "close" but never propagated into "adjusted_close" for that
boundary, or a ticker-reuse stitching artifact), and from a transient bad
tick (a spike that reverts within a few sessions, in both series)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/local/store/git/ai-trading-system")
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_eodhd_prices import price_path  # noqa: E402

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
    d["raw_ret"] = d["close"].pct_change()
    d["adj_ret"] = d["adjusted_close"].pct_change()
    d["ratio"] = (d["adjusted_close"] / d["close"]).where(d["close"] > 0)
    return d


def classify_ticker_window(d: pd.DataFrame, entry_date, exit_date) -> dict:
    # Scan the FULL hold (entry->exit), not just near the boundaries: an
    # uncorrected split anywhere inside the hold corrupts the cumulative
    # entry-to-exit return, not only one that lands on the entry/exit day.
    sub = d[(d["date"] >= entry_date - pd.Timedelta(days=3)) &
            (d["date"] <= exit_date + pd.Timedelta(days=3))].reset_index(drop=True)
    if len(sub) < 3:
        return {"verdict": "insufficient_data", "spikes": []}
    spikes = []
    for i in range(1, len(sub)):
        ar = sub["adj_ret"].iloc[i]
        if not np.isfinite(ar) or not (ar > 2.0 or ar < -0.70):
            continue
        rr = sub["raw_ret"].iloc[i]
        # "unadjusted" test: the raw move and the adjusted move are ~equal
        # (the ratio did NOT step to absorb a real-looking raw jump) AND the
        # ratio is NOT simply 1.0 throughout (i.e. there IS split-adjustment
        # machinery active for this ticker elsewhere in its history, so a
        # constant non-1 ratio spanning this jump means the factor failed to
        # update at the correct boundary) -- a jump with ratio==1.0 on both
        # sides has no split machinery involved at all and is far more
        # likely genuine news (earnings, trial data, M&A) than a data bug.
        ratio_before = sub["ratio"].iloc[i - 1]
        ratio_after = sub["ratio"].iloc[i]
        ratio_nontrivial = (np.isfinite(ratio_before) and abs(ratio_before - 1.0) > 0.02) or \
                           (np.isfinite(ratio_after) and abs(ratio_after - 1.0) > 0.02)
        unadjusted = (np.isfinite(rr) and abs(rr) > 1.0 and abs(ar - rr) < 0.25 * max(abs(rr), abs(ar))
                     and ratio_nontrivial)
        # "properly adjusted split" test: raw jumps a lot but adjusted_close
        # is nearly continuous around it (ratio stepped to absorb the raw move)
        continuous_adj = np.isfinite(rr) and abs(rr) > 1.0 and abs(ar) < 0.10
        pre = sub["adjusted_close"].iloc[i - 1]
        fwd = sub["adjusted_close"].iloc[i:i + 6]
        reverts = bool((abs(fwd - pre) / pre < 0.30).any()) if pre and np.isfinite(pre) else False
        spikes.append({
            "date": str(sub["date"].iloc[i].date()), "adj_ret": float(ar),
            "raw_ret": float(rr) if np.isfinite(rr) else None,
            "ratio_before": float(sub["ratio"].iloc[i - 1]) if np.isfinite(sub["ratio"].iloc[i - 1]) else None,
            "ratio_after": float(sub["ratio"].iloc[i]) if np.isfinite(sub["ratio"].iloc[i]) else None,
            "raw_close_before": float(sub["close"].iloc[i - 1]), "raw_close_after": float(sub["close"].iloc[i]),
            "adj_close_before": float(pre), "adj_close_after": float(sub["adjusted_close"].iloc[i]),
            "volume": int(sub["volume"].iloc[i]) if np.isfinite(sub["volume"].iloc[i]) else None,
            "unadjusted_split_like": bool(unadjusted), "properly_adjusted_split": bool(continuous_adj),
            "reverts_within_5d": reverts,
        })
    if not spikes:
        return {"verdict": "no_spike_in_window", "spikes": []}
    if any(s["unadjusted_split_like"] for s in spikes):
        return {"verdict": "BAD_DATA_unadjusted_split_or_ticker_reuse", "spikes": spikes}
    if any(s["reverts_within_5d"] and not s["properly_adjusted_split"] for s in spikes):
        return {"verdict": "BAD_DATA_transient_bad_tick", "spikes": spikes}
    if all(s["properly_adjusted_split"] for s in spikes):
        return {"verdict": "plausibly_real_correctly_adjusted_split", "spikes": spikes}
    return {"verdict": "plausibly_real_large_move", "spikes": spikes}


def main() -> None:
    ev = pd.read_parquet(PRIMARY / "events_evaluated.parquet")
    ev["abs_xs_net"] = ev["xs_net"].abs()
    top = ev.sort_values("abs_xs_net", ascending=False).head(25)

    cache: dict[str, pd.DataFrame] = {}
    report = []
    for r in top.itertuples(index=False):
        if r.ticker not in cache:
            cache[r.ticker] = load_price(r.ticker)
        d = cache[r.ticker]
        if d is None:
            report.append({"ticker": r.ticker, "hold": r.hold_name, "verdict": "no_price_file"})
            continue
        res = classify_ticker_window(d, r.entry_date, r.exit_date)
        report.append({
            "ticker": r.ticker, "hold": r.hold_name,
            "entry_date": str(r.entry_date.date()), "exit_date": str(r.exit_date.date()),
            "gross": float(r.gross), "net": float(r.net), "xs_net": float(r.xs_net),
            "bench_primary": float(r.bench_primary), "adv_bucket": r.adv_bucket,
            **res,
        })

    (OUT / "offenders_v2.json").write_text(json.dumps(report, indent=2, default=str))
    bad_tickers = sorted({o["ticker"] for o in report if o.get("verdict", "").startswith("BAD_DATA")})
    print("BAD DATA tickers flagged:", bad_tickers)
    for o in report:
        print(f"{o['ticker']:8s} {o['hold']:8s} xs_net={o.get('xs_net', float('nan')):+8.3f}  {o.get('verdict')}")

    (OUT / "bad_tickers.json").write_text(json.dumps(bad_tickers))


if __name__ == "__main__":
    main()
