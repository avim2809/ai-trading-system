#!/usr/bin/env python3
"""INDEPENDENT recompute of the frozen insider-purchase-cluster evaluation.

Written from scratch against the FREEZE_NOTES / TIER_A_BARS / COSTS / WINDOWS /
BOOTSTRAP / PLACEBO / DSR / PBO / classify() text in
scripts/insider_cluster_preregistered_bars.py. Does NOT import or copy
scripts/run_insider_cluster_evaluation.py -- only the frozen-design module
(constants + classify()), fetch_eodhd_prices helpers (norm_name, price_path)
and firm.eval.overfitting (DSR/PBO) are reused, per the task's explicit
allowance.

Every interpretive choice left ambiguous by the freeze text is called out in
a comment at the point it's made, so a side-by-side diff against the primary
run's report.json can attribute any mismatch to a specific choice.
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

import insider_cluster_preregistered_bars as prereg  # noqa: E402
from fetch_eodhd_prices import norm_name, price_path  # noqa: E402
from firm.eval.overfitting import cscv_pbo, deflated_sharpe  # noqa: E402

EODHD = ROOT / "data" / "research" / "eodhd"
INSIDER = ROOT / "data" / "research" / "insider"
OUT = Path(
    "/tmp/claude-0/-local-store-git-ai-trading-system/"
    "c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/runs/insider_recompute"
)
OUT.mkdir(parents=True, exist_ok=True)

HOLDS = prereg.EXECUTION["holds_trading_days"]  # {"3_month": 63, "6_month": 126}
# Universe filter values, read directly out of FREEZE_NOTES["universe_filter"]
# text ("20-trading-day median dollar volume ... in [$0.5M, $50M], and
# entry-day adjusted open >= $2.00") -- not copied from the harness, derived
# from the same frozen sentence it was derived from.
ADV_MIN, ADV_MAX, PRICE_MIN = 500_000.0, 50_000_000.0, 2.0
DELIST_HAIRCUT = -0.30  # FREEZE_NOTES["delisting_stress"]: "extra -30%"
ALPHA = prereg.BOOTSTRAP["alpha_one_sided"]  # 0.05/8, one-sided
SEED = prereg.SEED  # 20260930, frozen
N_BOOT = prereg.BOOTSTRAP["n_boot"]  # 5000
N_PLACEBO = prereg.PLACEBO["n_draws"]  # 500
PLACEBO_PCTL = prereg.PLACEBO["pass_percentile"]  # 95.0


# ---------------------------------------------------------------------------
# Price series
# ---------------------------------------------------------------------------

def load_price(ticker: str) -> pd.DataFrame | None:
    """Load a ticker's price series trimmed to what's actually needed
    downstream (date, adjusted_close, adj_open, dollar_vol), as float32 --
    this evaluation touches ~8,300 tickers' full histories at once and the
    host is memory-constrained (a first pass keeping all 6 raw OHLCV columns
    at float64 was OOM-killed at ~3GB RSS with other live services running;
    this trims the resident cache to roughly a fifth of that)."""
    path = price_path(ticker)
    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["date", "open", "close", "adjusted_close", "volume"])
    if d.empty:
        return None
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for c in ("open", "close", "adjusted_close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # FREEZE_NOTES: "EODHD volume is split-adjusted while close is raw ... so
    # close x volume would overstate pre-split dollar volume" -> adj_open =
    # open * adjusted_close / close, dollar_vol = adjusted_close * volume.
    ratio = (d["adjusted_close"] / d["close"]).where(d["close"] > 0)
    adj_open = (d["open"] * ratio).astype("float32")
    dollar_vol = (d["adjusted_close"] * d["volume"]).astype("float32")
    adjusted_close = d["adjusted_close"].astype("float32")
    out = pd.DataFrame({"date": d["date"], "adjusted_close": adjusted_close,
                        "adj_open": adj_open, "dollar_vol": dollar_vol})
    return out


def adv_bucket(adv: float) -> str:
    if adv < 1e6:
        return "adv_lt_1m"
    if adv < 5e6:
        return "adv_1m_5m"
    if adv < 20e6:
        return "adv_5m_20m"
    return "adv_gt_20m"


def bench_symbol(adv: float) -> str:
    # FREEZE_NOTES benchmark_primary: ADV20 < $5M -> IWC, $5M-$20M -> IWM, > $20M -> IJH
    if adv < 5e6:
        return "IWC"
    if adv < 20e6:
        return "IWM"
    return "IJH"


# ---------------------------------------------------------------------------
# Event universe / coverage
# ---------------------------------------------------------------------------

def load_name_index() -> dict[str, set[str]]:
    syms = pd.concat(
        [pd.read_parquet(EODHD / "symbols_active.parquet"),
         pd.read_parquet(EODHD / "symbols_delisted.parquet")],
        ignore_index=True,
    )
    syms = syms[syms["Country"] == "USA"]
    out: dict[str, set[str]] = {}
    for code, grp in syms.groupby("Code")["Name"]:
        out[code] = {norm_name(x) for x in grp}
    return out


def eodhd_code_for(ticker: str, manifest: dict) -> str:
    m = manifest.get(ticker, {})
    rc = m.get("resolved_code")
    if rc:
        return rc
    # same class-separator convention as fetch_eodhd_prices.eodhd_symbol(),
    # minus the ".US" suffix (Code entries in symbols_*.parquet omit it)
    return ticker.strip().upper().replace("/", "-").replace(".", "-")


def name_matches(issuer_name: str, code: str, names_by_code: dict[str, set[str]]) -> bool | None:
    names = names_by_code.get(code)
    if not names:
        return None
    a = norm_name(issuer_name)
    if not a:
        return None
    for n in names:
        if a == n:
            return True
        # "or same first word" per FREEZE_NOTES event_sets.primary
        a_first = a.split()[0] if a.split() else ""
        n_first = n.split()[0] if n.split() else ""
        if a_first and a_first == n_first:
            return True
    return False


def build_coverage() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ev = pd.read_parquet(INSIDER / "cluster_events.parquet")
    tm = pd.read_parquet(EODHD / "ticker_map.parquet")
    manifest = json.loads((EODHD / "manifest.json").read_text())
    names_by_code = load_name_index()
    ev = ev.merge(tm, left_on="ticker", right_on="raw_ticker", how="left", suffixes=("_raw", ""))
    ev["known_date"] = pd.to_datetime(ev["known_date"])

    cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for r in ev.itertuples(index=False):
        t = r.ticker
        base = {"issuer_cik": r.issuer_cik, "issuer_name": r.issuer_name, "raw_ticker": r.raw_ticker,
                "ticker": t, "known_date": r.known_date}
        if not isinstance(t, str) or not t.strip():
            rows.append({**base, "status": "unmappable_ticker"})
            continue
        if t not in cache:
            cache[t] = load_price(t)
        d = cache[t]
        if d is None:
            rows.append({**base, "status": "no_price_data"})
            continue
        # "price series spans the entry date": require the series to already
        # be running at known_date (so ADV/price history exists for the
        # filter), and for a next-session-open entry to exist after it.
        if r.known_date < d["date"].iloc[0]:
            rows.append({**base, "status": "series_not_spanning_event"})
            continue
        idx = int(d["date"].searchsorted(r.known_date, side="right"))
        if idx >= len(d):
            rows.append({**base, "status": "series_not_spanning_event"})
            continue
        code = eodhd_code_for(t, manifest)
        ok = name_matches(r.issuer_name, code, names_by_code)
        rows.append({**base, "status": "covered", "entry_idx": idx,
                     "entry_date": d["date"].iloc[idx], "name_ok": ok})
    return pd.DataFrame(rows), cache


# ---------------------------------------------------------------------------
# Event-level returns
# ---------------------------------------------------------------------------

def compute_event_returns(events: pd.DataFrame, cache: dict, benches: dict[str, pd.DataFrame],
                          hold: int, data_end: pd.Timestamp) -> pd.DataFrame:
    out = []
    open_until: dict[str, int] = {}
    covered = events[events["status"] == "covered"].sort_values("entry_date")
    for r in covered.itertuples(index=False):
        d = cache[r.ticker]
        e = int(r.entry_idx)
        if e < 20:
            continue  # not enough history for a 20-session ADV window
        window = d["dollar_vol"].iloc[e - 20:e].to_numpy()
        adv = float(np.nanmedian(window)) if len(window) else float("nan")
        px = float(d["adj_open"].iloc[e])
        if not (np.isfinite(adv) and ADV_MIN <= adv <= ADV_MAX and np.isfinite(px) and px >= PRICE_MIN):
            continue
        if open_until.get(r.ticker, -1) >= e:
            continue  # overlap rule
        x = e + hold - 1
        early = False
        if x >= len(d):
            if d["date"].iloc[-1] >= data_end - pd.Timedelta(days=5):
                continue  # hold hasn't completed as of the data end yet
            x, early = len(d) - 1, True
        open_until[r.ticker] = x
        entry_date, exit_date = d["date"].iloc[e], d["date"].iloc[x]
        gross = float(d["adjusted_close"].iloc[x] / px - 1.0)
        cost = prereg.COSTS["round_trip_bps_by_adv_bucket"][adv_bucket(adv)] / 1e4
        net = gross - cost
        net_2x = gross - 2 * cost
        net_delist = ((1 + gross) * (1 + DELIST_HAIRCUT) - 1 - cost) if early else net
        row = {"issuer_cik": r.issuer_cik, "ticker": r.ticker, "known_date": r.known_date,
               "entry_date": entry_date, "exit_date": exit_date, "hold": hold, "early_exit": early,
               "name_ok": r.name_ok, "adv20": adv, "adv_bucket": adv_bucket(adv),
               "gross": gross, "cost": cost, "net": net, "net_2x_cost": net_2x,
               "net_delist_stress": net_delist, "bench_primary_sym": bench_symbol(adv)}
        for label, sym in (("bench_primary", bench_symbol(adv)), ("bench_iwm", "IWM")):
            b = benches[sym]
            bi = int(b["date"].searchsorted(entry_date))
            bx = int(b["date"].searchsorted(exit_date))
            if bi < len(b) and bx < len(b) and b["date"].iloc[bi] == entry_date and b["date"].iloc[bx] == exit_date:
                row[label] = float(b["adjusted_close"].iloc[bx] / b["adj_open"].iloc[bi] - 1.0)
            else:
                row[label] = float("nan")
        out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["xs_net"] = df["net"] - df["bench_primary"]
    df["xs_net_2x_cost"] = df["net_2x_cost"] - df["bench_primary"]
    df["xs_net_delist_stress"] = df["net_delist_stress"] - df["bench_primary"]
    df["xs_net_iwm"] = df["net"] - df["bench_iwm"]
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def month_cluster_bootstrap_mean(values: np.ndarray, month_keys: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Cluster-bootstrap the mean by resampling calendar months of entry
    (FREEZE_NOTES bootstrap) with replacement, keeping every event in a
    drawn month."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(month_keys, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq))
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def summarize(df: pd.DataFrame, col: str, seed: int) -> dict:
    if df.empty:
        return {"n": 0, "mean": float("nan")}
    v = df[col].to_numpy(dtype=float)
    finite = np.isfinite(v)
    v = v[finite]
    if len(v) < 30:
        return {"n": int(len(v)), "mean": float(np.mean(v)) if len(v) else float("nan")}
    months = df.loc[finite, "entry_date"].dt.to_period("M").astype(str).to_numpy()
    boot = month_cluster_bootstrap_mean(v, months, N_BOOT, seed)
    return {
        "n": int(len(v)), "months": int(len(np.unique(months))),
        "mean": float(v.mean()), "median": float(np.median(v)),
        "hit_rate": float((v > 0).mean()),
        "lb": float(np.quantile(boot, ALPHA)), "ub": float(np.quantile(boot, 1 - ALPHA)),
    }


def positions_of(src: np.ndarray, target: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(target, src)
    ok = pos < len(target)
    ok2 = ok.copy()
    ok2[ok] = target[pos[ok]] == src[ok]
    return np.where(ok2, pos, -1)


def calendar_time_series(df: pd.DataFrame, cache: dict, benches: dict, bench_col: str,
                         cal_dates: pd.DatetimeIndex, prep: dict) -> pd.Series:
    """Daily equal-weight net-of-cost excess return across open positions.
    Day 1 = adj_open->close (for stock and benchmark alike); later days
    close->close. Round-trip cost is split half on the entry day, half on
    the exit day (a reasonable way to place a single round-trip charge into
    a daily series when the freeze text doesn't specify the split -- flagged
    as an interpretive choice). Days with no open position earn 0."""
    cal = cal_dates.to_numpy()
    num = np.zeros(len(cal))
    cnt = np.zeros(len(cal))
    for r in df.itertuples(index=False):
        sym = r.bench_primary_sym if bench_col == "bench_primary" else "IWM"
        if r.ticker not in prep:
            d = cache[r.ticker]
            dd = d["date"].to_numpy()
            ret = d["adjusted_close"].pct_change().to_numpy().astype("float32")
            day1 = ((d["adjusted_close"] / d["adj_open"]).to_numpy() - 1).astype("float32")
            prep[r.ticker] = (dd, ret, day1, positions_of(dd, cal).astype("int32"), {})
        dd, ret, day1, cal_pos, bmap = prep[r.ticker]
        if sym not in bmap:
            b = benches[sym]
            bret = b["adjusted_close"].pct_change().to_numpy()
            bday1 = (b["adjusted_close"] / b["adj_open"]).to_numpy() - 1
            bpos = positions_of(dd, b["date"].to_numpy())
            bmap[sym] = (np.where(bpos >= 0, bret[bpos], np.nan),
                        np.where(bpos >= 0, bday1[bpos], np.nan))
        bs_all, bday1_all = bmap[sym]
        e = int(np.searchsorted(dd, np.datetime64(r.entry_date)))
        x = int(np.searchsorted(dd, np.datetime64(r.exit_date)))
        if x - e + 1 < 2:
            continue
        s = ret[e:x + 1].copy()
        s[0] = day1[e]
        bs = bs_all[e:x + 1].copy()
        bs[0] = bday1_all[e]
        xs = s - bs
        xs[np.isnan(xs)] = 0.0
        xs[0] -= r.cost / 2
        xs[-1] -= r.cost / 2
        cp = cal_pos[e:x + 1]
        m = cp >= 0
        num[cp[m]] += xs[m]
        cnt[cp[m]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(cnt > 0, num / cnt, 0.0)
    return pd.Series(vals, index=cal_dates)


def run_placebo(strict: pd.DataFrame, all_covered_events: pd.DataFrame, cache: dict, benches: dict,
                hold: int, n_draws: int, seed: int, data_end: pd.Timestamp) -> np.ndarray:
    """For each ticker with >=1 strict event, draw as many random entry
    indices (same ADV/price filter, same hold, same cost rule, benchmark by
    ADV bucket) as it has strict events, excluding +/-63 trading days around
    ANY covered real event on that ticker (not just strict ones -- a real
    signal is a real signal whether or not the name-match passed)."""
    per_ticker = strict.groupby("ticker").size()
    real_idx: dict[str, np.ndarray] = {}
    for t, g in all_covered_events[all_covered_events["status"] == "covered"].groupby("ticker"):
        real_idx[t] = g["entry_idx"].astype(int).to_numpy()

    pools: dict[str, np.ndarray] = {}
    for t in per_ticker.index:
        d = cache[t]
        n = len(d)
        if n <= 20 + hold:
            continue
        e = np.arange(20, n - hold)
        adv = d["dollar_vol"].rolling(20).median().shift(1).to_numpy()
        px = d["adj_open"].to_numpy()
        ok = (adv[e] >= ADV_MIN) & (adv[e] <= ADV_MAX) & np.isfinite(px[e]) & (px[e] >= PRICE_MIN)
        for ri in real_idx.get(t, []):
            ok &= np.abs(e - ri) > 63
        cand = e[ok]
        if len(cand) == 0:
            continue
        entry_dates = d["date"].to_numpy()[cand]
        exit_dates = d["date"].to_numpy()[cand + hold - 1]
        gross = d["adjusted_close"].to_numpy()[cand + hold - 1] / px[cand] - 1
        advs = adv[cand]
        buckets = np.select([advs < 1e6, advs < 5e6, advs < 20e6], [120.0, 60.0, 30.0], 10.0)
        cost = buckets / 1e4
        syms = np.where(advs < 5e6, "IWC", np.where(advs < 20e6, "IWM", "IJH"))
        bench = np.full(len(cand), np.nan)
        for sym in ("IWC", "IWM", "IJH"):
            msk = syms == sym
            if not msk.any():
                continue
            b = benches[sym]
            bd = b["date"].to_numpy()
            bi = np.searchsorted(bd, entry_dates[msk])
            bx = np.searchsorted(bd, exit_dates[msk])
            valid = (bi < len(bd)) & (bx < len(bd))
            sub = np.full(msk.sum(), np.nan)
            vi, vx = bi[valid], bx[valid]
            exact = (bd[vi] == entry_dates[msk][valid]) & (bd[vx] == exit_dates[msk][valid])
            tmp = np.full(valid.sum(), np.nan)
            tmp[exact] = (b["adjusted_close"].to_numpy()[vx[exact]] /
                         b["adj_open"].to_numpy()[vi[exact]] - 1.0)
            sub[valid] = tmp
            bench[msk] = sub
        xs = gross - cost - bench
        keep = np.isfinite(xs)
        if keep.any():
            pools[t] = xs[keep]

    rng = np.random.default_rng(seed)
    tickers = [t for t in per_ticker.index if t in pools]
    counts = np.array([per_ticker[t] for t in tickers])
    means = np.full(n_draws, np.nan)
    for k in range(n_draws):
        tot, n = 0.0, 0
        for t, c in zip(tickers, counts):
            pool = pools[t]
            tot += pool[rng.integers(0, len(pool), size=c)].sum()
            n += c
        if n:
            means[k] = tot / n
    return means


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("fingerprint (frozen design):", prereg.bars_fingerprint())

    benches = {s: load_price(s) if False else None for s in ()}  # placeholder, replaced below
    benches = {}
    for s in ("IWC", "IWM", "IJH"):
        d = pd.read_parquet(EODHD / "etfs" / f"{s}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        for c in ("open", "close", "adjusted_close", "volume"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        ratio = (d["adjusted_close"] / d["close"]).where(d["close"] > 0)
        d["adj_open"] = d["open"] * ratio
        benches[s] = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    data_end = benches["IWM"]["date"].iloc[-1]
    print("data_end (last IWM bar):", data_end)

    events, cache = build_coverage()
    events.to_parquet(OUT / "event_coverage.parquet")
    coverage_counts = events["status"].value_counts().to_dict()
    print("coverage:", coverage_counts)

    primary_start = pd.Timestamp(prereg.WINDOWS["primary_post_sample"]["start"])
    primary_end = pd.Timestamp(prereg.WINDOWS["primary_post_sample"]["end"])
    secondary_start = pd.Timestamp(prereg.WINDOWS["secondary_post_publication"]["start"])

    # windowed subset for evaluation: only covered events are subject to the
    # primary window; everything else is kept for coverage bookkeeping only.
    is_covered = events["status"] == "covered"
    in_window = (~is_covered) | ((events["entry_date"] >= primary_start) & (events["entry_date"] <= primary_end))
    events_windowed = events[in_window].copy()

    cal_dates = pd.DatetimeIndex(benches["IWM"]["date"][benches["IWM"]["date"] >= primary_start])
    results: dict = {"coverage_all_events": coverage_counts, "holds": {}}
    variants: dict[str, pd.Series] = {}
    prep: dict = {}
    all_rows = []

    for hname, hold in HOLDS.items():
        df = compute_event_returns(events_windowed, cache, benches, hold, data_end)
        df["hold_name"] = hname
        all_rows.append(df)
        strict = df[df["name_ok"] == True]  # noqa: E712
        res = {"n_evaluated_covered": int(len(df)), "n_primary_strict": int(len(strict)),
               "early_exit_rate": float(strict["early_exit"].mean()) if len(strict) else None}
        seed = SEED + hold
        res["primary"] = summarize(strict, "xs_net", seed)
        res["iwm_benchmark"] = summarize(strict, "xs_net_iwm", seed + 1)
        res["cost_2x"] = summarize(strict, "xs_net_2x_cost", seed + 2)
        res["delist_stress"] = summarize(strict, "xs_net_delist_stress", seed + 3)
        res["post_2013"] = summarize(strict[strict["entry_date"] >= secondary_start], "xs_net", seed + 4)
        res["sensitivity_all_covered"] = summarize(df, "xs_net", seed + 5)
        res["gross_raw_mean"] = float(strict["gross"].mean()) if len(strict) else float("nan")
        res["by_adv_bucket"] = {k: summarize(g, "xs_net", seed + 6) for k, g in strict.groupby("adv_bucket")}
        res["by_year"] = {int(y): float(g["xs_net"].mean()) for y, g in strict.groupby(strict["entry_date"].dt.year)}

        print(f"{hname}: placebo {N_PLACEBO} draws ...")
        pl = run_placebo(strict, events, cache, benches, hold, N_PLACEBO, SEED + 1 + hold, data_end)
        pd.Series(pl).to_frame("placebo_mean_xs").to_parquet(OUT / f"placebo_{hname}.parquet")
        res["placebo_p95"] = float(np.nanpercentile(pl, PLACEBO_PCTL))
        res["placebo_median"] = float(np.nanmedian(pl))

        for set_name, sub in (("strict", strict), ("covered", df)):
            for bcol in ("bench_primary", "bench_iwm"):
                key = f"{hname}|{set_name}|{bcol}"
                variants[key] = calendar_time_series(sub, cache, benches, bcol, cal_dates, prep)

        results["holds"][hname] = res
        print(f"{hname}: n={res['primary']['n']} mean={res['primary']['mean']:.4f} "
              f"lb={res['primary'].get('lb', float('nan')):.4f} placebo_p95={res['placebo_p95']:.4f}")

    events_evaluated = pd.concat(all_rows, ignore_index=True)
    events_evaluated.to_parquet(OUT / "events_evaluated.parquet")
    cal = pd.DataFrame(variants)
    cal.to_parquet(OUT / "calendar_time.parquet")

    trial_sr = (cal.mean() / cal.std(ddof=1)).to_numpy()
    pbo = float(cscv_pbo(cal.to_numpy(), n_partitions=prereg.PBO["n_partitions"]))
    results["pbo"] = pbo
    results["calendar_time"] = {}
    for hname in HOLDS:
        key = f"{hname}|strict|bench_primary"
        x = cal[key].to_numpy()
        active = cal[key] != 0
        results["calendar_time"][hname] = {
            "ann_mean_excess": float(np.mean(x) * 252),
            "ann_vol": float(np.std(x, ddof=1) * math.sqrt(252)),
            "sharpe": float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(252)),
            "dsr": float(deflated_sharpe(x, trial_sr, prior_trials=prereg.DSR["prior_trials"])),
            "days_with_positions": int(active.sum()),
        }

    h = results["holds"]
    both = list(HOLDS)
    bars = {
        "A1": all(h[k]["primary"]["mean"] > 0 and h[k]["primary"].get("lb", -1) > 0 for k in both),
        "A2": all(results["calendar_time"][k]["dsr"] > 0.95 for k in both),
        "A3": all(h[k]["primary"]["mean"] > h[k]["placebo_p95"] for k in both),
        "A4": all(h[k]["primary"]["mean"] > 0 and h[k]["post_2013"]["mean"] > 0 for k in both),
        "A5": all(h[k]["cost_2x"]["mean"] > 0 for k in both),
        "A6": pbo < 0.50,
        "A7": all(h[k]["delist_stress"]["mean"] > 0 for k in both),
    }
    tier_d = any(h[k]["primary"].get("ub", 1) < 0 for k in both)
    tier_b = {"B_a": all(h[k]["primary"]["mean"] > 0 for k in both),
              "B_b": bars["A3"] and bars["A5"] and bars["A7"]}
    results.update({"bars_A": bars, "tier_D_condition": tier_d, "bars_B": tier_b,
                    "tier": prereg.classify(bars, tier_d, tier_b)})

    ev2 = events.copy()
    ev2["year"] = ev2["known_date"].dt.year
    results["missing_data_by_year"] = (ev2.assign(covered=ev2["status"] == "covered")
                                       .groupby("year")["covered"].mean().round(4).to_dict())

    (OUT / "recompute.json").write_text(json.dumps(results, indent=2, default=float))
    print("bars", bars, "tierD", tier_d, "tierB", tier_b, "=>", results["tier"], "PBO", pbo)


if __name__ == "__main__":
    main()
