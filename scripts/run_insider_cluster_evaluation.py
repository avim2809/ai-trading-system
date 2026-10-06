"""Pre-registered evaluation of SEC insider-purchase clusters (2026-09-30).

Frozen design: ``scripts/insider_cluster_preregistered_bars.py`` (FREEZE_NOTES
supersede the draft text where they conflict). Inputs (gitignored, local):

    data/research/insider/cluster_events.parquet      SEC Form 4 cluster events
    data/research/eodhd/prices/<T>.parquet            EODHD EOD incl. delisted
    data/research/eodhd/etfs/{IWC,IWM,IJH}.parquet    benchmarks
    data/research/eodhd/{ticker_map,symbols_*}.parquet, manifest.json

    python scripts/run_insider_cluster_evaluation.py --out <dir> \
        [--report docs/insider_cluster_evaluation_2026_09.json] [--append-ledger]

Writes events.parquet (one row per evaluated event x hold), calendar-time
series, placebo draws and report.json to --out, so an independent recompute
can start from the same rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import insider_cluster_preregistered_bars as prereg
from fetch_eodhd_prices import norm_name, price_path

log = logging.getLogger(__name__)

EODHD = _ROOT / "data" / "research" / "eodhd"
INSIDER = _ROOT / "data" / "research" / "insider"
LEDGER = _ROOT / "docs" / "insider_cluster_trial_history.json"
HOLDS = prereg.EXECUTION["holds_trading_days"]            # {"3_month": 63, "6_month": 126}
ADV_MIN, ADV_MAX, PRICE_MIN = 500_000.0, 50_000_000.0, 2.0
DELIST_HAIRCUT = -0.30
ALPHA = prereg.BOOTSTRAP["alpha_one_sided"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_series(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for c in ("open", "close", "adjusted_close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ratio = (d["adjusted_close"] / d["close"]).where(d["close"] > 0)
    d["adj_open"] = d["open"] * ratio
    d["dollar_vol"] = d["adjusted_close"] * d["volume"]
    return d


def adv_bucket(adv: float) -> str:
    if adv < 1e6:
        return "adv_lt_1m"
    if adv < 5e6:
        return "adv_1m_5m"
    if adv < 20e6:
        return "adv_5m_20m"
    return "adv_gt_20m"


def bench_symbol(adv: float) -> str:
    return "IWC" if adv < 5e6 else ("IWM" if adv < 20e6 else "IJH")


def name_agrees(issuer_name: str, codes: list[str], names_by_code: dict) -> bool | None:
    names: set[str] = set()
    for c in codes:
        names |= names_by_code.get(c, set())
    if not names:
        return None
    a = norm_name(issuer_name)
    return any(a == n or a.split()[:1] == n.split()[:1] for n in names)


def build_events(last_market_date: pd.Timestamp) -> pd.DataFrame:
    ev = pd.read_parquet(INSIDER / "cluster_events.parquet")
    tm = pd.read_parquet(EODHD / "ticker_map.parquet")
    manifest = json.loads((EODHD / "manifest.json").read_text())
    ev = ev.merge(tm, left_on="ticker", right_on="raw_ticker", how="left", suffixes=("_raw", ""))
    ev["known_date"] = pd.to_datetime(ev["known_date"])
    syms = pd.concat([pd.read_parquet(EODHD / "symbols_active.parquet"),
                      pd.read_parquet(EODHD / "symbols_delisted.parquet")])
    syms = syms[syms["Country"] == "USA"]
    names_by_code = syms.groupby("Code")["Name"].apply(lambda s: {norm_name(x) for x in s}).to_dict()

    rows = []
    cache: dict[str, pd.DataFrame | None] = {}
    for r in ev.itertuples(index=False):
        t = r.ticker
        base = {"issuer_cik": r.issuer_cik, "issuer_name": r.issuer_name, "raw_ticker": r.raw_ticker,
                "ticker": t, "known_date": r.known_date}
        if not isinstance(t, str) or not t:
            rows.append({**base, "status": "unmappable_ticker"})
            continue
        if t not in cache:
            cache[t] = load_series(price_path(t))
        d = cache[t]
        if d is None or d.empty:
            rows.append({**base, "status": "no_price_data"})
            continue
        idx = int(d["date"].searchsorted(r.known_date, side="right"))  # first date > known_date
        if idx >= len(d) or r.known_date < d["date"].iloc[0]:
            rows.append({**base, "status": "series_not_spanning_event"})
            continue
        m = manifest.get(t, {})
        codes = [m.get("resolved_code") or t.replace(".", "-")]
        rows.append({**base, "status": "covered", "entry_idx": idx, "entry_date": d["date"].iloc[idx],
                     "name_ok": name_agrees(r.issuer_name, codes, names_by_code)})
    return pd.DataFrame(rows), cache


def event_returns(events: pd.DataFrame, cache: dict, benches: dict, hold: int,
                  last_market_date: pd.Timestamp) -> pd.DataFrame:
    """One row per covered event passing the universe filter, with net/excess returns."""
    out = []
    open_until: dict[str, int] = {}
    for r in events[events["status"] == "covered"].sort_values("entry_date").itertuples(index=False):
        d = cache[r.ticker]
        e = int(r.entry_idx)
        if e < 20:
            continue
        adv = float(np.nanmedian(d["dollar_vol"].iloc[e - 20:e].to_numpy()))
        px = float(d["adj_open"].iloc[e])
        if not (np.isfinite(adv) and ADV_MIN <= adv <= ADV_MAX and np.isfinite(px) and px >= PRICE_MIN):
            continue
        # overlap rule: skip while an earlier event's position on this ticker is open
        if open_until.get(r.ticker, -1) >= e:
            continue
        x = e + hold - 1
        early = False
        if x >= len(d):
            if d["date"].iloc[-1] >= last_market_date - pd.Timedelta(days=5):
                continue  # hold runs past the data end: not evaluable yet
            x, early = len(d) - 1, True  # series ended (delisted) inside the hold
        open_until[r.ticker] = x
        entry_date, exit_date = d["date"].iloc[e], d["date"].iloc[x]
        gross = float(d["adjusted_close"].iloc[x] / px - 1.0)
        cost = prereg.COSTS["round_trip_bps_by_adv_bucket"][adv_bucket(adv)] / 1e4
        row = {"issuer_cik": r.issuer_cik, "ticker": r.ticker, "known_date": r.known_date,
               "entry_date": entry_date, "exit_date": exit_date, "hold": hold, "early_exit": early,
               "name_ok": r.name_ok, "adv20": adv, "adv_bucket": adv_bucket(adv), "gross": gross,
               "cost": cost, "net": gross - cost, "net_2x_cost": gross - 2 * cost,
               "net_delist_stress": ((1 + gross) * (1 + DELIST_HAIRCUT) - 1 - cost) if early else gross - cost}
        for label, sym in (("bench_primary", bench_symbol(adv)), ("bench_iwm", "IWM")):
            b = benches[sym]
            bi = int(b["date"].searchsorted(entry_date))
            bx = int(b["date"].searchsorted(exit_date))
            if bi >= len(b) or bx >= len(b) or b["date"].iloc[bi] != entry_date or b["date"].iloc[bx] != exit_date:
                row[label] = np.nan
            else:
                row[label] = float(b["adjusted_close"].iloc[bx] / b["adj_open"].iloc[bi] - 1.0)
        row["bench_primary_sym"] = bench_symbol(adv)
        out.append(row)
    cols = ["issuer_cik", "ticker", "known_date", "entry_date", "exit_date", "hold", "early_exit",
            "name_ok", "adv20", "adv_bucket", "gross", "cost", "net", "net_2x_cost", "net_delist_stress",
            "bench_primary", "bench_iwm", "bench_primary_sym"]
    df = pd.DataFrame(out, columns=cols)
    for col in ("net", "net_2x_cost", "net_delist_stress"):
        df[f"xs_{col}"] = df[col] - df["bench_primary"]
    df["xs_net_iwm"] = df["net"] - df["bench_iwm"]
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def month_cluster_boot(values: np.ndarray, months: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Cluster bootstrap of the mean: resample calendar months with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(months, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    return sums[draws].sum(1) / counts[draws].sum(1)


def summarize(df: pd.DataFrame, col: str, seed: int) -> dict:
    v = df[col].to_numpy(dtype=float)
    ok = np.isfinite(v)
    v, months = v[ok], df.loc[ok, "entry_date"].dt.to_period("M").astype(str).to_numpy()
    if len(v) < 30:
        return {"n": len(v), "mean": float(np.mean(v)) if len(v) else float("nan")}
    boot = month_cluster_boot(v, months, prereg.BOOTSTRAP["n_boot"], seed)
    return {"n": len(v), "months": len(np.unique(months)), "mean": float(v.mean()),
            "median": float(np.median(v)), "hit_rate": float((v > 0).mean()),
            "lb": float(np.quantile(boot, ALPHA)), "ub": float(np.quantile(boot, 1 - ALPHA))}


def _positions(src: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Index of each ``src`` date in sorted ``target``, -1 where absent."""
    pos = np.searchsorted(target, src)
    ok = pos < len(target)
    ok[ok] = target[pos[ok]] == src[ok]
    return np.where(ok, pos, -1)


def calendar_time(df: pd.DataFrame, cache: dict, benches: dict, bench_col: str,
                  dates: pd.DatetimeIndex, prep: dict | None = None) -> pd.Series:
    """Daily equal-weight excess return of all open positions (net of costs).

    Day 1 is adj_open -> close, later days close -> close, for stock and benchmark
    alike; stock days without a benchmark bar (or with a missing price) count as
    zero excess. Per-ticker arrays are built once, so each event is a slice.
    """
    cal = dates.to_numpy()
    num = np.zeros(len(cal))
    cnt = np.zeros(len(cal))
    prep = {} if prep is None else prep   # shareable across calls with the same dates
    for r in df.itertuples(index=False):
        sym = r.bench_primary_sym if bench_col == "bench_primary" else "IWM"
        if r.ticker not in prep:
            d = cache[r.ticker]
            dd = d["date"].to_numpy()
            prep[r.ticker] = (dd, d["adjusted_close"].pct_change().to_numpy(),
                              (d["adjusted_close"] / d["adj_open"]).to_numpy() - 1,
                              _positions(dd, cal), {})
        dd, ret, day1, cal_pos, bmap = prep[r.ticker]
        if sym not in bmap:
            b = benches[sym]
            bret = b["adjusted_close"].pct_change().to_numpy()
            bday1 = (b["adjusted_close"] / b["adj_open"]).to_numpy() - 1
            bpos = _positions(dd, b["date"].to_numpy())
            bmap[sym] = (np.where(bpos >= 0, bret[bpos], np.nan), np.where(bpos >= 0, bday1[bpos], np.nan), bpos)
        bs_all, bday1_all, bpos = bmap[sym]
        e = int(np.searchsorted(dd, np.datetime64(r.entry_date)))
        x = int(np.searchsorted(dd, np.datetime64(r.exit_date)))
        if x - e + 1 < 2:
            continue
        if bpos[e] < 0:
            b = benches[sym]
            if ((b["date"] >= r.entry_date) & (b["date"] <= r.exit_date)).any():
                raise KeyError(f"{sym} has no bar on entry date {r.entry_date}")
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
        out = np.where(cnt > 0, num / cnt, 0.0)
    return pd.Series(out, index=dates)


def placebo(df: pd.DataFrame, cache: dict, benches: dict, hold: int, real_events: pd.DataFrame,
            n_draws: int, seed: int, last_market_date: pd.Timestamp) -> np.ndarray:
    """Mean primary excess return per draw: each evaluated event re-drawn at a random
    date of the same ticker (same filters, hold, costs, benchmark rule), excluding
    +/- 63 trading days around any real event of that ticker."""
    rng = np.random.default_rng(seed)
    per_ticker = df.groupby("ticker").size()
    real_idx = {}
    for t, g in real_events[real_events["status"] == "covered"].groupby("ticker"):
        real_idx[t] = g["entry_idx"].astype(int).to_numpy()
    prepared = {}
    for t in per_ticker.index:
        d = cache[t]
        n = len(d)
        e = np.arange(20, n - hold)                       # full hold must fit
        adv = d["dollar_vol"].rolling(20).median().shift(1).to_numpy()
        px = d["adj_open"].to_numpy()
        ok = (adv[e] >= ADV_MIN) & (adv[e] <= ADV_MAX) & (px[e] >= PRICE_MIN) & np.isfinite(px[e])
        for ri in real_idx.get(t, []):
            ok &= np.abs(e - ri) > 63
        cand = e[ok]
        if len(cand) == 0:
            continue
        a = cand
        entry_dates = d["date"].to_numpy()[a]
        exit_dates = d["date"].to_numpy()[a + hold - 1]
        gross = d["adjusted_close"].to_numpy()[a + hold - 1] / px[a] - 1
        advs = adv[a]
        cost = np.select([advs < 1e6, advs < 5e6, advs < 20e6], [120.0, 60.0, 30.0], 10.0) / 1e4
        bsyms = np.where(advs < 5e6, "IWC", np.where(advs < 20e6, "IWM", "IJH"))
        bench = np.full(len(a), np.nan)
        for sym in ("IWC", "IWM", "IJH"):
            msk = bsyms == sym
            if not msk.any():
                continue
            b = benches[sym]
            bd = b["date"].to_numpy()
            bi = np.searchsorted(bd, entry_dates[msk])
            bx = np.searchsorted(bd, exit_dates[msk])
            valid = (bi < len(bd)) & (bx < len(bd))
            vals = np.full(msk.sum(), np.nan)
            vi, vx = bi[valid], bx[valid]
            match = (bd[vi] == entry_dates[msk][valid]) & (bd[vx] == exit_dates[msk][valid])
            tmp = np.full(valid.sum(), np.nan)
            tmp[match] = b["adjusted_close"].to_numpy()[vx[match]] / b["adj_open"].to_numpy()[vi[match]] - 1
            vals[valid] = tmp
            bench[msk] = vals
        xs = gross - cost - bench
        keep = np.isfinite(xs)
        if keep.any():
            prepared[t] = xs[keep]
    tickers = [t for t in per_ticker.index if t in prepared]
    counts = np.array([per_ticker[t] for t in tickers])
    means = np.empty(n_draws)
    for k in range(n_draws):
        tot, n = 0.0, 0
        for t, c in zip(tickers, counts):
            pool = prepared[t]
            tot += pool[rng.integers(0, len(pool), size=c)].sum()
            n += c
        means[k] = tot / n if n else np.nan
    return means


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    ap.add_argument("--append-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if prereg.DRAFT:
        raise SystemExit("pre-registration is still DRAFT; freeze it first")
    fp = prereg.bars_fingerprint()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log.info("insider-cluster evaluation, prereg fingerprint %s", fp)

    benches = {s: load_series(EODHD / "etfs" / f"{s}.parquet") for s in ("IWC", "IWM", "IJH")}
    last_market_date = benches["IWM"]["date"].iloc[-1]
    events, cache = build_events(last_market_date)
    events.to_parquet(out / "event_coverage.parquet")
    coverage = events["status"].value_counts().to_dict()
    log.info("coverage: %s", coverage)

    primary_start = pd.Timestamp(prereg.WINDOWS["primary_post_sample"]["start"])
    primary_end = pd.Timestamp(prereg.WINDOWS["primary_post_sample"]["end"])
    secondary_start = pd.Timestamp(prereg.WINDOWS["secondary_post_publication"]["start"])
    in_window = (events["status"] != "covered") | (
        (events.get("entry_date") >= primary_start) & (events.get("entry_date") <= primary_end))

    results: dict = {"fingerprint": fp, "coverage_all_events": coverage, "holds": {}}
    cal_dates = benches["IWM"]["date"]
    cal_dates = pd.DatetimeIndex(cal_dates[(cal_dates >= primary_start)])
    variants: dict[str, pd.Series] = {}
    prep: dict = {}
    all_rows = []
    for hname, hold in HOLDS.items():
        ev_h = events[in_window].copy()
        df = event_returns(ev_h, cache, benches, hold, last_market_date)
        df["hold_name"] = hname
        all_rows.append(df)
        strict = df[df["name_ok"] == True]
        res = {"n_evaluated_covered": len(df), "n_primary_strict": len(strict),
               "early_exit_rate": float(strict["early_exit"].mean()) if len(strict) else None}
        seed = prereg.BOOTSTRAP["seed"] + hold
        res["primary"] = summarize(strict, "xs_net", seed)
        res["iwm_benchmark"] = summarize(strict, "xs_net_iwm", seed + 1)
        res["cost_2x"] = summarize(strict, "xs_net_2x_cost", seed + 2)
        res["delist_stress"] = summarize(strict, "xs_net_delist_stress", seed + 3)
        res["post_2013"] = summarize(strict[strict["entry_date"] >= secondary_start], "xs_net", seed + 4)
        res["sensitivity_all_covered"] = summarize(df, "xs_net", seed + 5)
        res["gross_raw_mean"] = float(strict["gross"].mean())
        res["by_adv_bucket"] = {k: summarize(g, "xs_net", seed + 6) for k, g in strict.groupby("adv_bucket")}
        res["by_year"] = {int(y): float(g["xs_net"].mean()) for y, g in strict.groupby(strict["entry_date"].dt.year)}
        log.info("%s: placebo %d draws", hname, prereg.PLACEBO["n_draws"])
        pl = placebo(strict, cache, benches, hold, events, prereg.PLACEBO["n_draws"],
                     prereg.PLACEBO["seed"] + hold, last_market_date)
        pd.Series(pl).to_frame("placebo_mean_xs").to_parquet(out / f"placebo_{hname}.parquet")
        res["placebo_p95"] = float(np.nanpercentile(pl, prereg.PLACEBO["pass_percentile"]))
        res["placebo_median"] = float(np.nanmedian(pl))
        # calendar-time variants: 2 event sets x 2 benchmarks
        for set_name, sub in (("strict", strict), ("covered", df)):
            for bcol in ("bench_primary", "bench_iwm"):
                key = f"{hname}|{set_name}|{bcol}"
                variants[key] = calendar_time(sub, cache, benches, bcol, cal_dates, prep)
        results["holds"][hname] = res
        log.info("%s: n=%d mean excess %.4f LB %.4f UB %.4f | placebo p95 %.4f | 2x cost %.4f | "
                 "delist stress %.4f | post-2013 %.4f", hname, res["primary"]["n"], res["primary"]["mean"],
                 res["primary"].get("lb", float("nan")), res["primary"].get("ub", float("nan")),
                 res["placebo_p95"], res["cost_2x"]["mean"], res["delist_stress"]["mean"], res["post_2013"]["mean"])

    pd.concat(all_rows, ignore_index=True).to_parquet(out / "events_evaluated.parquet")
    cal = pd.DataFrame(variants)
    cal.to_parquet(out / "calendar_time.parquet")

    from firm.eval.overfitting import cscv_pbo, deflated_sharpe

    trial_sr = (cal.mean() / cal.std(ddof=1)).to_numpy()
    pbo = float(cscv_pbo(cal.to_numpy(), n_partitions=prereg.PBO["n_partitions"]))
    results["pbo"] = pbo
    results["calendar_time"] = {}
    for hname in HOLDS:
        key = f"{hname}|strict|bench_primary"
        x = cal[key].to_numpy()
        active = cal[key] != 0
        results["calendar_time"][hname] = {
            "ann_mean_excess": float(np.mean(x) * 252), "ann_vol": float(np.std(x, ddof=1) * math.sqrt(252)),
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
                    "tier": prereg.classify(bars, tier_d, tier_b),
                    "generated_at": datetime.now(UTC).isoformat(),
                    "prereg_at": prereg.PREREGISTERED_AT})

    # missing-data bias check (reported)
    ev = events.copy()
    ev["year"] = ev["known_date"].dt.year
    results["missing_data_by_year"] = (ev.assign(covered=ev["status"] == "covered")
                                       .groupby("year")["covered"].mean().round(4).to_dict())
    (out / "report.json").write_text(json.dumps(results, indent=2, default=float))
    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=2, default=float))
    if args.append_ledger:
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"family": "insider_cluster", "entries": []}
        ledger["entries"].append({"date": datetime.now(UTC).date().isoformat(), "fingerprint": fp,
                                  "n_trials": len(trial_sr), "trials": list(cal.columns),
                                  "trial_daily_sharpes": [float(v) for v in trial_sr],
                                  "tier": results["tier"]})
        ledger["cumulative_trials"] = sum(e["n_trials"] for e in ledger["entries"])
        LEDGER.write_text(json.dumps(ledger, indent=2))
    log.info("bars %s tierD %s tierB %s => TIER %s; PBO %.3f", bars, tier_d, tier_b, results["tier"], pbo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
