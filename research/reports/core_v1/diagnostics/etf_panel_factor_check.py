"""Read-only diagnostic of the real ETF panel behind the core_v1 gate-7 anomaly (w24a). NO engine, NO backtest, NO ledger.

Builds the panel with the evaluation's own loaders (``firm.data.etf_loader`` + ``core_v1_pipeline.panel_from_series``, snapshot
asof 2026-09-30, all reads through ``firm.research.data_access``) and checks, per ETF and per window (full, 2005-01-03..2026-09-29):
(a) raw ``close`` jumps and raw-vs-adjusted splits; (b) adj/raw factor steps against the dividend table; (c) CAGR of the provider
adjusted path vs the path reinvested from raw close + dividend table. Writes ``etf_panel_factor_check.json`` next to this file.

    FIRM_RESEARCH_LEDGER_ROOT=<tmp> PYTHONPATH=$PWD/src python research/reports/core_v1/diagnostics/etf_panel_factor_check.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("FIRM_RESEARCH_LEDGER_ROOT", "/nonexistent-diagnostic-no-ledger")

from firm.data.etf_loader import load_dividends, load_etf_universe, usd_ils
from firm.research import core_v1_pipeline as P
from firm.research import data_access, seal

HERE = Path(__file__).resolve().parent
MANIFEST = ROOT / "research/data_manifests/etf_ab5d0a987833b9e7f24bf1cf7d3a5d01661b5c8a4a9b843feaa03b1829491a0f.json"
WINDOWS = {"full": (None, None), "2005+": ("2005-01-03", "2026-09-29")}
STEP_TOL = 5e-4      # price rounding noise floor on a factor step
JUMP = 0.20


def sha(path: Path) -> str:
    return hashlib.sha256(data_access.assert_path_allowed(path).read_bytes()).hexdigest()


def cagr(path: pd.Series) -> float:
    path = path.dropna()
    years = (path.index[-1] - path.index[0]).days / 365.25
    return float((path.iloc[-1] / path.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def analyse(sym: str, bars: pd.DataFrame, div: pd.DataFrame, lo, hi) -> dict:
    b = bars[["close", "adjusted_close"]].astype(float).sort_index()
    if lo:
        b = b.loc[(b.index >= pd.Timestamp(lo)) & (b.index <= pd.Timestamp(hi))]
    raw, adj = b["close"], b["adjusted_close"]
    seg = bars["segment"].reindex(b.index)
    same = seg.eq(seg.shift())             # within-segment pairs only (cleaning gaps are not returns)
    rr = (raw / raw.shift() - 1).where(same)
    f = adj / raw
    step = (f / f.shift() - 1).where(same)
    d = div.set_index("date")["amount"].groupby(level=0).sum() if len(div) else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    d = d[(d.index >= b.index[0]) & (d.index <= b.index[-1])]
    in_idx = d.index.isin(b.index)
    prev_raw = raw.shift()
    # expected factor step on an ex-date: back-adjustment convention P_{t-1}/(P_{t-1}-D) - 1 ; tax-model convention D/P_t
    exp_prev = {t: float(a / (prev_raw[t] - a) ) for t, a in d[in_idx].items() if pd.notna(prev_raw[t]) and prev_raw[t] > a}
    exp_cur = {t: float(a / raw[t]) for t, a in d[in_idx].items()}
    jumps = [(t.date().isoformat(), float(v)) for t, v in rr.items() if pd.notna(v) and abs(v) > JUMP]
    big = step[step.abs() > 0.05]
    splits = [(t.date().isoformat(), float(v)) for t, v in big.items() if t not in d.index]
    # factor steps on non-dividend days larger than noise (also +-1 day of an ex-date counts as matched)
    ex_days = set(d.index)
    pos = {t: i for i, t in enumerate(b.index)}
    near = set()
    for t in ex_days:
        if t in pos:
            for k in (-1, 0, 1):
                if 0 <= pos[t] + k < len(b.index):
                    near.add(b.index[pos[t] + k])
    thr = step.abs() > STEP_TOL
    steps_no_div = [(t.date().isoformat(), float(v)) for t, v in step[thr].items() if t not in near]
    unmatched, offset1, amt_diff = [], 0, []
    for t, e in exp_prev.items():
        win = [b.index[pos[t] + k] for k in (0, -1, 1) if 0 <= pos[t] + k < len(b.index)]
        got = [(w, step[w]) for w in win if pd.notna(step[w]) and abs(step[w] - e) <= STEP_TOL + 0.15 * e]
        if not got:
            unmatched.append((t.date().isoformat(), float(d[t]), e, float(step[t]) if pd.notna(step[t]) else None))
            continue
        offset1 += int(got[0][0] != t)
        amt_diff.append(float(got[0][1] - e))
    not_in_index = [t.date().isoformat() for t in d.index[~in_idx]]
    years = (b.index[-1] - b.index[0]).days / 365.25
    table_log = float(sum(np.log1p(v) for v in exp_prev.values()))
    factor_log = float(np.log(f.iloc[-1] / f.iloc[0]))   # includes splits if any
    # reinvested-from-raw path (tax model drift: shares *(1 + D/P_t) on ex-date)
    gro = (1 + rr.fillna(0.0))
    div_ret = pd.Series(0.0, index=b.index)
    for t, e in exp_cur.items():
        div_ret[t] = e
    wealth_div = ((gro + div_ret).cumprod())
    cag_adj, cag_rein, cag_raw = cagr(adj), cagr(wealth_div), cagr(raw)
    return {
        "bars": len(b), "first": b.index[0].date().isoformat(), "last": b.index[-1].date().isoformat(), "years": round(years, 3),
        "raw_jumps_gt20pct": jumps, "splits_unadjusted_or_factor_jumps_gt5pct": splits,
        "ex_dates": len(d), "ex_dates_not_trading_days": not_in_index, "ex_dates_unmatched": unmatched,
        "ex_dates_matched_offset_1day": offset1, "factor_steps_without_dividend": steps_no_div,
        "n_steps_without_dividend": len(steps_no_div),
        "matched_step_minus_expected_mean": float(np.mean(amt_diff)) if amt_diff else None,
        "matched_step_minus_expected_sum": float(np.sum(amt_diff)) if amt_diff else 0.0,
        "table_div_yield_per_year": table_log / years if years else None,
        "factor_drift_per_year": factor_log / years if years else None,
        "cagr_adjusted": cag_adj, "cagr_raw_price": cag_raw, "cagr_raw_plus_div_reinvested": cag_rein,
        "annual_mismatch_pts": 100 * (cag_adj - cag_rein),
    }


def split_events(series: dict, syms: list[str]) -> list[dict]:
    """Every raw-close split the adjusted series absorbs (factor step > 5% off a dividend day) with the 252d vol before it."""
    ev = []
    for s in syms:
        b = series[s].bars.sort_index()
        f = b["adjusted_close"] / b["close"]
        step = f / f.shift() - 1
        r = b["adjusted_close"].pct_change(fill_method=None)
        for t, v in step[step.abs() > 0.05].items():
            vol = float(r.loc[:t].iloc[:-1].tail(252).std() * np.sqrt(252))
            ev.append({"symbol": s, "date": t.date().isoformat(), "split_ratio_k": round(1 + v), "vol252_before": vol})
    return ev


def split_loss_estimate(events: list[dict], eng: dict) -> dict:
    """ESTIMATE (not a re-run). A split the raw close does not show makes the tax model hold too few shares: it buys the rest with cash at
    the post-split price, a permanent USD loss of N_i (1 - 1/k) with N_i the position value that day (no engine positions are read).
    N_i/equity ~ tau * w_i * IDM * FDM_i / vol_i * vol_scale_mean (forecast |f|/10 = 1, all long: an upper bound on sign) = case A at the
    frozen tau, case B scaled by realised/target vol of the stored engine series. The loss is a fixed USD amount, so its CAGR drag is
    -ln(1 - loss/E_end)/years with E_end/E_t from the stored engine CAGR."""
    c = json.loads((ROOT / "research/reports/core_v1/constants.json").read_text())
    tau = 0.09
    w, fdm, idm, vs = c["instrument_weights"], c["fdm"], c["idm"], 1.327134883513775   # vs: results.json sizing mean
    real_ratio = eng["2005+"]["vol_ann"] / tau
    cagr = eng["2005+"]["cagr_usd"]
    t0, t1 = pd.Timestamp("2005-01-03"), pd.Timestamp("2026-09-29")
    res = {"tau": tau, "realised_over_target_vol": real_ratio, "events": [], "drag_pts_per_year": {}}
    loss_a = {}
    for e in events:
        d = pd.Timestamp(e["date"])
        if d < t0 or e["split_ratio_k"] < 2:
            continue
        n = tau * w[e["symbol"]] * idm * fdm[e["symbol"]] / e["vol252_before"] * vs
        loss = n * (1 - 1 / e["split_ratio_k"])
        yrs_after = (t1 - d).days / 365.25
        frac_end = loss / (1 + cagr) ** yrs_after              # loss relative to the END equity
        loss_a[e["symbol"] + e["date"]] = frac_end
        res["events"].append({**e, "notional_over_equity_A": n, "loss_over_equity_A": loss, "loss_over_end_equity_A": frac_end})
    yrs = (t1 - t0).days / 365.25
    for case, k in (("A_upper_all_long_tau_delivered", 1.0), ("B_scaled_to_realised_vol", real_ratio)):
        tot = min(sum(v * k for v in loss_a.values()), 0.99)
        res["drag_pts_per_year"][case] = float(-np.log(1 - tot) / yrs * 100)
    return res


SELECTED_TRIAL = "f9f74dc1f4fc48898f3a3d7ae64c249d"   # results.json trial_ids['grid:grid:4'] (selected_index 4), committed ledger returns, READ ONLY


def engine_returns_window(manifest_fx: pd.Series) -> dict:
    """Read (not re-run) the committed engine return series of the selected trial through data_access; CAGR by window, USD and ILS."""
    path = Path(os.environ.get("RESEARCH_LEDGER_RETURNS", "/local/store/research-ledger/returns")) / f"{SELECTED_TRIAL}.parquet"
    df = data_access.read_parquet(path, asof=seal.max_research_date())
    r = df["returns"].astype(float)
    r.index = pd.DatetimeIndex(pd.to_datetime(df.index if "date" not in df.columns else df["date"]))
    res = {"file_sha256": sha(path), "n": len(r), "first": r.index[0].date().isoformat(), "last": r.index[-1].date().isoformat()}
    fx = manifest_fx.sort_index()
    for w, (lo, hi) in WINDOWS.items():
        x = r if lo is None else r.loc[lo:hi]
        eq = (1 + x.fillna(0)).cumprod()
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        out = {"days": len(x), "cagr_usd": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol_ann": float(x.std() * np.sqrt(252)),
               "sharpe_ann": float(x.mean() / x.std() * np.sqrt(252))}
        if lo is not None:
            f = fx.reindex(eq.index, method="ffill")
            out["cagr_ils_of_usd_path"] = float((eq * f / f.iloc[0]).iloc[-1] ** (1 / yrs) - 1)   # ILS value of the USD equity path
            out["fx_cagr_usdils"] = float((f.iloc[-1] / f.iloc[0]) ** (1 / yrs) - 1)
        res[w] = out
    return res


def main() -> int:
    asof = seal.max_research_date()
    seal.check_asof(asof, what="etf_panel_factor_check")
    data_root = ROOT / "data/research/eodhd"
    uni = ROOT / "config/universe_etf.yaml"
    import yaml
    syms = [str(i["symbol"]) for i in yaml.safe_load(uni.read_text())["instruments"]]
    series = load_etf_universe(uni, asof=asof, include_delisted=False)
    divs = load_dividends(syms, asof=asof)
    panel = P.panel_from_series({s: series[s] for s in syms}, snapshot_id="diag", dividends=divs, fx=None, source="diag")
    # the panel the engine saw must equal the bars on real bar dates (the rest is the documented back/forward fill)
    for s in syms:
        bb = series[s].bars
        assert np.allclose(panel.close.loc[bb.index, s], bb["adjusted_close"]) and np.allclose(panel.raw_close.loc[bb.index, s], bb["close"])
    manifest = json.loads(MANIFEST.read_text())
    man = {r["path"]: r["sha256"] for r in manifest["rows"]}
    hashes, bad = {}, []
    for rel, h in man.items():
        if rel.endswith(".parquet") and (rel.startswith("etfs_full/") or "dividends/" in rel) and Path(rel).stem in syms:
            got = sha(data_root / rel)
            hashes[rel] = got
            if got != h:
                bad.append(rel)
    out = {"asof": asof.isoformat(), "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
           "input_sha256": hashes, "hash_mismatch_vs_manifest": bad, "windows": {}}
    for w, (lo, hi) in WINDOWS.items():
        out["windows"][w] = {s: analyse(s, series[s].bars, divs[divs["symbol"] == s], lo, hi) for s in syms}
    out["split_events"] = split_events(series, syms)
    out["engine_returns_readonly"] = engine_returns_window(manifest_fx=usd_ils(asof, source="eodhd"))
    out["split_loss_estimate"] = split_loss_estimate(out["split_events"], out["engine_returns_readonly"])
    (HERE / "etf_panel_factor_check.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print("manifest", out["manifest_sha256"], "| input files hashed:", len(hashes), "| mismatches vs manifest:", bad)
    for rel, h in sorted(hashes.items()):
        print(" ", h, rel)
    for w in WINDOWS:
        print(f"\n== window {w}")
        print("sym   jumps splits exD unmatched steps_nodiv  cagr_adj  cagr_reinv  mismatch_pts")
        for s, r in out["windows"][w].items():
            print(f"{s:5} {len(r['raw_jumps_gt20pct']):5} {len(r['splits_unadjusted_or_factor_jumps_gt5pct']):6} {r['ex_dates']:4} "
                  f"{len(r['ex_dates_unmatched']):9} {r['n_steps_without_dividend']:10} {r['cagr_adjusted']*100:9.3f} "
                  f"{r['cagr_raw_plus_div_reinvested']*100:10.3f} {r['annual_mismatch_pts']:12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
