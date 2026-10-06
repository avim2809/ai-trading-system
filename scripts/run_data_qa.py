"""Owner-triggered data QA over the ETF universe (ticket P2-03). Run niced; not a backtest, not a ledger trial.

  nice -n 10 python scripts/run_data_qa.py --manifest research/data_manifests/<id>.json [--asof 2026-09-30]

Series are loaded ONLY through ``firm.data.etf_loader.load_etf_universe`` (which uses ``firm.research.data_access``);
the extra raw-bar / SPY-calendar / USDILS checks also read only via ``firm.research.data_access.read_parquet`` (never raw
``pd.read_parquet``); it never reads manifest paths and takes only ``snapshot_id`` from the manifest JSON.
It imports ``firm.research`` for the seal / wrapping check; QA runs are reporting, so they are exempt from the trial
ledger and the report states that. Output: research/data_qa/<snapshot_id>.json and .md.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import firm.research  # noqa: F401  (seal / wrapped entry-point check)
from firm.data import qa
from firm.data.etf_loader import DEFAULT_DATA_ROOT, load_etf_universe_with_report
from firm.research import data_access, seal

log = logging.getLogger("run_data_qa")


# Flags the owner verified against a second source (annotation only: nothing is cleaned, dropped or adjusted because of it).
VERIFIED_REAL = {
    ("spike", "SPY", "2008-10-13"): "owner-verified real: the SPY 2008-10-10 vendor close matches Yahoo, so the +14.5% return into 2008-10-13 is genuine",
}


def _verified(f) -> str | None:
    return VERIFIED_REAL.get((f.check, f.symbol, f.date.isoformat() if f.date else None))


def _findings_json(fs) -> list[dict]:
    return [{**f.__dict__, "date": f.date.isoformat() if f.date else None, **({"verified_real": v} if (v := _verified(f)) else {})} for f in fs]


def _ret_note(series, f) -> str:
    """Diagnostic for a spike flag: adjusted-close return vs raw-close return that day (a gap between them = dividend adjustment)."""
    if f.check != "spike" or f.symbol not in series or f.date is None:
        return ""
    b = series[f.symbol].bars
    ts = pd.Timestamp(f.date)
    seg = b["segment"] if "segment" in b.columns else pd.Series(0, index=b.index)
    adj = b["adjusted_close"].groupby(seg).pct_change(fill_method=None)
    cl = b["close"].groupby(seg).pct_change(fill_method=None)
    return f" [adj ret {adj.get(ts, float('nan')):+.2%}, close ret {cl.get(ts, float('nan')):+.2%}]"


def _spy_calendar_acceptance(root: Path, asof: dt.date) -> dict:
    """SPY raw bar dates versus ``expected_trading_days`` over the whole SPY history (both directions). Read-only."""
    spy = data_access.read_parquet(root / "etfs_full" / "SPY.parquet", asof=asof)
    obs = pd.DatetimeIndex(pd.to_datetime(spy["date"])).normalize()
    exp = qa.expected_trading_days(obs.min().date(), obs.max().date())  # first..last SPY bar
    absent = exp.difference(obs)
    extra = obs.difference(exp)
    tail = qa.expected_trading_days(obs.max().date() + dt.timedelta(days=1), asof) if obs.max().date() < asof else pd.DatetimeIndex([])
    return {"first": obs.min().date().isoformat(), "last": obs.max().date().isoformat(), "n_bars": len(obs), "n_expected": len(exp),
            "expected_sessions_after_last_bar_up_to_asof": [d.date().isoformat() for d in tail],
            "expected_but_absent": [d.date().isoformat() for d in absent], "bars_on_non_expected_days": [d.date().isoformat() for d in extra],
            "pass": bool(len(absent) == 0 and len(extra) == 0)}


def _fx_checks(root: Path, asof: dt.date) -> dict:
    """USD/ILS (EODHD; the Bank of Israel series is not in allow_roots): positivity, >10% daily moves, inverse-pair consistency."""
    usd_ils = data_access.read_parquet(root / "forex" / "USDILS.parquet", asof=asof)
    s = pd.Series(usd_ils["close"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(usd_ils["date"])))
    out = {"n": len(s), "first": s.index.min().date().isoformat(), "last": s.index.max().date().isoformat(),
           "duplicate_dates": int(s.index.duplicated().sum()), "findings": []}
    fs = qa.check_fx_conversion(pd.Series(1.0, index=s.index), s, s, symbol="USDILS")  # positivity and >10% day moves only
    out["findings"] = _findings_json(f for f in fs if "differs" not in f.detail)
    inv_path = root / "forex" / "ILSUSD.parquet"
    if inv_path.exists():
        inv = data_access.read_parquet(inv_path, asof=asof)
        i = pd.Series(inv["close"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(inv["date"])))
        j = pd.concat({"u": s, "i": i}, axis=1, join="inner")
        err = (j["u"] * j["i"] - 1.0).abs()
        out["inverse_pair"] = {"n_common": len(j), "max_abs_dev_from_1": float(err.max()), "n_over_1pct": int((err > 0.01).sum()),
                               "dates_over_1pct": [d.date().isoformat() for d in err.index[(err > 0.01).to_numpy()][:20]]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--universe", type=Path, default=ROOT / "config" / "universe_etf.yaml")
    ap.add_argument("--asof", type=dt.date.fromisoformat, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "data_qa")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    asof = a.asof or seal.max_research_date()
    seal.check_asof(asof, what="run_data_qa")
    root = Path(a.data_root) if a.data_root is not None else DEFAULT_DATA_ROOT
    snapshot = json.loads(a.manifest.read_text(encoding="utf-8"))["snapshot_id"]
    series, rep = load_etf_universe_with_report(a.universe, asof=asof, data_root=a.data_root)
    findings = qa.run_all({k: v.bars for k, v in series.items()}, asset="etf")
    counts = collections.Counter((f.check, f.severity) for f in findings)
    # Cross-check on the RAW (uncleaned) bars: cleaning may have dropped bars that QA on cleaned data cannot see. Reporting only.
    raw = {k: data_access.read_parquet(root / "etfs_full" / f"{k}.parquet", asof=asof) for k in series}
    raw_findings = qa.run_all(raw, asset="etf")
    cleaned_keys = {(f.check, f.symbol, f.date) for f in findings}
    raw_only = [f for f in raw_findings if f.severity != "info" and (f.check, f.symbol, f.date) not in cleaned_keys]
    report = {
        "snapshot_id": snapshot, "asof": asof.isoformat(), "n_symbols": len(series), "symbols": sorted(series),
        "n_missing_histories": len(rep["missing"]), "missing_histories": rep["missing"],
        "exemption": "reporting run, not a backtest: not a ledger trial; data read only via firm.research.data_access",
        "no_post_seal_data_read": True,
        "counts": {f"{c}/{s}": n for (c, s), n in sorted(counts.items())},
        "clean_reports": {k: v.clean_report for k, v in series.items()},
        "findings": _findings_json(findings),
        "raw_only_findings": _findings_json(raw_only),
        "spy_calendar_acceptance": _spy_calendar_acceptance(root, asof),
        "fx_usdils": _fx_checks(root, asof),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{snapshot}.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    acc = report["spy_calendar_acceptance"]
    lines = [f"# Data QA {snapshot[:12]}", "", f"asof {asof}; {len(series)} universe symbols; {report['exemption']}.",
             ("No post-seal data was read. QA reports only: nothing was cleaned, dropped or adjusted by this run (the loader's own "
              "cleaning v3 is the one in `clean_reports`)."), "",
             "## Counts (cleaned bars)", "", "| check/severity | count |", "|---|---|", *[f"| {k} | {v} |" for k, v in report["counts"].items()], "",
             "## Flags for the owner (cleaned bars)", "", "| check | symbol | date | detail |", "|---|---|---|---|",
             *[f"| {f.check} | {f.symbol} | {f.date} | {(f.detail + _ret_note(series, f) + (f" **VERIFIED REAL: {_verified(f)}**" if _verified(f) else "")).replace(chr(124), chr(92) + chr(124))} |" for f in findings if f.severity != "info"], "",
             f"## Raw-bar cross-check: findings present on raw bars but not on cleaned bars: {len(raw_only)}", "",
             *[f"- {f.check} {f.symbol} {f.date} [{f.severity}] {f.detail}" for f in raw_only], "",
             "## SPY bar dates vs exchange calendar (1993-2026)", "",
             (f"{acc['first']}..{acc['last']}: {acc['n_bars']} bars, {acc['n_expected']} expected sessions; "
              f"expected-but-absent {len(acc['expected_but_absent'])}, bars-on-non-expected-days {len(acc['bars_on_non_expected_days'])}; "
              f"**{'PASS' if acc['pass'] else 'FAIL'}** (first..last bar). Expected sessions after the last bar up to asof: "
              f"{', '.join(acc['expected_sessions_after_last_bar_up_to_asof']) or 'none'}."), "",
             *[f"- absent: {d}" for d in acc["expected_but_absent"][:50]], *[f"- extra: {d}" for d in acc["bars_on_non_expected_days"][:50]], "",
             "## USD/ILS (EODHD; Bank of Israel il_macro not available)", "", "```", json.dumps(report["fx_usdils"], indent=1, default=str), "```", "",
             (f"## Missing histories: {len(rep['missing'])} delisted ETF codes have no downloaded file (full list in the JSON); "
              f"first 20: {', '.join(rep['missing'][:20]) or 'none'}"), ""]
    (a.out / f"{snapshot}.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("report written to %s", a.out / f"{snapshot}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
