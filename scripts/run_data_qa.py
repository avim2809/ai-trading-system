"""Owner-triggered data QA over the ETF universe (ticket P2-03). Run niced; not a backtest, not a ledger trial.

  nice -n 10 python scripts/run_data_qa.py --manifest research/data_manifests/<id>.json [--asof 2026-09-30]

Series are loaded ONLY through ``firm.data.etf_loader.load_etf_universe`` (which uses ``firm.research.data_access``);
this script never reads manifest paths or the EODHD store itself, and takes only ``snapshot_id`` from the manifest JSON.
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

import firm.research  # noqa: F401  (seal / wrapped entry-point check)
from firm.data import qa
from firm.data.etf_loader import load_etf_universe_with_report
from firm.research import seal

log = logging.getLogger("run_data_qa")


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
    snapshot = json.loads(a.manifest.read_text(encoding="utf-8"))["snapshot_id"]
    series, rep = load_etf_universe_with_report(a.universe, asof=asof, data_root=a.data_root)
    findings = qa.run_all({k: v.bars for k, v in series.items()}, asset="etf")
    counts = collections.Counter((f.check, f.severity) for f in findings)
    report = {
        "snapshot_id": snapshot, "asof": asof.isoformat(), "n_symbols": len(series), "missing_histories": rep["missing"],
        "exemption": "reporting run, not a backtest: not a ledger trial; data read only via firm.research.data_access",
        "counts": {f"{c}/{s}": n for (c, s), n in sorted(counts.items())},
        "findings": [{**f.__dict__, "date": f.date.isoformat() if f.date else None} for f in findings],
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{snapshot}.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# Data QA {snapshot[:12]}", "", f"asof {asof}; {len(series)} symbols; {report['exemption']}.", "",
             "| check/severity | count |", "|---|---|", *[f"| {k} | {v} |" for k, v in report["counts"].items()], "",
             f"Missing histories: {', '.join(rep['missing']) or 'none'}", ""]
    (a.out / f"{snapshot}.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("report written to %s", a.out / f"{snapshot}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
