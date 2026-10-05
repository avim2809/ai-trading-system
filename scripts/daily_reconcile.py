"""Daily fidelity reconcile CLI for candidates (P5-04). Alerts only; never halts, never places orders.

    run      one daily evaluation; needs --fetcher module:callable returning the fetch_live payload (see firm.monitoring.fidelity.daily_job).
             No live fetcher exists until a candidate instance is provisioned (P6-03 / OD-11 / OD-15); fixtures only before that.
    report   print the latest daily result from the state dir
    export   OWNER-RUN: copy the latest result into the candidate's sealed monitoring directory (never by a timer)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from firm.monitoring import fidelity as F


def _default_state(candidate: str) -> Path:
    return F._STATE_ROOT / candidate


def _notify(**kw) -> None:  # alert sink: stdout only (webhook wiring belongs to the P6 timer unit)
    print(json.dumps({"alert": kw}, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report", "export"):
        p = sub.add_parser(name)
        p.add_argument("--candidate", required=True)
        p.add_argument("--state-dir", type=Path)
        if name == "run":
            p.add_argument("--fetcher", required=True, help="module:callable returning the fetch_live payload")
        if name == "export":
            p.add_argument("--out-root", type=Path, help="default: the sealed candidates directory")
    a = ap.parse_args(argv)
    state = a.state_dir or _default_state(a.candidate)
    if a.cmd == "run":
        mod, _, fn = a.fetcher.partition(":")
        res = F.daily_job(a.candidate, state, fetch_live=getattr(importlib.import_module(mod), fn), notify=_notify)
        print(json.dumps({"breach": res["breach"], "report": res["report"]["g_paper_3_status"]}))
        return 1 if res["breach"] else 0
    if a.cmd == "report":
        print((state / "state.json").read_text())
        return 0
    print(F.export_report(a.candidate, state, a.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
