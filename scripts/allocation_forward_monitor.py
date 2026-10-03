"""Allocation forward-test monitor CLI (P5-06). Alerts only; never halts, never writes tracked paths.

    refresh-inputs  fetch the 15 replay input files into <state>/inputs (network; OD-15 unit only)
    run             daily I1-I5 evaluation (GET-only loopback NAV source)
    report          print the latest daily result
    validate        offline integrity checks (fingerprint, thresholds, I5 params, trial-history entries)
    export          OWNER-RUN: copy the latest result to --out (outside the repo unless --allow-in-repo)

Secrets come from os.environ only (TIINGO_API_KEY, FRED_API_KEY, ALERT_WEBHOOK_URL); never .env.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from firm.monitoring import allocation_forward as m  # noqa: E402

log = logging.getLogger("allocation_forward_monitor")
DEFAULT_STATE = _ROOT / "data" / "forward_monitors" / "allocation"


def _environ(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise SystemExit(f"{key} not set in the environment")
    return val


def cmd_refresh(state: Path) -> int:
    import alt_premia_data as d  # fetch functions only; its .env-reading _env() helper is never called

    tiingo, fred = _environ("TIINGO_API_KEY"), _environ("FRED_API_KEY")
    fetchers = {}
    for t in ("SPY", "IEF", "EFA", "EEM", "TLT", "GLD", "DBC", "VNQ", "SVXY"):
        fetchers[f"tiingo_{t}.parquet"] = lambda t=t: d.fetch_tiingo(t, tiingo)
    fetchers["tiingo_BTCUSD.parquet"] = lambda: d.fetch_tiingo_crypto("btcusd", tiingo)
    for s in ("VIX", "VIX3M", "PUT"):
        fetchers[f"cboe_{s}.parquet"] = lambda s=s: d.fetch_cboe(s)
    fetchers["fred_DTB3.parquet"] = lambda: d.fetch_fred("DTB3", fred)
    fetchers["fomc_scheduled.parquet"] = d.fetch_fomc
    m.refresh_inputs(state / "inputs", fetchers)
    return 0


def cmd_run(state: Path, base_url: str) -> int:
    client = m.GetOnlyClient(base_url)
    res = m.run_daily(state, live_nav_fetcher=lambda: m.fetch_live_snapshots(client), notify=m.make_notify())
    print(json.dumps({k: res[k] for k in ("status", "failed") if k in res}))
    return 0 if res["status"] in ("ok", "fail") else 2


def _latest(state: Path) -> Path:
    files = sorted((state / "daily").glob("*.json"))
    if not files:
        raise SystemExit("no daily results yet")
    return files[-1]


def cmd_validate() -> int:
    frozen = m.load_frozen()
    dep = m.read_deployment(_ROOT / "docs" / "allocation_forward_test_trial_history.json")
    checks = {
        "fingerprint_matches": frozen.bars_fingerprint() == dep.expected_fingerprint,
        "live_nav_rule_recorded": dep.rule_entry_ok,
        "i5_params": m.verify_frozen_params(m.load_replay_config(_ROOT / "config" / "live_alpaca.yaml"),
                                            frozen.PORTFOLIO).passed,
    }
    m.check_thresholds_against_frozen(frozen)
    print(json.dumps(checks))
    return 0 if all(checks.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["refresh-inputs", "run", "report", "validate", "export"])
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--allow-in-repo", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.command == "refresh-inputs":
        return cmd_refresh(a.state)
    if a.command == "run":
        return cmd_run(a.state, a.base_url)
    if a.command == "report":
        print(_latest(a.state).read_text())
        return 0
    if a.command == "validate":
        return cmd_validate()
    if a.out is None:
        raise SystemExit("export needs --out")
    out = a.out.resolve()
    if not a.allow_in_repo and (out == _ROOT or _ROOT in out.parents):
        raise SystemExit("--out is inside the repo; pass --allow-in-repo only if the path is gitignored/sealed")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_latest(a.state), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
