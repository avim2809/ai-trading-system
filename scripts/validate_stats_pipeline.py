#!/usr/bin/env python3
"""P1-08: synthetic GARCH-t validation of the statistics pipeline (size / power / PBO / DSR at pre-set rates).

    nice -n 10 ionice -c3 python scripts/validate_stats_pipeline.py --out research/validation [--quick] [--resume]

Every scenario parameter and acceptance bar is READ from the frozen ``p1_08_acceptance`` block of config/gates.yaml
(plus the two values it defers, from plan/drafts/P1-08/scenario_freeze.yaml). ``--sims``/``--seed`` that differ from the
frozen values are refused; ``--quick`` (reduced smoke run) is labelled NON-EVIDENCE in the report. Research-venv only;
never imported by a live module. Synthetic trials go to a THROWAWAY ledger root; the host ledger is checked unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firm.validation import synthetic as syn

log = logging.getLogger("validate_stats_pipeline")

HOST_LEDGER = Path("/local/store/research-ledger/trials.jsonl")
# 09:15 ET planning cycle = 13:15 UTC (EDT); heavy runs wait outside this UTC window (minutes since midnight).
AVOID_UTC = (12 * 60 + 45, 14 * 60)


def _gates_hash(path: Path) -> str:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.pop("meta", None)
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _host_ledger_state() -> str:
    try:
        if not HOST_LEDGER.exists():
            return "absent"
        raw = HOST_LEDGER.read_bytes()
        last = raw.rstrip(b"\n").rsplit(b"\n", 1)[-1] if raw else b""
        return f"size={len(raw)} last_sha256={hashlib.sha256(last).hexdigest()}"
    except OSError as exc:
        return f"unreadable({exc.__class__.__name__})"


def _git(*a: str) -> str:
    return subprocess.run(["git", "-c", "safe.directory=*", *a], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def _wait_outside_window() -> None:
    while True:
        now = datetime.now(UTC)
        m = now.hour * 60 + now.minute
        if not AVOID_UTC[0] <= m < AVOID_UTC[1]:
            return
        log.info("inside the 09:15 ET planning window (UTC %02d:%02d); sleeping 60 s", now.hour, now.minute)
        time.sleep(60)


def _quick(s: syn.Scenario) -> syn.Scenario:
    return dataclasses.replace(s, n_sims=min(s.n_sims, 3), B=min(s.B, 50), n_days=min(s.n_days, 640),
                               k_trials=min(s.k_trials, 8), S=min(s.S, 8))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gates", type=Path, default=ROOT / "config" / "gates.yaml")
    ap.add_argument("--quick", action="store_true", help="smoke run, NON-EVIDENCE")
    ap.add_argument("--resume", action="store_true", help="reuse per-scenario checkpoints")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sims", type=int, default=None, help="refused unless equal to the frozen n_sims")
    ap.add_argument("--seed", type=int, default=None, help="refused unless equal to the frozen master_seed")
    args = ap.parse_args(argv)

    acc = syn.load_acceptance(args.gates)
    frozen_sims = {s.n_sims for s in acc["scenarios"]}
    if not args.quick:
        if args.sims is not None and args.sims not in frozen_sims:
            ap.error(f"--sims {args.sims} differs from the frozen n_sims; only --quick may shrink the run")
        if args.seed is not None and args.seed != acc["master_seed"]:
            ap.error(f"--seed {args.seed} differs from the frozen master_seed {acc['master_seed']}")
    if args.workers > 2:
        ap.error("at most 2 worker processes (host resource policy)")

    status = _git("status", "--porcelain", "--untracked-files=all")
    dirty = [ln for ln in status.splitlines() if not ln[3:].startswith(("runs/", "research/validation/"))]
    if dirty and not args.quick:
        log.error("dirty git tree; commit first (clean tree required for an evidence run): %s", dirty[:5])
        return 2

    date_utc = datetime.now(UTC).strftime("%Y%m%d")
    scenarios = [_quick(s) if args.quick else s for s in acc["scenarios"]]
    scratch = Path(tempfile.mkdtemp(prefix="synthetic_ledger_", dir=os.environ.get("TMPDIR") or tempfile.gettempdir()))
    (scratch / "returns").mkdir()
    os.environ["FIRM_RESEARCH_LEDGER_ROOT"] = str(scratch)
    os.environ.setdefault("FIRM_LEDGER_ALLOW_DIRTY", "1")  # exploratory rows only; recorded in the ledger provenance
    ckpt = Path(args.out) / ".checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    host_before = _host_ledger_state()
    t0 = time.time()
    results: dict = {}
    for s in scenarios:
        key = hashlib.sha256(json.dumps(dataclasses.asdict(s), sort_keys=True, default=str).encode()).hexdigest()[:16]
        cp = ckpt / f"{s.name}_{key}.json"
        if args.resume and cp.exists():
            results[s.name] = json.loads(cp.read_text())
            log.info("resumed %s from checkpoint", s.name)
            continue
        if not args.quick:
            _wait_outside_window()
        ts = time.time()
        log.info("scenario %s: K=%d T=%d sims=%d B=%d", s.name, s.k_trials, s.n_days, s.n_sims, s.B)
        results[s.name] = syn.run_scenario(s, workers=args.workers)
        results[s.name]["runtime_s"] = round(time.time() - ts, 1)
        cp.write_text(json.dumps(results[s.name], default=str))
        log.info("scenario %s done in %.0f s", s.name, time.time() - ts)

    host_after = _host_ledger_state()
    gates_sha = _gates_hash(args.gates)
    acc["gates_sha256"] = gates_sha
    failures = syn.evaluate(results, acc)
    if host_before != host_after:
        failures.append(f"HOST LEDGER CHANGED: before [{host_before}] after [{host_after}]")
    marginal = syn.find_marginal(results, acc)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    meta = {
        "date_utc": date_utc, "quick": args.quick, "non_evidence": args.quick,
        "git_commit": _git("rev-parse", "HEAD"), "git_tree_clean": not dirty,
        "gates_yaml_sha256_(meta removed)": gates_sha, "gates_sha256": gates_sha,
        "host_ledger_before": host_before, "host_ledger_after": host_after,
        "throwaway_ledger_root": str(scratch), "workers": args.workers,
        "runtime_s": round(time.time() - t0, 1), "children_user_cpu_s": round(ru.ru_utime, 1),
        "peak_rss_children_MB": round(ru.ru_maxrss / 1024, 1), "python": platform.python_version(),
        "master_seed": acc["master_seed"], "cross_correlation": acc["cross_correlation"],
        "base_seeds": {s.name: s.base_seed for s in scenarios},
    }
    meta["gates_yaml_sha256"] = gates_sha
    path = syn.write_report(results, failures, args.out, meta=meta, marginal=marginal, acceptance=acc)
    log.info("report: %s verdict=%s failures=%d marginal=%d", path, "PASS" if not failures else "FAIL", len(failures), len(marginal))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
