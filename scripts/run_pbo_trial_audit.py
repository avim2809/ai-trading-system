#!/usr/bin/env python
"""Audit walk-forward runs for formal PBO / deflated-Sharpe overfitting stats.

Reads completed fold runs from the experiment registry and delegates to
:class:`firm.experiments.runner.ExperimentRunner` (which loads ``equity.json``
and ``walk_forward_selection.json`` per fold). Use this after a walk-forward
with a real ``param_grid`` (>=2 candidates) so PBO/DSR reflect genuine
competing trials, not sequential OOS folds of one config.

Examples:
    # Audit the three most recent completed runs (must be walk-forward folds)
    python scripts/run_pbo_trial_audit.py

    # Audit explicit fold run ids
    python scripts/run_pbo_trial_audit.py --fold-ids id1 id2 id3

    # Custom registry directory
    python scripts/run_pbo_trial_audit.py --runs-dir /path/to/runs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.experiments.registry import RunRegistry
from firm.experiments.runner import ExperimentRunner

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        default="runs",
        help="Experiment registry base directory (default: runs)",
    )
    p.add_argument(
        "--fold-ids",
        nargs="*",
        help="Explicit fold run ids to audit (default: latest completed runs)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="When auto-selecting folds, take this many latest completed runs",
    )
    return p.parse_args()


def _select_runs(registry: RunRegistry, fold_ids: list[str] | None, limit: int):
    if fold_ids:
        runs = []
        for rid in fold_ids:
            run = registry.get_run(rid)
            if run is None:
                log.error("run not found: %s", rid)
                raise SystemExit(1)
            runs.append(run)
        return runs

    completed = [r for r in registry.list_runs() if r.status == "completed"]
    if not completed:
        log.error("no completed runs in %s", registry.base_dir)
        raise SystemExit(1)
    completed.sort(key=lambda r: r.start_time)
    return completed[-limit:]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    registry = RunRegistry(base_dir=args.runs_dir)
    runs = _select_runs(registry, args.fold_ids, args.limit)

    log.info("Auditing %d fold run(s)", len(runs))
    aggregate = ExperimentRunner(registry=registry).aggregate_walk_forward(runs)
    overfit = aggregate.get("overfitting") or {}

    print(json.dumps({"fold_ids": [r.run_id for r in runs], **aggregate}, indent=2, default=str))

    if not overfit:
        log.warning(
            "No overfitting block produced — need >=2 completed folds with "
            "equity.json; PBO/DSR additionally need walk_forward_selection.json "
            "from a walk-forward with param_grid >= 2 candidates"
        )
        return 1
    log.info(
        "PBO=%s DSR=%s verdict=%s",
        overfit.get("pbo"),
        overfit.get("deflated_sharpe"),
        overfit.get("verdict"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
