#!/usr/bin/env python
"""Workstream D: the single pre-registered walk-forward evaluation of
pattern_recognition's ML confirmation/meta-labeling layer, per
``docs/pattern_recognition_plan.md``'s Workstream D and the "Decision point"
section of ``/root/.claude/plans/typed-splashing-moonbeam.md``.

WHAT THIS ANSWERS
==================
Mean expectancy ~= 0 on the raw signal population does not by itself mean no
profitable subset exists -- finding the profitable minority of a zero-mean
population is exactly a classifier's job. This script asks, once, whether
the meta-labeling model identifies such a subset, on HONEST (post-Workstream-C
edge-triggered) labels, against four bars fixed in
``scripts/pattern_ml_preregistered_bars.py`` BEFORE this script is ever run
for real:

  1. Executable expectancy (R/trade, next-bar-open fill, risk = |fill-stop|)
     > 0, with a symbol-block bootstrap CI excluding zero, that ALSO beats a
     direction-flipped placebo (mirrored barriers, flipped direction).
  2. Deflated Sharpe Ratio > 0.95, deflated by the CUMULATIVE cross-session
     trial count (this run's grid + every prior session's, tracked in
     ``docs/pattern_ml_trial_history.json``), not just this run's own grid.
  3. PBO < 0.50, computed over >= 5 genuinely distinct candidates (a 2-
     candidate grid degenerates to a coin flip -- see cscv_pbo's docstring).
  4. Fold consistency >= 75% (the same bar/logic
     ``scripts/validate_pattern_cnn_walkforward.py``'s own
     ``derive_recommendation`` already enforces -- reused here, not
     reimplemented).

REUSE, NOT REBUILD
===================
- Portfolio-level (isolated single-strategy, full capital/risk budget) walk-
  forward mechanics: ``scripts/validate_pattern_cnn_walkforward.py``'s
  ``_build_config``/``ExperimentRunner.run_walk_forward``/
  ``derive_recommendation``/``_fold_flag_selection_pattern`` -- imported and
  reused directly, with ``_STRATEGIES`` overridden to ``["pattern_recognition"]``
  (the same one-line override the 2026-09-27 isolated evaluation used; see
  ``docs/pattern_ml_isolated_evaluation_2026_09.md`` section 3).
- Trade-level executable-expectancy/placebo/symbol-block-bootstrap mechanics:
  ``scripts/measure_pattern_executable_expectancy.py`` (Workstream B, already
  landed) -- ``scan_events``/``price_executable``/``price_placebo``/
  ``symbol_block_bootstrap`` reused directly, restricted here to the walk-
  forward's own out-of-sample test-window union and to the pre-registered
  meta-confidence threshold.
- Deflation-by-prior-trials: ``firm.eval.overfitting.deflated_sharpe``'s new
  ``prior_trials`` parameter (added alongside this script), threaded through
  ``ExperimentRunner.aggregate_walk_forward``.

ONE-SHOT ENFORCEMENT
=====================
Bars, target threshold, and full test design live in
``scripts/pattern_ml_preregistered_bars.py``, committed separately and
never edited after a real run -- see that module's own docstring. Every
invocation of this script (dry-run or real) appends to
``docs/pattern_ml_workstream_d_run_log.json`` (append-only); only a REAL
(non-``--dry-run``) run also appends to ``docs/pattern_ml_trial_history.json``,
since a dry run's models/results are not a genuine test of this ML layer
(see ``check_workstream_c_landed``). A real run additionally refuses to
proceed at all unless Workstream C's fix looks like it has actually landed
(``find_confirmation``'s fix committed, and the on-disk models postdate
that commit) -- pass ``--skip-c-check`` only to knowingly force a dry-run-
grade smoke test through the "real" code path (this does NOT flip
``dry_run`` off in the report, so a forced run still cannot silently
contaminate the trial-history ledger).

USAGE
=====
    # Fast smoke test against whatever models are currently on disk --
    # NOT a real verdict, clearly labeled as such in the report.
    python scripts/validate_pattern_ml_workstream_d.py --dry-run

    # The real, pre-registered evaluation -- refuses to run until
    # Workstream C's fix has landed (see check_workstream_c_landed).
    python scripts/validate_pattern_ml_workstream_d.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import measure_pattern_executable_expectancy as pit
import pattern_ml_preregistered_bars as bars_module
import validate_pattern_cnn_walkforward as cnn_wf

from firm.experiments.registry import RunRegistry
from firm.experiments.runner import ExperimentRunner

log = logging.getLogger("validate_pattern_ml_workstream_d")

TRIAL_HISTORY_PATH = _ROOT / "docs" / "pattern_ml_trial_history.json"
RUN_LOG_PATH = _ROOT / "docs" / "pattern_ml_workstream_d_run_log.json"
MODEL_PATHS = [
    _ROOT / "data" / "models" / "pattern_xgb.pkl",
    _ROOT / "data" / "models" / "pattern_xgb_meta.pkl",
]
CONFIRMATION_PY_REL = "src/firm/patterns/confirmation.py"


# ---------------------------------------------------------------------------
# Workstream C landed-check
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=_ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        log.warning("git %s failed", " ".join(args), exc_info=True)
        return ""


def check_workstream_c_landed() -> dict[str, Any]:
    """Best-effort automated check for "has Workstream C's find_confirmation
    fix + retrain actually landed", per this task's explicit instruction to
    check model timestamps against confirmation.py's git history before
    trusting a real run. A heuristic, not proof: it cannot detect a retrain
    that reused the OLD confirmation.py semantics, nor a commit that touched
    confirmation.py for an unrelated reason. Human judgement (reading the
    actual commit) is still the final check before treating a report as a
    real verdict.
    """
    reasons: list[str] = []
    landed = True

    dirty = _git("status", "--porcelain", "--", CONFIRMATION_PY_REL)
    if dirty:
        landed = False
        reasons.append(
            f"{CONFIRMATION_PY_REL} has uncommitted changes ({dirty!r}) -- "
            "Workstream C's fix is still in flight, not landed."
        )

    last_commit_iso = _git("log", "-1", "--format=%cI", "--", CONFIRMATION_PY_REL)
    if not last_commit_iso:
        landed = False
        reasons.append(f"could not read {CONFIRMATION_PY_REL}'s last commit timestamp")
    else:
        last_commit_dt = datetime.fromisoformat(last_commit_iso)
        for p in MODEL_PATHS:
            if not p.exists():
                landed = False
                reasons.append(f"{p} does not exist")
                continue
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=last_commit_dt.tzinfo)
            if mtime < last_commit_dt:
                landed = False
                reasons.append(
                    f"{p.name} (mtime {mtime.isoformat()}) predates "
                    f"{CONFIRMATION_PY_REL}'s last commit ({last_commit_dt.isoformat()}) "
                    "-- likely trained on pre-fix (mis-anchored) labels."
                )
    if landed:
        reasons.append(
            f"{CONFIRMATION_PY_REL} is committed and every tracked model "
            "artifact postdates its last commit."
        )
    return {"landed": landed, "reasons": reasons}


# ---------------------------------------------------------------------------
# Trial-history ledger (cumulative cross-session deflation)
# ---------------------------------------------------------------------------

def load_trial_history() -> int:
    """Sum of every prior session's ``n_trials`` -- passed as ``prior_trials``
    to :func:`firm.eval.overfitting.deflated_sharpe` (via
    ``ExperimentRunner.aggregate_walk_forward``) so DSR is deflated by the
    layer's full cross-session test history."""
    if not TRIAL_HISTORY_PATH.exists():
        return 0
    data = json.loads(TRIAL_HISTORY_PATH.read_text(encoding="utf-8"))
    return sum(int(e.get("n_trials", 0)) for e in data.get("entries", []))


def append_trial_history(entry: dict) -> None:
    data = (
        json.loads(TRIAL_HISTORY_PATH.read_text(encoding="utf-8"))
        if TRIAL_HISTORY_PATH.exists() else {"entries": []}
    )
    entries = data.setdefault("entries", [])
    entries.append(entry)
    data["cumulative_trials_through_last_entry"] = sum(
        int(e.get("n_trials", 0)) for e in entries
    )
    TRIAL_HISTORY_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    log.info(
        "appended trial-history entry (n_trials=%d); cumulative now %d",
        entry.get("n_trials", 0), data["cumulative_trials_through_last_entry"],
    )


def append_run_log(report: dict) -> None:
    data = (
        json.loads(RUN_LOG_PATH.read_text(encoding="utf-8"))
        if RUN_LOG_PATH.exists() else {"runs": []}
    )
    data.setdefault("runs", []).append({
        "run_started_at": report["run_started_at"],
        "run_finished_at": report["run_finished_at"],
        "dry_run": report["dry_run"],
        "workstream_c_landed": report["workstream_c_check"]["landed"],
        "bars_fingerprint": report["bars_fingerprint"],
        "overall_pass": report["overall_pass"],
    })
    RUN_LOG_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Portfolio-level (isolated single-strategy) walk-forward
# ---------------------------------------------------------------------------

def run_portfolio_walk_forward(*, n_splits: int, start_date: str | None = None) -> dict[str, Any]:
    """Reuses ``validate_pattern_cnn_walkforward``'s ``_build_config`` /
    ``ExperimentRunner.run_walk_forward`` / ``derive_recommendation`` /
    ``_fold_flag_selection_pattern`` verbatim, isolating ``pattern_recognition``
    alone (full capital/risk budget) via the same one-line ``_STRATEGIES``
    override the 2026-09-27 isolated evaluation used.

    ``start_date`` defaults to the pre-registered
    ``bars_module.WALK_FORWARD_START_DATE`` (2018-01-01) -- a real run must
    never override it. It exists as a parameter purely so ``--dry-run`` can
    pass a much shorter window and get a genuinely fast smoke test instead
    of silently still backtesting the full 8-year range with only the fold
    count reduced.
    """
    cnn_wf._STRATEGIES = ["pattern_recognition"]
    config = cnn_wf._build_config(
        start_date=start_date or bars_module.WALK_FORWARD_START_DATE,
        end_date=bars_module.WALK_FORWARD_END_DATE,
        settings_path=None,
    )
    registry = RunRegistry(base_dir="runs")
    runner = ExperimentRunner(registry=registry)
    runs = runner.run_walk_forward(
        config,
        n_splits=n_splits,
        train_pct=bars_module.WALK_FORWARD_TRAIN_PCT,
        seed=42,
        param_grid=bars_module.PARAM_GRID,
        selection_metric=bars_module.WALK_FORWARD_SELECTION_METRIC,
        embargo_days=bars_module.WALK_FORWARD_EMBARGO_DAYS,
    )
    failed = [r for r in runs if r.status != "completed"]
    if failed:
        raise RuntimeError(
            f"{len(failed)}/{len(runs)} walk-forward fold(s) failed: "
            f"{[r.run_id for r in failed]}"
        )

    prior_trials = load_trial_history()
    aggregate = runner.aggregate_walk_forward(
        runs, embargo_pct=0.0, prior_trials=prior_trials,
    )
    overfit = aggregate.get("overfitting") or {}

    fold_pattern = cnn_wf._fold_flag_selection_pattern(
        runs, bars_module.PARAM_GRID, flag_key="xgb_meta_confirmation_enabled",
    )
    recommendation = cnn_wf.derive_recommendation(
        overfit.get("verdict"), fold_pattern,
        flag_key="xgb_meta_confirmation_enabled",
        flag_label="meta-confidence-gated XGBoost",
    )
    return {
        "fold_ids": [r.run_id for r in runs],
        "n_folds": n_splits,
        "n_candidates": len(bars_module.PARAM_GRID),
        "overfitting": overfit,
        "fold_selection_pattern": fold_pattern,
        "recommendation": recommendation,
        "prior_trials_used": prior_trials,
        "this_run_trials": overfit.get("this_run_trials", n_splits * len(bars_module.PARAM_GRID)),
    }


# ---------------------------------------------------------------------------
# Trade-level executable-expectancy / placebo check
# ---------------------------------------------------------------------------

def _oos_test_windows() -> list[tuple[str, str]]:
    splits = ExperimentRunner._compute_walk_forward_splits(
        bars_module.WALK_FORWARD_START_DATE, bars_module.WALK_FORWARD_END_DATE,
        bars_module.WALK_FORWARD_N_SPLITS, bars_module.WALK_FORWARD_TRAIN_PCT,
        embargo_days=bars_module.WALK_FORWARD_EMBARGO_DAYS,
    )
    return [(test_start, test_end) for (_, _, test_start, test_end) in splits]


def _in_any_window(date: pd.Timestamp, windows: list[tuple[str, str]]) -> bool:
    return any(pd.Timestamp(s) <= date <= pd.Timestamp(e) for s, e in windows)


def _price_subset(
    events: list[dict], series_by_symbol: dict, mask_fn: Callable[[dict], bool],
    timeout_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(symbols, exec_net_r, placebo_net_r)`` for events passing
    ``mask_fn``, using Workstream B's own pricing functions unmodified."""
    sub = [ev for ev in events if mask_fn(ev)]
    syms = np.array([ev["symbol"] for ev in sub])
    exec_rows = [
        pit.price_executable(ev, series_by_symbol[ev["symbol"]], timeout_bars) for ev in sub
    ]
    placebo_rows = [
        pit.price_placebo(ev, series_by_symbol[ev["symbol"]], timeout_bars) for ev in sub
    ]
    exec_net = np.array([r["r_net"] if r is not None else np.nan for r in exec_rows])
    placebo_net = np.array([r["r_net"] if r is not None else np.nan for r in placebo_rows])
    return syms, exec_net, placebo_net


def run_trade_level_check(*, start: str, end: str) -> dict[str, Any]:
    """Executable expectancy vs. direction-flipped placebo, symbol-block
    bootstrap CI, restricted to the walk-forward's own OOS test-window union
    and the pre-registered meta-confidence threshold. Reuses
    ``measure_pattern_executable_expectancy`` (Workstream B) end to end.
    """
    symbols = pit.live_universe()
    panel = pit.load_panel(symbols, start, end)
    if panel.empty:
        return {"error": "no cached panel rows for this universe/date range"}

    events, series_by_symbol = pit.scan_events(
        panel,
        zigzag_pct=pit.SCAN_DEFAULTS["zigzag_pct"],
        min_score=pit.SCAN_DEFAULTS["min_score"],
        confirm_lookback_bars=pit.SCAN_DEFAULTS["confirm_lookback_bars"],
        stop_atr_floor=pit.SCAN_DEFAULTS["stop_atr_floor"],
        min_window_bars=60, step_bars=10, score_meta=True,
    )
    windows = _oos_test_windows()
    threshold = bars_module.TARGET_META_CONFIDENCE_THRESHOLD
    timeout_bars = bars_module.TRADE_LEVEL_TIMEOUT_BARS
    n_boot = bars_module.TRADE_LEVEL_N_BOOT
    seed = bars_module.TRADE_LEVEL_SEED

    def _target_mask(ev: dict) -> bool:
        return (
            ev["p_act"] is not None and ev["p_act"] >= threshold
            and _in_any_window(pd.Timestamp(ev["confirm_date"]), windows)
        )

    syms, exec_net, placebo_net = _price_subset(events, series_by_symbol, _target_mask, timeout_bars)
    exec_stats = pit.symbol_block_bootstrap(exec_net, syms, n_boot=n_boot, seed=seed)
    placebo_stats = pit.symbol_block_bootstrap(placebo_net, syms, n_boot=n_boot, seed=seed)

    diagnostics: dict[str, Any] = {}
    for thr in bars_module.DIAGNOSTIC_META_CONFIDENCE_THRESHOLDS:
        def _diag_mask(ev: dict, thr: float = thr) -> bool:
            return (
                ev["p_act"] is not None and ev["p_act"] >= thr
                and _in_any_window(pd.Timestamp(ev["confirm_date"]), windows)
            )
        d_syms, d_exec, d_placebo = _price_subset(events, series_by_symbol, _diag_mask, timeout_bars)
        diagnostics[f">={thr:.2f}"] = {
            "executable": pit.symbol_block_bootstrap(d_exec, d_syms, n_boot=n_boot, seed=seed),
            "placebo": pit.symbol_block_bootstrap(d_placebo, d_syms, n_boot=n_boot, seed=seed),
        }

    return {
        "target_threshold": threshold,
        "n_events_total": len(events),
        "n_events_target_subset": int(np.isfinite(exec_net).sum()),
        "executable": exec_stats,
        "placebo": placebo_stats,
        "diagnostics_other_thresholds": diagnostics,
        "caveat": (
            "The meta model was trained on the FULL panel including these "
            "dates -- this subset is in-sample w.r.t. the model itself (same "
            "caveat measure_pattern_executable_expectancy.py's own p_act-bucket "
            "section states) even though it is restricted to the walk-forward's "
            "OOS test-window union. Read as an upper bound, not a clean OOS "
            "estimate of what a per-fold-retrained model would do."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Fast smoke test of the harness mechanics only -- NOT a real "
        "verdict, and never written to docs/pattern_ml_trial_history.json.",
    )
    ap.add_argument(
        "--n-splits", type=int, default=None,
        help="Fold count override (dry-run only; a real run always uses "
        "pattern_ml_preregistered_bars.WALK_FORWARD_N_SPLITS).",
    )
    ap.add_argument(
        "--dry-run-start-date", default="2023-01-01",
        help="Shorter start date for a fast dry-run smoke test only.",
    )
    ap.add_argument(
        "--skip-c-check", action="store_true",
        help="Force past the Workstream-C landed-check without --dry-run "
        "(does NOT mark the report as non-dry -- see module docstring).",
    )
    ap.add_argument("--output", default="/tmp/pattern_ml_workstream_d_report.json")
    args = ap.parse_args(argv)

    c_status = check_workstream_c_landed()
    log.info("Workstream C landed-check: landed=%s reasons=%s", c_status["landed"], c_status["reasons"])
    if not args.dry_run and not c_status["landed"] and not args.skip_c_check:
        log.error(
            "Refusing to run the REAL pre-registered evaluation: Workstream "
            "C's find_confirmation fix + retrain does not look landed yet "
            "(%s). Use --dry-run for a smoke test, or wait for C.",
            "; ".join(c_status["reasons"]),
        )
        return 2

    n_splits = args.n_splits or bars_module.WALK_FORWARD_N_SPLITS
    if args.dry_run and args.n_splits is None:
        n_splits = min(2, bars_module.WALK_FORWARD_N_SPLITS)
    wf_start = args.dry_run_start_date if args.dry_run else bars_module.WALK_FORWARD_START_DATE

    run_started_at = datetime.now(UTC).isoformat()
    log.info(
        "Starting %s Workstream D evaluation: n_splits=%d start=%s bars_fingerprint=%s",
        "DRY-RUN" if args.dry_run else "REAL", n_splits, wf_start, bars_module.bars_fingerprint(),
    )

    portfolio = run_portfolio_walk_forward(
        n_splits=n_splits, start_date=wf_start if args.dry_run else None,
    )
    trade_level = run_trade_level_check(
        start=wf_start, end=bars_module.WALK_FORWARD_END_DATE,
    )

    exec_stats = trade_level.get("executable", {}) or {}
    placebo_stats = trade_level.get("placebo", {}) or {}
    exec_mean = exec_stats.get("mean")
    exec_ci_low = exec_stats.get("ci_low")
    placebo_mean = placebo_stats.get("mean")

    def _finite(x) -> bool:
        return x is not None and isinstance(x, (int, float)) and x == x

    measurements = {
        "executable_expectancy_r_positive": exec_mean if _finite(exec_mean) else None,
        "expectancy_ci_excludes_zero": (_finite(exec_ci_low) and exec_ci_low > 0) if _finite(exec_ci_low) else None,
        "expectancy_beats_placebo": (
            (exec_mean > placebo_mean) if (_finite(exec_mean) and _finite(placebo_mean)) else None
        ),
        "deflated_sharpe_min": portfolio["overfitting"].get("deflated_sharpe"),
        "pbo_max": portfolio["overfitting"].get("pbo"),
        "pbo_min_candidates": portfolio["n_candidates"],
        "fold_consistency_min": portfolio["recommendation"].get("true_fraction"),
    }
    verdict = bars_module.evaluate_all(measurements)

    report = {
        "run_started_at": run_started_at,
        "run_finished_at": datetime.now(UTC).isoformat(),
        "dry_run": bool(args.dry_run),
        "workstream_c_check": c_status,
        "bars_fingerprint": verdict["bars_fingerprint"],
        "preregistered_at": verdict["preregistered_at"],
        "measurements": measurements,
        "per_bar": verdict["per_bar"],
        "overall_pass": verdict["overall_pass"],
        "portfolio_walk_forward": portfolio,
        "trade_level_check": trade_level,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))

    log.info(
        "Overall verdict: %s (dry_run=%s) -- see %s",
        "PASS" if verdict["overall_pass"] else "FAIL", args.dry_run, args.output,
    )
    for name, b in verdict["per_bar"].items():
        log.info("  bar %-32s value=%s passed=%s", name, b["value"], b["passed"])

    if not args.dry_run:
        append_trial_history({
            "date": datetime.now(UTC).date().isoformat(),
            "source": "scripts/validate_pattern_ml_workstream_d.py (Workstream D pre-registered evaluation)",
            "n_folds": portfolio["n_folds"],
            "n_candidates": portfolio["n_candidates"],
            "candidates": [json.dumps(c, sort_keys=True) for c in bars_module.PARAM_GRID],
            "n_trials": portfolio["this_run_trials"],
            "reported_pbo": portfolio["overfitting"].get("pbo"),
            "reported_deflated_sharpe": portfolio["overfitting"].get("deflated_sharpe"),
            "verdict": "pass" if verdict["overall_pass"] else "fail",
        })
    append_run_log(report)

    return 0 if verdict["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
