#!/usr/bin/env python
"""Genuine multi-fold walk-forward + PBO/DSR audit of pattern_recognition's
``cnn_scoring_enabled`` toggle -- the live CNN/GAF chart-pattern quality
scorer currently flipped ``true`` in ``config/live.yaml``.

Why this script exists: that live-config flip is justified today by a
*single* anchored walk-forward window (2010-06-01 train -> ... -> 2023-12-31
test; see the comment above ``strategy_params.pattern_recognition`` in
``config/live.yaml``), explicitly flagged in that same comment as "not this
codebase's usual 3-window robustness check ... revisit if live results
diverge". This script repeats that same CNN-on vs CNN-off comparison but as
a genuine walk-forward across >=5 folds spanning multiple market regimes
(2018-2019 low-vol, 2020 crash/recovery, 2022 rate-driven bear, 2023+
recovery/bull), so the result carries real statistical backing (PBO/DSR)
instead of one anchored split.

Mirrors ``scripts/run_walk_forward_pbo_audit.py``'s ``_build_config``/
argparse/logging structure closely (same ``ExperimentRunner.run_walk_forward``
+ ``ExperimentRunner.aggregate_walk_forward`` mechanic) with two differences:
the strategy roster is the *full* live 11-strategy roster (not the smaller
subset that script uses for combination/allocation experiments -- pattern_
recognition's CNN toggle only matters when it actually runs inside the real
live pipeline, competing for capital against every other live strategy) and
the ``param_grid`` is this specific 2-candidate CNN on/off decision instead
of a signal-combination/allocation sweep.

Usage:
    python scripts/validate_pattern_cnn_walkforward.py
    python scripts/validate_pattern_cnn_walkforward.py --n-splits 5 --output /tmp/pattern_cnn_walkforward_audit.json
    python scripts/validate_pattern_cnn_walkforward.py --param-grid-json grid.json

Output: full JSON results written to ``--output`` (default
``/tmp/pattern_cnn_walkforward_audit.json``), plus a plain-English
KEEP/HOLD/ROLLBACK recommendation (see ``derive_recommendation`` below)
printed and logged.
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

from firm.config import get_settings  # noqa: E402
from firm.experiments.registry import RunRegistry  # noqa: E402
from firm.experiments.runner import ExperimentRunner  # noqa: E402
from firm.strategies.pattern_recognition import (  # noqa: E402
    PatternRecognitionStrategy,
)

log = logging.getLogger(__name__)

# Same 25-symbol universe as scripts/run_walk_forward_pbo_audit.py, which is
# itself identical (verified 2026-09-25 against config/live.yaml's
# `universe.symbols`) to the live universe -- reused verbatim rather than a
# fresh choice, since pattern_recognition scans whatever universe it's given
# and there is no reason for its CNN-toggle decision to be validated against
# a different symbol set than the live book it would actually run in.
_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "NFLX", "ADBE", "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH",
    "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM",
]

# Full live 11-strategy roster (config/live.yaml `strategies.enabled`,
# verified 2026-09-25 -- ml_prediction/gann are registered but permanently
# disabled, not part of this list). pattern_recognition must run alongside
# every other live strategy here, not in isolation: its CNN toggle changes
# how much conviction it contributes to the shared cross-sectional z-score
# and downstream capital allocation, which only means something in the
# context of the full roster it actually competes against live.
_STRATEGIES = [
    "momentum", "trend", "mean_reversion", "stat_arb", "multi_factor",
    "sentiment", "event_driven", "volatility_breakout", "seasonality",
    "regime_hmm", "pattern_recognition",
]

# The genuine, currently-relevant decision: is config/live.yaml's
# `strategy_params.pattern_recognition.cnn_scoring_enabled: true` justified
# by real walk-forward evidence? Candidate 0 = current rule-based-only
# fallback (the strategy's own default_params default); candidate 1 =
# today's live setting.
#
# NOTE for future maintainers: once a separate workstream lands an
# XGBoost-ensemble mode for pattern_recognition (a new strategy_params key,
# name TBD), add a 3rd candidate here exercising it, e.g.
#   {"strategy_params": {"pattern_recognition": {"cnn_scoring_enabled": False,
#                                                  "<ensemble_key>": True}}}
# As of 2026-09-25, PatternRecognitionStrategy.default_params (see
# src/firm/strategies/pattern_recognition.py) has exactly one relevant key
# (`cnn_scoring_enabled`) -- no ensemble key exists yet, so the grid below
# stays at 2 candidates. The check below fails loudly (rather than silently
# running a stale 2-candidate grid forever) once new keys do show up, so a
# future run of this script surfaces the need to update DEFAULT_PARAM_GRID
# instead of a human having to remember to.
_KNOWN_PATTERN_RECOGNITION_PARAMS = frozenset(
    PatternRecognitionStrategy.default_params.keys()
)
_EXPECTED_PATTERN_RECOGNITION_PARAMS = frozenset({
    "lookback_days", "zigzag_pct", "min_score", "min_risk_reward",
    "confirm_lookback_bars", "stop_atr_floor", "horizon", "enabled_patterns",
    "cnn_scoring_enabled",
})
_NEW_PATTERN_RECOGNITION_PARAMS = (
    _KNOWN_PATTERN_RECOGNITION_PARAMS - _EXPECTED_PATTERN_RECOGNITION_PARAMS
)
if _NEW_PATTERN_RECOGNITION_PARAMS:
    log.warning(
        "pattern_recognition.default_params grew new key(s) %s since this "
        "script's DEFAULT_PARAM_GRID was written -- if one of these is the "
        "XGBoost-ensemble enable flag, add a 3rd param_grid candidate "
        "exercising it (see the comment above DEFAULT_PARAM_GRID) before "
        "trusting this audit as the final word on pattern_recognition's "
        "current parameter set",
        sorted(_NEW_PATTERN_RECOGNITION_PARAMS),
    )

DEFAULT_PARAM_GRID: list[dict] = [
    {"strategy_params": {"pattern_recognition": {"cnn_scoring_enabled": False}}},
    {"strategy_params": {"pattern_recognition": {"cnn_scoring_enabled": True}}},
]


def _build_config(
    *,
    start_date: str,
    end_date: str,
    settings_path: str | None,
) -> dict:
    """Mirrors run_walk_forward_pbo_audit.py's _build_config exactly, but
    with the full 11-strategy live roster (see _STRATEGIES above) instead of
    that script's smaller combination/allocation-experiment subset.
    """
    settings = get_settings(settings_path)
    bt = settings.backtest.model_dump()
    return {
        "name": "pattern_cnn_walkforward_audit",
        "backtest": {
            **bt,
            "start_date": start_date,
            "end_date": end_date,
        },
        "strategies": {"enabled": list(_STRATEGIES)},
        "strategy_params": dict(settings.strategy_params or {}),
        "allocation_method": settings.allocation_method,
        "kelly_fraction": settings.kelly_fraction,
        "signal_combination": settings.signal_combination,
        # Backtest parity with live for TraderAgent's conviction-EMA
        # smoothing -- see firm.config.Settings' field docstring.
        "conviction_smoothing_enabled": settings.conviction_smoothing_enabled,
        "conviction_smoothing_halflife_days": settings.conviction_smoothing_halflife_days,
        # Backtest parity with live for analyst cross-sectional
        # normalization -- see the field docstring in firm.config.Settings.
        "zscore_demean": settings.zscore_demean,
        "rebalance_band_pct": settings.backtest.rebalance_band_pct,
        "rebalance_fraction": settings.backtest.rebalance_fraction,
        "strategy_circuit_breaker": settings.strategy_circuit_breaker,
        "strategy_regime_weights": settings.strategy_regime_weights,
        "data_source": "cache",
        "universe_symbols": list(_UNIVERSE),
        "risk": settings.risk.model_dump(),
        "seed": 42,
    }


def _fold_cnn_selection_pattern(
    runs: list, param_grid: list[dict]
) -> list[dict]:
    """Read each fold's ``walk_forward_selection.json`` (written by
    ``ExperimentRunner.run_walk_forward`` for every fold with a genuine
    multi-candidate grid) and report, per fold, which candidate won on the
    train window and whether that candidate has ``cnn_scoring_enabled: True``.

    Degrades gracefully (an entry with ``selected_cnn_enabled: None``) for a
    fold that has no selection file (e.g. every candidate failed on train
    and _select_candidate_on_train defaulted to candidate 0 without writing
    one) so the pattern below is always well-formed for downstream reporting.
    """
    pattern: list[dict] = []
    for i, run in enumerate(runs):
        sel_path = Path(run.artifacts_dir) / "walk_forward_selection.json"
        if not sel_path.exists():
            log.warning(
                "fold %d (%s): no walk_forward_selection.json -- cannot "
                "attribute a winning candidate for this fold",
                i, run.run_id,
            )
            pattern.append({
                "fold": i, "run_id": run.run_id,
                "selected_index": None, "selected_cnn_enabled": None,
            })
            continue
        selection = json.loads(sel_path.read_text(encoding="utf-8"))
        idx = selection.get("selected_index")
        cnn_enabled = None
        if idx is not None and 0 <= idx < len(param_grid):
            cnn_enabled = bool(
                param_grid[idx]
                .get("strategy_params", {})
                .get("pattern_recognition", {})
                .get("cnn_scoring_enabled", False)
            )
        pattern.append({
            "fold": i, "run_id": run.run_id,
            "selected_index": idx, "selected_cnn_enabled": cnn_enabled,
        })
    return pattern


# Fraction of folds that must select a cnn_scoring_enabled=True candidate
# for the pattern to count as "consistently favoring True" per the KEEP
# branch below. Deliberately stricter than a bare >50% majority ("mixed" is
# still mixed even if True edges out False 3-2) -- see derive_recommendation.
CONSISTENT_MAJORITY_THRESHOLD = 0.75


def derive_recommendation(
    verdict: str | None,
    fold_selection_pattern: list[dict],
    consistent_majority_threshold: float = CONSISTENT_MAJORITY_THRESHOLD,
) -> dict:
    """Deterministic KEEP / HOLD / ROLLBACK decision rule.

    Pure function over already-computed inputs (no I/O) so it's directly
    unit-testable and auditable independent of the walk-forward run itself.

    Rule (explicit, matches the task's decision policy):
      1. ``verdict != "pass"`` (fail, or no verdict could be computed at all,
         e.g. too few usable folds) -> ROLLBACK. An audit that doesn't clear
         the overfitting bar is not evidence strong enough to keep a change
         live; the conservative default is the strategy's own pre-CNN
         baseline (``cnn_scoring_enabled: False``).
      2. ``verdict == "pass"`` AND at least ``consistent_majority_threshold``
         (default 75%) of folds with a usable selection independently picked
         the CNN-on candidate on their train window -> KEEP
         ``cnn_scoring_enabled: true`` as today's live config already has it.
      3. Everything else (verdict passes but the per-fold pattern is mixed,
         or ties, or there are no usable folds to attribute a pattern from)
         -> HOLD at the current live setting pending more data -- passing the
         aggregate overfitting check without a consistent per-fold
         preference is genuinely ambiguous, not grounds for either action.
    """
    usable = [
        f for f in fold_selection_pattern if f["selected_cnn_enabled"] is not None
    ]
    n_true = sum(1 for f in usable if f["selected_cnn_enabled"] is True)
    true_fraction = (n_true / len(usable)) if usable else None

    if verdict != "pass":
        return {
            "recommendation": "ROLLBACK",
            "target_cnn_scoring_enabled": False,
            "reason": (
                f"verdict={verdict!r} (not 'pass') -> ROLLBACK to "
                "cnn_scoring_enabled=false. An overfitting audit that does "
                "not clear PBO<0.5/DSR>0.95 is not strong enough evidence "
                "to justify keeping the CNN scorer live; reverting to the "
                "strategy's pre-CNN rule-based-only baseline is the "
                "conservative default."
            ),
            "true_fraction": true_fraction,
            "n_usable_folds": len(usable),
        }

    if true_fraction is not None and true_fraction >= consistent_majority_threshold:
        return {
            "recommendation": "KEEP",
            "target_cnn_scoring_enabled": True,
            "reason": (
                f"verdict='pass' AND {true_fraction * 100:.0f}% of "
                f"{len(usable)} usable fold(s) selected cnn_scoring_enabled="
                f"True on their train window (>= "
                f"{consistent_majority_threshold * 100:.0f}% consistency "
                "threshold) -> KEEP cnn_scoring_enabled=true as config/"
                "live.yaml already has it; today's setting now has genuine "
                "walk-forward statistical backing instead of the prior "
                "single-window validation."
            ),
            "true_fraction": true_fraction,
            "n_usable_folds": len(usable),
        }

    return {
        "recommendation": "HOLD",
        "target_cnn_scoring_enabled": None,
        "reason": (
            f"verdict='pass' but the per-fold selection pattern is mixed/"
            f"ambiguous ({true_fraction * 100:.0f}% of {len(usable)} usable "
            "fold(s) favored True" if true_fraction is not None
            else "verdict='pass' but no fold had a usable selection to "
            "attribute a winning candidate from"
        ) + " -- HOLD at the current live setting (cnn_scoring_enabled: "
            "true) pending more live data rather than acting on an "
            "ambiguous backtest read.",
        "true_fraction": true_fraction,
        "n_usable_folds": len(usable),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=None, help="Settings YAML path")
    parser.add_argument(
        "--start-date", default="2018-01-01",
        help="See the timing note in this script's module docstring / final "
        "audit report for why this range (and --n-splits) were chosen.",
    )
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--train-pct", type=float, default=0.7)
    parser.add_argument("--selection-metric", default="sharpe_ratio")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--param-grid-json",
        default=None,
        help="JSON file with a list of config override dicts (default: the "
        "2-candidate CNN on/off DEFAULT_PARAM_GRID above)",
    )
    parser.add_argument(
        "--output", default="/tmp/pattern_cnn_walkforward_audit.json",
    )
    parser.add_argument(
        "--embargo-days", type=int, default=1,
        help="Calendar-day gap between each fold's train and test windows "
        "(default 1). See ExperimentRunner._compute_walk_forward_splits.",
    )
    parser.add_argument(
        "--pbo-embargo-pct", type=float, default=0.0,
        help="Fraction of each CSCV block purged from the edges of "
        "out-of-sample blocks adjacent to in-sample blocks when computing "
        "PBO (default 0.0 = original, unpurged CSCV split). See "
        "firm.eval.overfitting.cscv_pbo.",
    )
    parser.add_argument(
        "--consistent-majority-threshold", type=float,
        default=CONSISTENT_MAJORITY_THRESHOLD,
        help="Fraction of folds that must select the CNN-on candidate for "
        "derive_recommendation to call the pattern consistent (default "
        f"{CONSISTENT_MAJORITY_THRESHOLD}).",
    )
    args = parser.parse_args()

    param_grid = DEFAULT_PARAM_GRID
    if args.param_grid_json:
        param_grid = json.loads(Path(args.param_grid_json).read_text(encoding="utf-8"))
    if len(param_grid) < 2:
        log.error("param_grid must have >= 2 candidates for genuine PBO")
        return 1

    config = _build_config(
        start_date=args.start_date,
        end_date=args.end_date,
        settings_path=args.settings,
    )
    registry = RunRegistry(base_dir=args.runs_dir)
    runner = ExperimentRunner(registry=registry)

    log.info(
        "pattern_recognition CNN walk-forward: %s -> %s, %d folds, %d "
        "candidate(s), strategies=%s",
        args.start_date, args.end_date, args.n_splits, len(param_grid),
        _STRATEGIES,
    )
    runs = runner.run_walk_forward(
        config,
        n_splits=args.n_splits,
        train_pct=args.train_pct,
        seed=42,
        param_grid=param_grid,
        selection_metric=args.selection_metric,
        embargo_days=args.embargo_days,
    )
    failed = [r for r in runs if r.status != "completed"]
    if failed:
        log.error("%d fold(s) failed", len(failed))
        for r in failed:
            log.error("  %s: %s", r.run_id, r.notes)
        return 1

    aggregate = runner.aggregate_walk_forward(runs, embargo_pct=args.pbo_embargo_pct)
    overfit = aggregate.get("overfitting") or {}
    fold_selection_pattern = _fold_cnn_selection_pattern(runs, param_grid)
    recommendation = derive_recommendation(
        overfit.get("verdict"),
        fold_selection_pattern,
        consistent_majority_threshold=args.consistent_majority_threshold,
    )

    result = {
        "fold_ids": [r.run_id for r in runs],
        "param_grid": param_grid,
        "n_splits": args.n_splits,
        "train_pct": args.train_pct,
        "selection_metric": args.selection_metric,
        "embargo_days": args.embargo_days,
        "pbo_embargo_pct": args.pbo_embargo_pct,
        "date_range": {"start": args.start_date, "end": args.end_date},
        "strategies": list(_STRATEGIES),
        "universe": list(_UNIVERSE),
        "fold_selection_pattern": fold_selection_pattern,
        "recommendation": recommendation,
        **aggregate,
    }

    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, indent=2, default=str))
    if overfit:
        log.info(
            "PBO=%s DSR=%s PSR=%s verdict=%s (pbo_n_folds=%s)",
            overfit.get("pbo"),
            overfit.get("deflated_sharpe"),
            overfit.get("probabilistic_sharpe"),
            overfit.get("verdict"),
            overfit.get("pbo_n_folds"),
        )
    else:
        log.warning("No overfitting block — check walk_forward_selection.json per fold")

    print(
        f"\nRECOMMENDATION: {recommendation['recommendation']} "
        f"(target cnn_scoring_enabled={recommendation['target_cnn_scoring_enabled']})"
        f"\n  {recommendation['reason']}"
    )
    log.info(
        "recommendation=%s target_cnn_scoring_enabled=%s true_fraction=%s "
        "n_usable_folds=%s",
        recommendation["recommendation"],
        recommendation["target_cnn_scoring_enabled"],
        recommendation["true_fraction"],
        recommendation["n_usable_folds"],
    )
    log.info("Full results: %s", out)

    if not overfit:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
