#!/usr/bin/env python
"""Run a genuine walk-forward with ``param_grid`` and print PBO / DSR stats.

Each fold backtests every grid candidate on the train window, selects the
best in-sample ``selection_metric``, then runs the winner on the test window.
``walk_forward_selection.json`` per fold feeds :func:`walk_forward_overfitting`
so PBO reflects real competing trials (not sequential OOS folds of one config).

Default grid varies ``signal_combination`` and ``allocation_method`` — the
same knobs the firm actually tunes between backtest and live.

Usage:
    python scripts/run_walk_forward_pbo_audit.py
    python scripts/run_walk_forward_pbo_audit.py --n-splits 3 --output /tmp/pbo_audit.json
    python scripts/run_walk_forward_pbo_audit.py --param-grid-json grid.json
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

log = logging.getLogger(__name__)

_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "NFLX", "ADBE", "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH",
    "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM",
]

_STRATEGIES = [
    "momentum", "trend", "mean_reversion", "stat_arb", "multi_factor",
    "sentiment", "event_driven", "volatility_breakout", "seasonality",
    "regime_hmm",
]

DEFAULT_PARAM_GRID: list[dict] = [
    {
        "signal_combination": {"method": "confidence"},
        "allocation_method": "conviction_weighted",
    },
    {
        "signal_combination": {"method": "optimal"},
        "allocation_method": "conviction_weighted",
    },
    {
        "signal_combination": {"method": "optimal"},
        "allocation_method": "equal_weight",
    },
    {
        # PART 2 redesign candidate: joint mean-variance-with-costs QP
        # (firm.portfolio.optimizer) replacing L1-normalize-to-full-
        # investment sizing + RiskAgent's sequential clip passes. See PART 2
        # of the remediation plan — this is the mandatory walk-forward+PBO
        # gate before any live promotion.
        "signal_combination": {"method": "optimal"},
        "allocation_method": "joint_optimizer",
        # The optimizer's native transaction-cost term already produces an
        # endogenous no-trade region and partial rebalancing (see PART 2's
        # design doc) -- leaving the #58 bolt-ons (no-trade band, turnover-
        # aware sizing fraction, conviction-EMA smoothing) on top would
        # double-damp the same effect and bias this candidate's turnover/
        # Sharpe away from what the native cost model alone actually does.
        # Both are surfaced as flat top-level keys specifically so a
        # param_grid candidate can override them (see _build_config below).
        "rebalance_band_pct": 0.0,
        "rebalance_fraction": 1.0,
        "conviction_smoothing_enabled": False,
    },
    {
        # PART 3 Phase 1 candidate: drop stat_arb's 2 ETF pairs (SPY/QQQ,
        # SPY/IWM) from its 6 predefined pairs, keeping the 4 single-name
        # pairs. An ETF pair's 60-day cointegration test can't distinguish
        # a temporary mean-reverting dislocation from a structural regime
        # break (e.g. 2022's rate-driven tech underperformance vs. the
        # broader market) -- it would mechanically hold "long QQQ / short
        # SPY" through the whole break. See PART 3 of the remediation plan.
        # No allowlist changes needed: strategy_params already flows
        # through ExperimentRunner._flatten_config and is deep-merged (not
        # shallow-clobbered) by _merge_override.
        "signal_combination": {"method": "optimal"},
        "allocation_method": "conviction_weighted",
        "strategy_params": {
            "stat_arb": {
                "predefined_pairs": [
                    ["AAPL", "MSFT"],
                    ["JPM", "BAC"],
                    ["XOM", "CVX"],
                    ["GOOG", "META"],
                ],
            },
        },
    },
    {
        # PART 3 Phase 3 candidate: reroute `seasonality` from a per-symbol
        # strategy (identical score across every symbol, a market-wide
        # timing signal wrongly shaped for the stock-picker pipeline -- see
        # PART 3 of the remediation plan) to a RiskAgent multiplicative
        # gross-exposure overlay (_seasonality_exposure_overlay), the same
        # pattern already used for the HMM regime signal. Drop it from the
        # strategy roster here (not a shared default -- see _STRATEGIES
        # above, left unchanged pending this gate) and enable the overlay
        # via the flat top-level `seasonality_overlay` key (explicitly
        # allowlisted in ExperimentRunner._flatten_config) rather than
        # nesting it under "risk", which _merge_override's shallow top-level
        # merge would otherwise clobber wholesale.
        "signal_combination": {"method": "optimal"},
        "allocation_method": "conviction_weighted",
        "strategies": {"enabled": [s for s in _STRATEGIES if s != "seasonality"]},
        "seasonality_overlay": {"enabled": True, "scale": 0.15},
    },
    {
        # PART 3 Phase 4 candidate: macro/rate exposure overlay
        # (_macro_exposure_overlay) -- none of the 10 live strategies
        # capture interest-rate direction, the primary driver of the 2022
        # bear market that's been the worst OOS fold in every audit this
        # session regardless of combination mechanism. Full strategy roster
        # unchanged (this overlay is orthogonal to the signal set, not a
        # replacement for anything). See PART 3 of the remediation plan.
        "signal_combination": {"method": "optimal"},
        "allocation_method": "conviction_weighted",
        "macro_overlay": {"enabled": True},
    },
]


def _build_config(
    *,
    start_date: str,
    end_date: str,
    settings_path: str | None,
) -> dict:
    settings = get_settings(settings_path)
    bt = settings.backtest.model_dump()
    return {
        "name": "walk_forward_pbo_audit",
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
        # smoothing — see firm.config.Settings' field docstring.
        "conviction_smoothing_enabled": settings.conviction_smoothing_enabled,
        "conviction_smoothing_halflife_days": settings.conviction_smoothing_halflife_days,
        # Backtest parity with live for analyst cross-sectional
        # normalization — see the field docstring in firm.config.Settings.
        "zscore_demean": settings.zscore_demean,
        # Also surfaced as explicit top-level keys (in addition to living
        # inside the "backtest" sub-dict via bt above) so a param_grid
        # candidate can override them: ExperimentRunner._merge_override is a
        # shallow top-level merge, so an override nested under "backtest"
        # would replace the *entire* backtest sub-dict (clobbering start/end
        # dates, commission_pct, etc.) instead of just these two fields.
        "rebalance_band_pct": settings.backtest.rebalance_band_pct,
        "rebalance_fraction": settings.backtest.rebalance_fraction,
        "strategy_circuit_breaker": settings.strategy_circuit_breaker,
        "strategy_regime_weights": settings.strategy_regime_weights,
        "data_source": "cache",
        "universe_symbols": list(_UNIVERSE),
        "risk": settings.risk.model_dump(),
        "seed": 42,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=None, help="Settings YAML path")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--train-pct", type=float, default=0.7)
    parser.add_argument("--selection-metric", default="sharpe_ratio")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--param-grid-json",
        default=None,
        help="JSON file with a list of config override dicts",
    )
    parser.add_argument("--output", default="/tmp/walk_forward_pbo_audit.json")
    parser.add_argument(
        "--embargo-days", type=int, default=1,
        help="Calendar-day gap between each fold's train and test windows "
        "(default 1, matching prior behaviour). See "
        "ExperimentRunner._compute_walk_forward_splits.",
    )
    parser.add_argument(
        "--pbo-embargo-pct", type=float, default=0.0,
        help="Fraction of each CSCV block purged from the edges of "
        "out-of-sample blocks adjacent to in-sample blocks when computing "
        "PBO (default 0.0 = original, unpurged CSCV split). See "
        "firm.eval.overfitting.cscv_pbo.",
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
        "Walk-forward: %s → %s, %d folds, %d candidates",
        args.start_date, args.end_date, args.n_splits, len(param_grid),
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
    result = {
        "fold_ids": [r.run_id for r in runs],
        "param_grid": param_grid,
        "n_splits": args.n_splits,
        "train_pct": args.train_pct,
        "selection_metric": args.selection_metric,
        "embargo_days": args.embargo_days,
        "pbo_embargo_pct": args.pbo_embargo_pct,
        "date_range": {"start": args.start_date, "end": args.end_date},
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
        return 1

    log.info("Full results: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
