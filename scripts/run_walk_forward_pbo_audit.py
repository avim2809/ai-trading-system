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
    )
    failed = [r for r in runs if r.status != "completed"]
    if failed:
        log.error("%d fold(s) failed", len(failed))
        for r in failed:
            log.error("  %s: %s", r.run_id, r.notes)
        return 1

    aggregate = runner.aggregate_walk_forward(runs)
    overfit = aggregate.get("overfitting") or {}
    result = {
        "fold_ids": [r.run_id for r in runs],
        "param_grid": param_grid,
        "n_splits": args.n_splits,
        "train_pct": args.train_pct,
        "selection_metric": args.selection_metric,
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
