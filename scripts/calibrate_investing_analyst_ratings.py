#!/usr/bin/env python
"""A/B the investing_analyst_ratings strategy: baseline roster vs +1 strategy.

Same 25-symbol universe, ``optimal`` combination, and diagnostic windows as
``calibrate_regime_ensemble.py``/``calibrate_strategy_regime_weights.py`` —
this is the Phase 2a evidence gate for
``firm.strategies.investing_analyst_ratings`` (docs/investing_pro_integration.md):
prove the strategy has a real edge on FMP's historical grades-consensus data
before ever wiring Investing.com Pro's live "Analyst Ratings" feed into the
same strategy and enabling it in config/live.yaml.

Usage:
    python scripts/calibrate_investing_analyst_ratings.py
    python scripts/calibrate_investing_analyst_ratings.py --windows run_18mo_2025_2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.backtest.run import execute_backtest  # noqa: E402
from firm.config import get_settings  # noqa: E402

log = logging.getLogger(__name__)

_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "NFLX", "ADBE", "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH",
    "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM",
]

_BASELINE_STRATEGIES = [
    "momentum", "trend", "mean_reversion", "stat_arb", "multi_factor",
    "sentiment", "event_driven", "volatility_breakout", "seasonality",
    "regime_hmm",
]

WINDOWS = [
    {"name": "run_18mo_2025_2026", "start_date": "2025-01-01", "end_date": "2026-06-30"},
    {"name": "wf_fold0_2020_2021", "start_date": "2020-12-01", "end_date": "2021-04-23"},
    {"name": "wf_fold1", "start_date": "2021-04-24", "end_date": "2021-09-15"},
]


def _base_config(window: dict, strategies: list[str]) -> dict:
    settings = get_settings()
    risk = settings.risk.model_dump()
    return {
        "data_source": "cache",
        "start_date": window["start_date"],
        "end_date": window["end_date"],
        "initial_capital": 1_000_000.0,
        "commission_pct": settings.backtest.commission_pct,
        "slippage_pct": settings.backtest.slippage_pct,
        "spread_pct": settings.backtest.spread_pct,
        "market_impact_coefficient": settings.backtest.market_impact_coefficient,
        "rebalance_frequency": "weekly",
        "universe_symbols": list(_UNIVERSE),
        "strategies": list(strategies),
        "strategy_params": dict(settings.strategy_params or {}),
        "seed": 42,
        **risk,
        "signal_combination": {"method": "optimal"},
    }


def _run_one(window: dict, *, with_ratings: bool) -> dict:
    strategies = _BASELINE_STRATEGIES + (["investing_analyst_ratings"] if with_ratings else [])
    config = _base_config(window, strategies)

    t0 = time.time()
    report = execute_backtest(config)
    elapsed = time.time() - t0
    d = report.to_dict()
    portfolio = d.get("portfolio", {})
    strategies_d = d.get("strategies", {})
    ratings = strategies_d.get("investing_analyst_ratings", {})
    overfitting = d.get("overfitting", {})
    return {
        "window": window["name"],
        "arm": "with_ratings" if with_ratings else "baseline",
        "elapsed_seconds": round(elapsed, 1),
        "portfolio_sharpe": portfolio.get("sharpe_ratio"),
        "portfolio_total_return": portfolio.get("total_return"),
        "portfolio_max_drawdown": portfolio.get("max_drawdown"),
        "investing_analyst_ratings_sharpe": ratings.get("sharpe_ratio"),
        "deflated_sharpe": overfitting.get("deflated_sharpe_ratio"),
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows", nargs="*", default=None,
        help="Subset of window names to run (default: all 3)",
    )
    parser.add_argument("--output", default="/tmp/investing_analyst_ratings_calibration.json")
    args = parser.parse_args()

    windows = WINDOWS
    if args.windows:
        windows = [w for w in WINDOWS if w["name"] in args.windows]

    results: list[dict] = []
    for window in windows:
        for with_ratings in (False, True):
            label = f"{window['name']} / {'with_ratings' if with_ratings else 'baseline'}"
            print(f"Running {label} ...", file=sys.stderr)
            try:
                results.append(_run_one(window, with_ratings=with_ratings))
            except Exception as exc:
                log.exception("run failed: %s", label)
                results.append({
                    "window": window["name"],
                    "arm": "with_ratings" if with_ratings else "baseline",
                    "error": str(exc),
                })

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2, default=str))

    print("\n=== investing_analyst_ratings: baseline vs +1 strategy ===\n")
    print(f"{'window':22s} {'arm':13s} {'sharpe':>8s} {'ret':>8s} {'max_dd':>8s} {'ratings_sharpe':>15s}")
    print("-" * 82)
    for r in results:
        if "error" in r:
            print(f"{r['window']:22s} {r['arm']:13s} ERROR: {r['error']}")
            continue
        print(
            f"{r['window']:22s} {r['arm']:13s} "
            f"{r['portfolio_sharpe']:8.3f} {r['portfolio_total_return']:8.3f} "
            f"{r['portfolio_max_drawdown']:8.3f} "
            f"{(r['investing_analyst_ratings_sharpe'] or 0.0):15.3f}"
        )
    print(f"\nFull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
