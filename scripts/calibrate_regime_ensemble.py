#!/usr/bin/env python
"""A/B the ensemble regime detector vs the single-model detector.

Compares portfolio Sharpe with ``strategy_regime_weights.enabled=True`` and
either the standalone Gaussian HMM or the ensemble-HMM
(``firm.regime.ensemble.EnsembleRegimeModel``) driving the market-regime
signal it depends on — same 10-strategy roster, ``optimal`` combination, and
diagnostic windows as ``calibrate_strategy_regime_weights.py``.

Usage:
    python scripts/calibrate_regime_ensemble.py
    python scripts/calibrate_regime_ensemble.py --windows run_18mo_2025_2026
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

_STRATEGIES = [
    "momentum", "trend", "mean_reversion", "stat_arb", "multi_factor",
    "sentiment", "event_driven", "volatility_breakout", "seasonality",
    "regime_hmm",
]

WINDOWS = [
    {"name": "run_18mo_2025_2026", "start_date": "2025-01-01", "end_date": "2026-06-30"},
    {"name": "wf_fold0_2020_2021", "start_date": "2020-12-01", "end_date": "2021-04-23"},
    {"name": "wf_fold1", "start_date": "2021-04-24", "end_date": "2021-09-15"},
]


def _base_config(window: dict) -> dict:
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
        "strategies": list(_STRATEGIES),
        "strategy_params": dict(settings.strategy_params or {}),
        "seed": 42,
        **risk,
        "signal_combination": {"method": "optimal"},
    }


def _run_one(window: dict, *, ensemble: bool) -> dict:
    config = _base_config(window)
    weights_cfg = dict(get_settings().strategy_regime_weights or {})
    config["strategy_regime_weights"] = {
        **weights_cfg, "enabled": True, "ensemble": ensemble,
    }

    t0 = time.time()
    report = execute_backtest(config)
    elapsed = time.time() - t0
    d = report.to_dict()
    portfolio = d.get("portfolio", {})
    strategies = d.get("strategies", {})
    regime_hmm = strategies.get("regime_hmm", {})
    return {
        "window": window["name"],
        "detector": "ensemble" if ensemble else "single",
        "elapsed_seconds": round(elapsed, 1),
        "portfolio_sharpe": portfolio.get("sharpe_ratio"),
        "portfolio_total_return": portfolio.get("total_return"),
        "portfolio_max_drawdown": portfolio.get("max_drawdown"),
        "regime_hmm_sharpe": regime_hmm.get("sharpe_ratio"),
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows", nargs="*", default=None,
        help="Subset of window names to run (default: all 3)",
    )
    parser.add_argument("--output", default="/tmp/regime_ensemble_calibration.json")
    args = parser.parse_args()

    windows = WINDOWS
    if args.windows:
        windows = [w for w in WINDOWS if w["name"] in args.windows]

    results: list[dict] = []
    for window in windows:
        for ensemble in (False, True):
            label = f"{window['name']} / detector={'ensemble' if ensemble else 'single'}"
            print(f"Running {label} ...", file=sys.stderr)
            try:
                results.append(_run_one(window, ensemble=ensemble))
            except Exception as exc:
                log.exception("run failed: %s", label)
                results.append({
                    "window": window["name"],
                    "detector": "ensemble" if ensemble else "single",
                    "error": str(exc),
                })

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2, default=str))

    print("\n=== regime-ensemble vs single-model detector (strategy_regime_weights=on) ===\n")
    print(f"{'window':22s} {'detector':9s} {'sharpe':>8s} {'ret':>8s} {'max_dd':>8s} {'regime_hmm':>11s}")
    print("-" * 75)
    for r in results:
        if "error" in r:
            print(f"{r['window']:22s} {r['detector']:9s} ERROR: {r['error']}")
            continue
        print(
            f"{r['window']:22s} {r['detector']:9s} "
            f"{r['portfolio_sharpe']:8.3f} {r['portfolio_total_return']:8.3f} "
            f"{r['portfolio_max_drawdown']:8.3f} "
            f"{(r['regime_hmm_sharpe'] or 0.0):11.3f}"
        )
    print(f"\nFull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
