#!/usr/bin/env python
"""A/B the danelfin_market_percentile strategy: baseline roster vs +1 strategy.

Single-window validation (not the full 3-window harness other Danelfin
strategies got) — a deliberate scope decision given a 10K Danelfin API
calls/month budget already shared with danelfin_live_signals and the
best-stocks-arm daily job. See
scripts/fetch_market_percentile_calibration_data.py's docstring for the
full cost rationale; run that script FIRST to populate the
combined/market_percentile cache this script reads from (zero additional
API cost here — this script only reads the cache, matching how
calibrate_danelfin_ai_score.py works).

Usage:
    python scripts/fetch_market_percentile_calibration_data.py   # once, real API cost
    python scripts/calibrate_danelfin_market_percentile.py       # free, reuses the cache
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

# Same window used to populate the calibration cache (see
# fetch_market_percentile_calibration_data.py's WINDOW_START/WINDOW_END) —
# matches danelfin_ai_score's own primary "run_18mo_2025_2026" window.
WINDOW = {"name": "run_18mo_2025_2026", "start_date": "2025-01-01", "end_date": "2026-06-30"}


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


def _run_one(window: dict, *, with_market_percentile: bool) -> dict:
    strategies = _BASELINE_STRATEGIES + (
        ["danelfin_market_percentile"] if with_market_percentile else []
    )
    config = _base_config(window, strategies)

    t0 = time.time()
    report = execute_backtest(config)
    elapsed = time.time() - t0
    d = report.to_dict()
    portfolio = d.get("portfolio", {})
    strategies_d = d.get("strategies", {})
    mp = strategies_d.get("danelfin_market_percentile", {})
    overfitting = d.get("overfitting", {})
    return {
        "window": window["name"],
        "arm": "with_market_percentile" if with_market_percentile else "baseline",
        "elapsed_seconds": round(elapsed, 1),
        "portfolio_sharpe": portfolio.get("sharpe_ratio"),
        "portfolio_total_return": portfolio.get("total_return"),
        "portfolio_max_drawdown": portfolio.get("max_drawdown"),
        "danelfin_market_percentile_sharpe": mp.get("sharpe_ratio"),
        "deflated_sharpe": overfitting.get("deflated_sharpe_ratio"),
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/danelfin_market_percentile_calibration.json")
    args = parser.parse_args()

    results: list[dict] = []
    for with_mp in (False, True):
        label = f"{WINDOW['name']} / {'with_market_percentile' if with_mp else 'baseline'}"
        print(f"Running {label} ...", file=sys.stderr)
        try:
            results.append(_run_one(WINDOW, with_market_percentile=with_mp))
        except Exception as exc:
            log.exception("run failed: %s", label)
            results.append({
                "window": WINDOW["name"],
                "arm": "with_market_percentile" if with_mp else "baseline",
                "error": str(exc),
            })

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2, default=str))

    print("\n=== danelfin_market_percentile: baseline vs +1 strategy (single window) ===\n")
    print(f"{'arm':24s} {'sharpe':>8s} {'ret':>8s} {'max_dd':>8s} {'mp_sharpe':>10s}")
    print("-" * 64)
    for r in results:
        if "error" in r:
            print(f"{r['arm']:24s} ERROR: {r['error']}")
            continue
        print(
            f"{r['arm']:24s} "
            f"{r['portfolio_sharpe']:8.3f} {r['portfolio_total_return']:8.3f} "
            f"{r['portfolio_max_drawdown']:8.3f} "
            f"{(r['danelfin_market_percentile_sharpe'] or 0.0):10.3f}"
        )
    print(f"\nFull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
