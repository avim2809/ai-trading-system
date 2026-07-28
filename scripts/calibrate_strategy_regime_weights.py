#!/usr/bin/env python
"""A/B calibrate ``strategy_regime_weights`` on the portfolio-construction windows.

Compares portfolio Sharpe with regime weights disabled vs enabled (using the
example weights from ``config/settings.yaml``) on the same 3 historical windows
as ``docs/portfolio_construction_diagnosis.md``. Uses ``optimal`` signal
combination throughout (production default).

Usage:
    python scripts/calibrate_strategy_regime_weights.py
    python scripts/calibrate_strategy_regime_weights.py --output /tmp/regime_weights_calibration.json
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

from firm.backtest.run import execute_backtest  # noqa: E402
from firm.config import get_settings  # noqa: E402

log = logging.getLogger(__name__)

_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "NFLX", "ADBE", "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH",
    "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM",
]

# Production 10-strategy roster (gann/ml_prediction off).
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


def _run_one(
    window: dict,
    *,
    regime_weights_enabled: bool,
    weights_cfg: dict | None = None,
) -> dict:
    config = _base_config(window)
    if weights_cfg is not None:
        weights_cfg = dict(weights_cfg)
    else:
        weights_cfg = dict(get_settings().strategy_regime_weights or {})
    if regime_weights_enabled:
        weights_cfg = {**weights_cfg, "enabled": True}
    else:
        weights_cfg = {"enabled": False}
    config["strategy_regime_weights"] = weights_cfg

    report = execute_backtest(config)
    d = report.to_dict()
    portfolio = d.get("portfolio", {})
    strategies = d.get("strategies", {})
    best = None
    if strategies:
        best = max(
            strategies.items(),
            key=lambda kv: kv[1].get("sharpe_ratio", float("-inf")),
        )
    return {
        "window": window["name"],
        "regime_weights": "on" if regime_weights_enabled else "off",
        "portfolio_sharpe": portfolio.get("sharpe_ratio"),
        "portfolio_total_return": portfolio.get("total_return"),
        "portfolio_max_drawdown": portfolio.get("max_drawdown"),
        "best_component_strategy": best[0] if best else None,
        "best_component_sharpe": best[1].get("sharpe_ratio") if best else None,
        "strategy_sharpes": {k: v.get("sharpe_ratio") for k, v in strategies.items()},
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/tmp/strategy_regime_weights_calibration.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--weights-json",
        default=None,
        help="JSON file with strategy_regime_weights block (e.g. from suggest script)",
    )
    args = parser.parse_args()

    weights_cfg: dict | None = None
    if args.weights_json:
        payload = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
        weights_cfg = payload.get("strategy_regime_weights", payload)

    results: list[dict] = []
    for window in WINDOWS:
        for enabled in (False, True):
            label = f"{window['name']} / weights={'on' if enabled else 'off'}"
            print(f"Running {label} ...", file=sys.stderr)
            try:
                results.append(_run_one(
                    window,
                    regime_weights_enabled=enabled,
                    weights_cfg=weights_cfg,
                ))
            except Exception as exc:
                log.exception("run failed: %s", label)
                results.append({
                    "window": window["name"],
                    "regime_weights": "on" if enabled else "off",
                    "error": str(exc),
                })

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2, default=str))

    print("\n=== strategy_regime_weights calibration (optimal combination) ===\n")
    print(f"{'window':22s} {'weights':5s} {'port_sharpe':>11s} {'port_ret':>10s} {'max_dd':>8s}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['window']:22s} {r['regime_weights']:5s} ERROR: {r['error']}")
            continue
        print(
            f"{r['window']:22s} {r['regime_weights']:5s} "
            f"{(r['portfolio_sharpe'] or 0):>11.3f} "
            f"{(r['portfolio_total_return'] or 0):>10.3f} "
            f"{(r['portfolio_max_drawdown'] or 0):>8.3f}"
        )

    # Per-window delta summary
    print("\nDelta (on - off) portfolio Sharpe:")
    for window in WINDOWS:
        off = next(
            (x for x in results if x.get("window") == window["name"] and x.get("regime_weights") == "off" and "error" not in x),
            None,
        )
        on = next(
            (x for x in results if x.get("window") == window["name"] and x.get("regime_weights") == "on" and "error" not in x),
            None,
        )
        if off and on:
            delta = (on.get("portfolio_sharpe") or 0) - (off.get("portfolio_sharpe") or 0)
            print(f"  {window['name']}: {delta:+.3f}")

    print(f"\nFull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
