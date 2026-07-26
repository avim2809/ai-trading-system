#!/usr/bin/env python3
"""Gann follow-up: signal correlation (step 5) + marginal Sharpe (step 6).

Usage::

    python scripts/gann_followup_study.py
    python scripts/gann_followup_study.py --output runs/gann_followup
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from firm.backtest.firm_strategy import PitViewAdapter
from firm.backtest.run import execute_backtest
from firm.config import get_settings
from firm.data.pit_store import PointInTimeDataStore
from firm.runtime import load_prices
from firm.strategies.registry import get

log = logging.getLogger(__name__)

LIVE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META",
    "TSLA", "AVGO", "AMD", "CRM", "NFLX", "ADBE",
    "JPM", "GS", "BAC", "V", "MA",
    "JNJ", "UNH", "LLY",
    "XOM", "CVX",
    "SPY", "QQQ", "IWM",
]

BASE_STRATEGIES = [
    "momentum", "trend", "mean_reversion", "stat_arb", "multi_factor",
    "sentiment", "event_driven", "volatility_breakout", "seasonality", "regime_hmm",
]

GANN_VARIANTS: dict[str, dict[str, Any]] = {
    "angles_swing": {
        "sub_weights": {"angles": 0.5, "sq9": 0.0, "cycles": 0.0, "swing": 0.5, "retracement": 0.0},
    },
    "full_default": {},
    "legacy_five_component": {
        "swing_period": 2,
        "retracement_mean_revert": True,
        "sub_weights": {
            "angles": 0.25, "sq9": 0.15, "cycles": 0.15,
            "swing": 0.25, "retracement": 0.20,
        },
    },
}

HOLDOUT_START = "2024-07-01"
HOLDOUT_END = "2026-06-30"
CORR_START = "2022-01-01"
CORR_END = "2026-06-30"
MIN_SYMBOLS = 8


def _weekly_dates(trading_dates: pd.DatetimeIndex, start: str, end: str) -> list[datetime]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    dates = trading_dates[(trading_dates >= start_ts) & (trading_dates <= end_ts)]
    if len(dates) == 0:
        return []
    df = pd.DataFrame({"date": dates})
    df["yw"] = (
        df["date"].dt.isocalendar().year.astype(str)
        + "-"
        + df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
    )
    return [d.to_pydatetime() for d in df.groupby("yw", as_index=False)["date"].max()["date"]]


def _scores(strategy, pit_view) -> dict[str, float]:
    return {s.symbol: s.score for s in strategy.generate(pit_view)}


def run_correlation_study(
    pit_store: PointInTimeDataStore,
    universe: list[str],
    trading_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Step 5: pooled cross-sectional Spearman corr vs momentum & trend."""
    dates = _weekly_dates(trading_dates, CORR_START, CORR_END)
    momentum = get("momentum")()
    trend = get("trend")()
    gann_strats = {name: get("gann")(params=p) for name, p in GANN_VARIANTS.items()}

    results: dict[str, Any] = {}
    for gann_name, gann in gann_strats.items():
        vs_mom: list[float] = []
        vs_trend: list[float] = []
        vs_both: list[tuple[float, float]] = []

        for asof in dates:
            pit = PitViewAdapter(pit_store, asof, universe)
            g = _scores(gann, pit)
            m = _scores(momentum, pit)
            t = _scores(trend, pit)
            common = sorted(set(g) & set(m) & set(t))
            if len(common) < MIN_SYMBOLS:
                continue
            x = [g[s] for s in common]
            ym = [m[s] for s in common]
            yt = [t[s] for s in common]
            cm, _ = stats.spearmanr(x, ym)
            ct, _ = stats.spearmanr(x, yt)
            if np.isfinite(cm):
                vs_mom.append(float(cm))
            if np.isfinite(ct):
                vs_trend.append(float(ct))
            if np.isfinite(cm) and np.isfinite(ct):
                vs_both.append((float(cm), float(ct)))

        mom_arr = np.asarray(vs_mom) if vs_mom else np.array([])
        tr_arr = np.asarray(vs_trend) if vs_trend else np.array([])
        results[gann_name] = {
            "n_weeks": len(vs_both),
            "vs_momentum": {
                "mean": float(mom_arr.mean()) if len(mom_arr) else None,
                "std": float(mom_arr.std(ddof=1)) if len(mom_arr) > 1 else None,
                "pct_above_0.55": float((mom_arr > 0.55).mean()) if len(mom_arr) else None,
                "pct_above_0.60": float((mom_arr > 0.60).mean()) if len(mom_arr) else None,
            },
            "vs_trend": {
                "mean": float(tr_arr.mean()) if len(tr_arr) else None,
                "std": float(tr_arr.std(ddof=1)) if len(tr_arr) > 1 else None,
                "pct_above_0.55": float((tr_arr > 0.55).mean()) if len(tr_arr) else None,
                "pct_above_0.60": float((tr_arr > 0.60).mean()) if len(tr_arr) else None,
            },
        }
        log.info(
            "Correlation %s: vs_mom=%.3f vs_trend=%.3f (%d weeks)",
            gann_name,
            results[gann_name]["vs_momentum"]["mean"] or 0.0,
            results[gann_name]["vs_trend"]["mean"] or 0.0,
            results[gann_name]["n_weeks"],
        )
    return results


def _load_live_backtest_config() -> dict[str, Any]:
    live_path = Path("config/live.yaml")
    live = yaml.safe_load(live_path.read_text(encoding="utf-8")) if live_path.exists() else {}
    settings = get_settings()
    risk = dict(live.get("risk") or {})
    costs = live.get("costs") or {}
    return {
        "data_source": "cache",
        "universe_symbols": LIVE_UNIVERSE,
        "initial_capital": live.get("initial_capital", 1_000_000),
        "commission_pct": costs.get("commission_pct", 0.0005),
        "slippage_pct": costs.get("slippage_pct", 0.0005),
        "rebalance_frequency": live.get("rebalance_frequency", "daily"),
        "warmup_days": 400,
        "strategy_params": live.get("strategy_params") or settings.strategy_params or {},
        "regime_overlay": risk.pop("regime_overlay", None),
        "allocation_method": live.get("allocation_method", settings.allocation_method),
        "kelly_fraction": live.get("kelly_fraction", settings.kelly_fraction),
        "signal_combination": live.get("signal_combination") or settings.signal_combination,
        "agent_modes": {"technical_analyst": "quant", "fundamental_analyst": "quant",
                        "sentiment_analyst": "quant", "bull_analyst": "quant",
                        "bear_analyst": "quant", "trader": "quant", "risk_manager": "quant"},
        **risk,
    }


def run_marginal_sharpe_study(
    start: str,
    end: str,
) -> dict[str, Any]:
    """Step 6: full pipeline backtest with and without Gann."""
    base_cfg = _load_live_backtest_config()
    base_cfg["start_date"] = start
    base_cfg["end_date"] = end

    scenarios: dict[str, dict[str, Any]] = {
        "baseline_10strat": {
            **base_cfg,
            "strategies": list(BASE_STRATEGIES),
        },
        "with_gann_angles_swing": {
            **base_cfg,
            "strategies": list(BASE_STRATEGIES) + ["gann"],
            "strategy_params": {
                **(base_cfg.get("strategy_params") or {}),
                "gann": GANN_VARIANTS["angles_swing"],
            },
        },
        "with_gann_legacy": {
            **base_cfg,
            "strategies": list(BASE_STRATEGIES) + ["gann"],
            "strategy_params": {
                **(base_cfg.get("strategy_params") or {}),
                "gann": GANN_VARIANTS["legacy_five_component"],
            },
        },
    }

    results: dict[str, Any] = {}
    for name, cfg in scenarios.items():
        log.info("Backtest %s (%s → %s)...", name, start, end)
        report = execute_backtest(cfg)
        port = report.portfolio_summary()
        bench = report.benchmark_summary()
        results[name] = {
            "sharpe_ratio": port.get("sharpe_ratio"),
            "sortino_ratio": port.get("sortino_ratio"),
            "total_return": port.get("total_return"),
            "max_drawdown": port.get("max_drawdown"),
            "annualized_return": port.get("annualized_return"),
            "volatility": port.get("volatility"),
            "benchmark_alpha": bench.get("alpha"),
            "benchmark_information_ratio": bench.get("information_ratio"),
            "n_trading_days": len(report.returns),
        }
        log.info(
            "  %s: Sharpe=%.3f total_return=%.2f%% max_dd=%.2f%%",
            name,
            results[name]["sharpe_ratio"] or 0.0,
            (results[name]["total_return"] or 0.0) * 100,
            (results[name]["max_drawdown"] or 0.0) * 100,
        )

    baseline_sharpe = results["baseline_10strat"].get("sharpe_ratio")
    for key in ("with_gann_angles_swing", "with_gann_legacy"):
        g_sharpe = results[key].get("sharpe_ratio")
        if baseline_sharpe is not None and g_sharpe is not None:
            results[key]["delta_sharpe_vs_baseline"] = g_sharpe - baseline_sharpe
        base_ret = results["baseline_10strat"].get("total_return")
        g_ret = results[key].get("total_return")
        if base_ret is not None and g_ret is not None:
            results[key]["delta_total_return_vs_baseline"] = g_ret - base_ret

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Gann follow-up correlation + marginal Sharpe")
    parser.add_argument("--output", default=None)
    parser.add_argument("--holdout-start", default=HOLDOUT_START)
    parser.add_argument("--holdout-end", default=HOLDOUT_END)
    parser.add_argument(
        "--full-sample",
        action="store_true",
        help="Also run 2022–2026 backtests (3× slower; default is holdout only)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    prices = load_prices(get_settings())
    prices["date"] = pd.to_datetime(prices["date"])
    universe = [s for s in LIVE_UNIVERSE if s in set(prices["symbol"].unique())]
    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices)
    trading_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))

    log.info("Step 5: signal correlation (%s → %s)", CORR_START, CORR_END)
    correlation = run_correlation_study(pit_store, universe, trading_dates)

    log.info("Step 6: marginal Sharpe holdout (%s → %s)", args.holdout_start, args.holdout_end)
    marginal_sharpe_holdout = run_marginal_sharpe_study(args.holdout_start, args.holdout_end)

    marginal_sharpe_full = None
    if args.full_sample:
        log.info("Step 6b: marginal Sharpe full sample (2022-01-01 → 2026-06-30)")
        marginal_sharpe_full = run_marginal_sharpe_study("2022-01-01", "2026-06-30")

    payload = {
        "study": "gann_followup_steps_5_and_6",
        "universe": universe,
        "correlation_period": {"start": CORR_START, "end": CORR_END},
        "holdout_period": {"start": args.holdout_start, "end": args.holdout_end},
        "gann_variants_tested": list(GANN_VARIANTS.keys()),
        "baseline_strategies": BASE_STRATEGIES,
        "correlation": correlation,
        "marginal_sharpe_holdout": marginal_sharpe_holdout,
        "marginal_sharpe_full": marginal_sharpe_full,
        "code_state": {
            "retracement_sign": "range_momentum (flipped per external research)",
            "default_sub_weights": "angles 40%, retracement 40%, swing 20%; sq9/cycles 0",
            "default_swing_period": 4,
        },
        "prior_ic_study": "runs/gann_ic_post_fix/results.json",
    }

    out_dir = Path(args.output or f"runs/gann_followup_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("STEP 5 — GANN vs MOMENTUM/TREND (weekly cross-sectional Spearman)")
    print("=" * 72)
    for name, row in correlation.items():
        m = row["vs_momentum"]["mean"]
        t = row["vs_trend"]["mean"]
        print(
            f"{name:<25} vs_momentum={m:+.3f}  vs_trend={t:+.3f}  "
            f"(n={row['n_weeks']} weeks, >0.55 mom={row['vs_momentum']['pct_above_0.55']:.0%})"
        )

    print("\n" + "=" * 72)
    print(f"STEP 6 — MARGINAL SHARPE (holdout {args.holdout_start} → {args.holdout_end})")
    print("=" * 72)
    for name, row in marginal_sharpe_holdout.items():
        ds = row.get("delta_sharpe_vs_baseline")
        ds_s = f" ΔSharpe={ds:+.3f}" if ds is not None else ""
        print(
            f"{name:<28} Sharpe={row.get('sharpe_ratio', 0):.3f}  "
            f"Return={row.get('total_return', 0):.2%}  MaxDD={row.get('max_drawdown', 0):.2%}{ds_s}"
        )

    print(f"\nFull results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
