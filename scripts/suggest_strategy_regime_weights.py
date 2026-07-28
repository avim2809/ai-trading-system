#!/usr/bin/env python
"""Suggest ``strategy_regime_weights`` from empirical strategy × regime performance.

Runs a train-window backtest (weights off), labels each day with Bull/Bear/Chop
via rolling HMM on SPY (no look-ahead), then maps per-strategy conditional
Sharpe in each regime to soft multipliers. Output is a **research draft** —
always validate with ``scripts/calibrate_strategy_regime_weights.py`` on
held-out windows before enabling live.

Usage:
    python scripts/suggest_strategy_regime_weights.py
    python scripts/suggest_strategy_regime_weights.py --validate
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.backtest.run import execute_backtest  # noqa: E402
from firm.config import get_settings  # noqa: E402
from firm.regime.features import compute_regime_features  # noqa: E402
from firm.regime.model import BEAR, BULL, CHOP, GaussianRegimeModel  # noqa: E402
from firm.runtime import load_prices  # noqa: E402

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

_HOLDOUT = {
    "name": "holdout_2025_h1",
    "start_date": "2025-01-01",
    "end_date": "2026-06-30",
}


def _base_config(*, start_date: str, end_date: str) -> dict:
    settings = get_settings()
    return {
        "data_source": "cache",
        "start_date": start_date,
        "end_date": end_date,
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
        **settings.risk.model_dump(),
        "signal_combination": {"method": "optimal"},
        "strategy_regime_weights": {"enabled": False},
    }


def _load_spy_ohlcv() -> pd.DataFrame:
    prices = load_prices(get_settings())
    spy = prices[prices["symbol"].astype(str) == "SPY"].copy()
    if spy.empty:
        raise ValueError("SPY not found in price cache")
    spy["date"] = pd.to_datetime(spy["date"])
    return spy.sort_values("date")


def _label_regimes(
    ohlcv: pd.DataFrame,
    *,
    lookback_days: int,
    retrain_frequency: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Point-in-time regime labels for each date in [start, end]."""
    df = ohlcv.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    features = compute_regime_features(df)
    if features.empty:
        raise ValueError("Could not compute regime features for SPY")

    feature_dates = pd.DatetimeIndex(pd.to_datetime(df.loc[features.index, "date"]))
    features.index = feature_dates.normalize()
    dates = features.index

    model = GaussianRegimeModel(n_states=3, random_state=42)
    labels: dict[pd.Timestamp, str] = {}
    bars_since_fit = retrain_frequency  # force initial fit

    for i, dt in enumerate(dates):
        if dt < start or dt > end:
            continue
        if bars_since_fit >= retrain_frequency:
            window_start = max(0, i - lookback_days + 1)
            X = features.iloc[window_start : i + 1].to_numpy()
            if len(X) < model.n_states:
                continue
            try:
                model.fit(X)
            except Exception as exc:
                log.debug("regime fit skip at %s (%s)", dt.date(), exc)
                continue
            bars_since_fit = 0
        bars_since_fit += 1
        if not model.fitted:
            continue
        window_start = max(0, i - lookback_days + 1)
        X = features.iloc[window_start : i + 1].to_numpy()
        if len(X) < model.n_states:
            continue
        try:
            labels[dt.normalize()] = model.classify(X).label
        except Exception as exc:
            log.debug("regime classify skip at %s (%s)", dt.date(), exc)

    return pd.Series(labels, name="regime")


def _sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 5:
        return 0.0
    std = float(r.std())
    if std <= 0:
        return 0.0
    return float(r.mean() / std * math.sqrt(252))


def _suggest_multipliers(
    strategy_returns: dict[str, pd.Series],
    regime_by_date: pd.Series,
    *,
    scale: float = 0.15,
    floor: float = 0.7,
    cap: float = 1.3,
) -> dict[str, dict[str, float]]:
    regimes = [BULL, BEAR, CHOP]
    sharpes: dict[str, dict[str, float]] = {r: {} for r in regimes}

    for strategy, series in strategy_returns.items():
        aligned = series.copy()
        aligned.index = pd.DatetimeIndex(aligned.index).normalize()
        for regime in regimes:
            mask_dates = regime_by_date[regime_by_date == regime].index
            subset = aligned.reindex(mask_dates).dropna()
            sharpes[regime][strategy] = _sharpe(subset)

    weights: dict[str, dict[str, float]] = {r: {} for r in regimes}
    for regime in regimes:
        vals = list(sharpes[regime].values())
        median = float(np.median(vals)) if vals else 0.0
        for strategy, s in sharpes[regime].items():
            delta = s - median
            mult = 1.0 + scale * delta
            mult = max(floor, min(cap, mult))
            if abs(mult - 1.0) > 0.02:
                weights[regime][strategy] = round(mult, 3)

    return weights


def _portfolio_sharpe(config: dict) -> float:
    report = execute_backtest(config)
    return float(report.portfolio_summary().get("sharpe_ratio", 0.0) or 0.0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--retrain-frequency", type=int, default=21)
    parser.add_argument("--scale", type=float, default=0.15, help="Sharpe delta → multiplier gain")
    parser.add_argument("--output", default="/tmp/suggested_strategy_regime_weights.json")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="A/B suggested weights vs off on hold-out window",
    )
    args = parser.parse_args()

    spy = _load_spy_ohlcv()
    start = pd.Timestamp(args.train_start)
    end = pd.Timestamp(args.train_end)
    log.info("Labeling regimes on SPY %s → %s", start.date(), end.date())
    regime_by_date = _label_regimes(
        spy,
        lookback_days=args.lookback_days,
        retrain_frequency=args.retrain_frequency,
        start=start,
        end=end,
    )
    log.info("Regime labels: %d days (%s)", len(regime_by_date), regime_by_date.value_counts().to_dict())

    train_cfg = _base_config(start_date=args.train_start, end_date=args.train_end)
    log.info("Train backtest %s → %s", args.train_start, args.train_end)
    report = execute_backtest(train_cfg)
    strategy_returns = report.attribution.get_all_strategy_returns(min_points=10)
    if not strategy_returns:
        log.error("No strategy return series from train backtest")
        return 1

    weights = _suggest_multipliers(
        strategy_returns, regime_by_date, scale=args.scale,
    )
    cfg = {
        "enabled": False,
        "benchmark_symbol": "SPY",
        "lookback_days": args.lookback_days,
        "retrain_frequency": args.retrain_frequency,
        "weights": weights,
    }
    result = {
        "train_window": {"start": args.train_start, "end": args.train_end},
        "strategy_regime_weights": cfg,
        "regime_day_counts": regime_by_date.value_counts().to_dict(),
    }

    if args.validate:
        holdout_off = _base_config(
            start_date=_HOLDOUT["start_date"], end_date=_HOLDOUT["end_date"],
        )
        holdout_on = {
            **holdout_off,
            "strategy_regime_weights": {**cfg, "enabled": True},
        }
        log.info("Hold-out validation on %s", _HOLDOUT["name"])
        off_sh = _portfolio_sharpe(holdout_off)
        on_sh = _portfolio_sharpe(holdout_on)
        result["holdout_validation"] = {
            "window": _HOLDOUT,
            "portfolio_sharpe_off": off_sh,
            "portfolio_sharpe_on": on_sh,
            "delta": on_sh - off_sh,
        }
        log.info(
            "Hold-out Sharpe off=%.3f on=%.3f delta=%+.3f",
            off_sh, on_sh, on_sh - off_sh,
        )

    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
