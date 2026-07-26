#!/usr/bin/env python3
"""Walk-forward IC study for the Gann strategy with component ablation.

Evaluates cross-sectional Spearman IC between Gann scores and 10-day forward
returns on the live 25-name universe, across walk-forward OOS folds.

Usage::

    python scripts/gann_ic_study.py
    python scripts/gann_ic_study.py --output runs/gann_ic_study
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from firm.backtest.firm_strategy import PitViewAdapter
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

FORWARD_HORIZON_DAYS = 10
MIN_SYMBOLS_FOR_IC = 8

# Component ablation variants (params passed to GannStrategy).
VARIANTS: dict[str, dict[str, Any]] = {
    "full_default": {},
    "legacy_five_component": {
        "swing_period": 2,
        "sub_weights": {
            "angles": 0.25, "sq9": 0.15, "cycles": 0.15,
            "swing": 0.25, "retracement": 0.20,
        },
        "retracement_mean_revert": True,
    },
    "redesigned_strict": {
        "min_confidence": 0.20,
        "trend_filter_threshold": 0.25,
    },
    "angles_only": {
        "sub_weights": {"angles": 1.0, "sq9": 0.0, "cycles": 0.0, "swing": 0.0, "retracement": 0.0},
    },
    "swing_only": {
        "sub_weights": {"angles": 0.0, "sq9": 0.0, "cycles": 0.0, "swing": 1.0, "retracement": 0.0},
    },
    "swing_only_period2": {
        "swing_period": 2,
        "sub_weights": {"angles": 0.0, "sq9": 0.0, "cycles": 0.0, "swing": 1.0, "retracement": 0.0},
    },
    "sq9_only": {
        "sub_weights": {"angles": 0.0, "sq9": 1.0, "cycles": 0.0, "swing": 0.0, "retracement": 0.0},
    },
    "cycles_only": {
        "sub_weights": {"angles": 0.0, "sq9": 0.0, "cycles": 1.0, "swing": 0.0, "retracement": 0.0},
    },
    "retracement_only": {
        "sub_weights": {"angles": 0.0, "sq9": 0.0, "cycles": 0.0, "swing": 0.0, "retracement": 1.0},
    },
    "angles_swing": {
        "sub_weights": {"angles": 0.5, "sq9": 0.0, "cycles": 0.0, "swing": 0.5, "retracement": 0.0},
    },
}


@dataclass
class FoldResult:
    fold: int
    test_start: str
    test_end: str
    mean_ic: float
    ic_ir: float
    hit_rate: float
    n_weeks: int
    weekly_ics: list[float] = field(default_factory=list)
    ls_ann_return: float | None = None
    ls_sharpe: float | None = None


def _walk_forward_splits(
    start: str,
    end: str,
    n_splits: int = 5,
    train_pct: float = 0.70,
) -> list[tuple[int, str, str, str, str]]:
    """Return (fold_idx, train_start, train_end, test_start, test_end)."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    total_days = (e - s).days
    window_days = total_days // n_splits
    splits: list[tuple[int, str, str, str, str]] = []
    for i in range(n_splits):
        w_start = s + pd.Timedelta(days=i * window_days)
        w_end = w_start + pd.Timedelta(days=window_days)
        if w_end > e:
            w_end = e
        train_days = int(window_days * train_pct)
        train_end = w_start + pd.Timedelta(days=train_days)
        test_start = train_end + pd.Timedelta(days=1)
        splits.append((
            i + 1,
            w_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            w_end.strftime("%Y-%m-%d"),
        ))
    return splits


def _weekly_rebalance_dates(
    trading_dates: pd.DatetimeIndex,
    start: str,
    end: str,
) -> list[datetime]:
    """One rebalance per ISO week inside [start, end]."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (trading_dates >= start_ts) & (trading_dates <= end_ts)
    dates = trading_dates[mask]
    if len(dates) == 0:
        return []
    df = pd.DataFrame({"date": dates})
    df["year_week"] = df["date"].dt.isocalendar().year.astype(str) + "-" + df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
    picked = df.groupby("year_week", as_index=False)["date"].max()
    return [d.to_pydatetime() for d in picked["date"]]


def _forward_returns(
    prices: pd.DataFrame,
    asof: datetime,
    symbols: list[str],
    horizon: int = FORWARD_HORIZON_DAYS,
) -> dict[str, float]:
    """10-trading-day forward return per symbol from adj_close."""
    asof_ts = pd.Timestamp(asof)
    out: dict[str, float] = {}
    for sym in symbols:
        sym_df = prices[prices["symbol"] == sym].sort_values("date")
        dates = sym_df["date"].values
        idx = np.searchsorted(dates, np.datetime64(asof_ts), side="right") - 1
        if idx < 0:
            continue
        fwd_idx = idx + horizon
        if fwd_idx >= len(sym_df):
            continue
        p0 = float(sym_df.iloc[idx]["adj_close"] if "adj_close" in sym_df.columns else sym_df.iloc[idx]["close"])
        p1 = float(sym_df.iloc[fwd_idx]["adj_close"] if "adj_close" in sym_df.columns else sym_df.iloc[fwd_idx]["close"])
        if p0 > 0:
            out[sym] = p1 / p0 - 1.0
    return out


def _long_short_spread(
    scores: dict[str, float],
    fwd: dict[str, float],
) -> float | None:
    """Top vs bottom tercile equal-weight spread return."""
    common = [s for s in scores if s in fwd]
    if len(common) < MIN_SYMBOLS_FOR_IC:
        return None
    ranked = sorted(common, key=lambda s: scores[s])
    n = len(ranked)
    k = max(1, n // 3)
    bottom = ranked[:k]
    top = ranked[-k:]
    long_ret = float(np.mean([fwd[s] for s in top]))
    short_ret = float(np.mean([fwd[s] for s in bottom]))
    return long_ret - short_ret


def evaluate_variant(
    variant: str,
    params: dict[str, Any],
    pit_store: PointInTimeDataStore,
    prices: pd.DataFrame,
    universe: list[str],
    trading_dates: pd.DatetimeIndex,
    folds: list[tuple[int, str, str, str, str]],
) -> dict[str, Any]:
    strategy = get("gann")(params=params)
    fold_results: list[FoldResult] = []
    all_ics: list[float] = []
    all_ls: list[float] = []

    for fold_idx, _train_s, _train_e, test_s, test_e in folds:
        rebal_dates = _weekly_rebalance_dates(trading_dates, test_s, test_e)
        weekly_ics: list[float] = []
        weekly_ls: list[float] = []

        for asof in rebal_dates:
            pit_view = PitViewAdapter(pit_store, asof, universe)
            signals = strategy.generate(pit_view)
            if not signals:
                continue
            scores = {s.symbol: s.score for s in signals}
            fwd = _forward_returns(prices, asof, list(scores.keys()))
            common = [sym for sym in scores if sym in fwd]
            if len(common) < MIN_SYMBOLS_FOR_IC:
                continue
            x = [scores[s] for s in common]
            y = [fwd[s] for s in common]
            ic, _ = stats.spearmanr(x, y)
            if np.isfinite(ic):
                weekly_ics.append(float(ic))
            spread = _long_short_spread(scores, fwd)
            if spread is not None and np.isfinite(spread):
                weekly_ls.append(spread)

        if not weekly_ics:
            fold_results.append(FoldResult(
                fold=fold_idx, test_start=test_s, test_end=test_e,
                mean_ic=0.0, ic_ir=0.0, hit_rate=0.0, n_weeks=0,
            ))
            continue

        arr = np.asarray(weekly_ics)
        mean_ic = float(arr.mean())
        ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 1e-9 else 0.0
        hit_rate = float((arr > 0).mean())

        ls_ann = ls_sharpe = None
        if len(weekly_ls) >= 2:
            ls = np.asarray(weekly_ls)
            all_ls.extend(weekly_ls)
            # ~52 rebalances/year; each spread is a 10d hold → scale cautiously
            ann_factor = 52.0
            ls_ann = float(ls.mean() * ann_factor)
            ls_std = float(ls.std(ddof=1))
            ls_sharpe = float(ls.mean() / ls_std * np.sqrt(ann_factor)) if ls_std > 1e-9 else 0.0

        all_ics.extend(weekly_ics)
        fold_results.append(FoldResult(
            fold=fold_idx,
            test_start=test_s,
            test_end=test_e,
            mean_ic=mean_ic,
            ic_ir=ic_ir,
            hit_rate=hit_rate,
            n_weeks=len(weekly_ics),
            weekly_ics=weekly_ics,
            ls_ann_return=ls_ann,
            ls_sharpe=ls_sharpe,
        ))

    pooled = np.asarray(all_ics) if all_ics else np.array([0.0])
    pooled_ls = np.asarray(all_ls) if all_ls else None
    summary = {
        "variant": variant,
        "params": params,
        "pooled_mean_ic": float(pooled.mean()) if len(all_ics) else None,
        "pooled_ic_ir": (
            float(pooled.mean() / pooled.std(ddof=1))
            if len(all_ics) > 1 and pooled.std(ddof=1) > 1e-9
            else None
        ),
        "pooled_hit_rate": float((pooled > 0).mean()) if len(all_ics) else None,
        "n_ic_observations": len(all_ics),
        "pooled_ls_ann_return": (
            float(pooled_ls.mean() * 52.0) if pooled_ls is not None and len(pooled_ls) else None
        ),
        "pooled_ls_sharpe": (
            float(pooled_ls.mean() / pooled_ls.std(ddof=1) * np.sqrt(52.0))
            if pooled_ls is not None and len(pooled_ls) > 1 and pooled_ls.std(ddof=1) > 1e-9
            else None
        ),
        "folds": [
            {
                "fold": f.fold,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "mean_ic": f.mean_ic,
                "ic_ir": f.ic_ir,
                "hit_rate": f.hit_rate,
                "n_weeks": f.n_weeks,
                "ls_ann_return": f.ls_ann_return,
                "ls_sharpe": f.ls_sharpe,
            }
            for f in fold_results
        ],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Gann walk-forward IC ablation study")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output", default=None, help="Output directory (default: runs/gann_ic_<ts>)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    prices = load_prices(get_settings())
    prices["date"] = pd.to_datetime(prices["date"])
    universe = [s for s in LIVE_UNIVERSE if s in set(prices["symbol"].unique())]
    log.info("Universe: %d symbols, prices %s → %s", len(universe), prices["date"].min().date(), prices["date"].max().date())

    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices)

    trading_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    folds = _walk_forward_splits(args.start, args.end, n_splits=args.n_splits)
    log.info("Walk-forward: %d folds, %s → %s", len(folds), args.start, args.end)

    results: list[dict[str, Any]] = []
    for name, params in VARIANTS.items():
        log.info("Evaluating variant: %s", name)
        summary = evaluate_variant(
            name, params, pit_store, prices, universe, trading_dates, folds,
        )
        results.append(summary)
        log.info(
            "  %s: mean_ic=%.4f hit_rate=%.1f%% n=%d ls_ann=%s",
            name,
            summary["pooled_mean_ic"] or 0.0,
            (summary["pooled_hit_rate"] or 0.0) * 100,
            summary["n_ic_observations"],
            f"{summary['pooled_ls_ann_return']:.2%}" if summary.get("pooled_ls_ann_return") is not None else "n/a",
        )

    results.sort(key=lambda r: r.get("pooled_mean_ic") or -999, reverse=True)

    out_dir = Path(args.output or f"runs/gann_ic_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": "gann_walk_forward_ic_ablation",
        "universe": universe,
        "start": args.start,
        "end": args.end,
        "forward_horizon_days": FORWARD_HORIZON_DAYS,
        "n_splits": args.n_splits,
        "variants": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("GANN WALK-FORWARD IC STUDY (25-name universe, 10d forward return)")
    print("=" * 88)
    print(f"{'Variant':<20} {'Mean IC':>9} {'IC IR':>8} {'Hit%':>7} {'N weeks':>8} {'L/S ann':>10} {'L/S Sharpe':>11}")
    print("-" * 88)
    for r in results:
        mic = r.get("pooled_mean_ic")
        iir = r.get("pooled_ic_ir")
        hit = r.get("pooled_hit_rate")
        ls = r.get("pooled_ls_ann_return")
        lss = r.get("pooled_ls_sharpe")
        mic_s = f"{mic:>9.4f}" if mic is not None else f"{'n/a':>9}"
        iir_s = f"{iir:>8.3f}" if iir is not None else f"{'n/a':>8}"
        hit_s = f"{hit * 100:>6.1f}%" if hit is not None else f"{'n/a':>7}"
        ls_s = f"{ls:>9.2%}" if ls is not None else f"{'n/a':>10}"
        lss_s = f"{lss:>11.2f}" if lss is not None else f"{'n/a':>11}"
        print(
            f"{r['variant']:<20} {mic_s} {iir_s} {hit_s} "
            f"{r['n_ic_observations']:>8} {ls_s} {lss_s}"
        )
    print("=" * 88)
    print(f"Full results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
