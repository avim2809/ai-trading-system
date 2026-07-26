#!/usr/bin/env python3
"""Experiment 2: Gann price-derived squaring event study (weekly bars).

Tests whether sqrt(normalize(pivot_price)) weeks from a major pivot predict
swing timing and direction.

Usage::

    python scripts/gann_squaring_event_study.py --threshold-sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from gann_weekly_helpers import (
    DEFAULT_UNIVERSE,
    PivotRecord,
    PivotType,
    aggregate_to_weekly,
    detect_major_pivots_weekly,
    load_prices,
    normalize_price,
    pivots_to_records,
)


def _within_days(a: pd.Timestamp, b: pd.Timestamp, tolerance_days: int) -> bool:
    return abs((a - b).days) <= tolerance_days


def _build_range_projections(pivots: list[PivotRecord]) -> list[tuple[pd.Timestamp, PivotType, str]]:
    """Range squaring targets from consecutive pivot pairs."""
    out: list[tuple[pd.Timestamp, PivotType, str]] = []
    for (d_a, p_a, _), (d_b, p_b, ptype_b) in zip(pivots, pivots[1:]):
        rng = abs(p_b - p_a)
        if rng <= 0:
            continue
        weeks = int(round(np.sqrt(normalize_price(rng))))
        target = pd.Timestamp(d_b) + pd.Timedelta(days=weeks * 7)
        out.append((target, ptype_b, "range"))
    return out


def _squaring_at_asof(
    asof: pd.Timestamp,
    pivots: list[PivotRecord],
    tolerance_days: int,
) -> dict[str, Any]:
    """Compute convergence and directional score at a single as-of date."""
    hits = 0
    total = 0
    score_num = 0.0
    score_den = 0.0

    hits_sqrt = total_sqrt = 0
    hits_price = total_price = 0
    hits_range = total_range = 0

    # Per-pivot sqrt and price projections
    for pdate, price, ptype in pivots:
        p_norm = normalize_price(price)
        weeks = int(round(np.sqrt(p_norm)))
        sqrt_target = pd.Timestamp(pdate) + pd.Timedelta(days=weeks * 7)
        total += 1
        total_sqrt += 1
        if _within_days(asof, sqrt_target, tolerance_days):
            hits += 1
            hits_sqrt += 1
            w = 1.0 / (max((asof - pd.Timestamp(pdate)).days / 7.0, 0) + 1.0)
            score_num += w if ptype == "low" else -w
            score_den += w

        if p_norm <= 365:
            price_target = pd.Timestamp(pdate) + pd.Timedelta(days=int(round(p_norm)))
            total += 1
            total_price += 1
            if _within_days(asof, price_target, tolerance_days):
                hits += 1
                hits_price += 1
                w = 1.0 / (max((asof - pd.Timestamp(pdate)).days / 7.0, 0) + 1.0)
                score_num += w if ptype == "low" else -w
                score_den += w

    # Range projections from consecutive pivots
    for target, ptype, _ in _build_range_projections(pivots):
        total += 1
        total_range += 1
        if _within_days(asof, target, tolerance_days):
            hits += 1
            hits_range += 1
            w = 0.5  # range proj has no single pivot date for recency; flat weight
            score_num += w if ptype == "low" else -w
            score_den += w

    convergence = hits / total if total else 0.0
    score = score_num / score_den if score_den > 0 else 0.0
    score = float(np.clip(score, -1.0, 1.0))

    def _ratio(h: int, t: int) -> float:
        return h / t if t else 0.0

    return {
        "convergence": convergence,
        "squaring_score": score,
        "sqrt_only": _ratio(hits_sqrt, total_sqrt),
        "price_only": _ratio(hits_price, total_price),
        "range_only": _ratio(hits_range, total_range),
    }


def walk_forward_squaring(
    weekly: pd.DataFrame,
    *,
    lookback_weeks: int,
    horizon_weeks: int,
    pivot_order: int,
    tolerance_days: int,
) -> dict[str, np.ndarray]:
    dates = weekly["week_start_date"].values
    closes = weekly["adj_close"].values.astype(float)
    highs = weekly["high"].values.astype(float)
    lows = weekly["low"].values.astype(float)
    n = len(weekly)

    conv = np.zeros(n, dtype=float)
    score = np.zeros(n, dtype=float)
    conv_sqrt = np.zeros(n, dtype=float)
    conv_price = np.zeros(n, dtype=float)
    conv_range = np.zeros(n, dtype=float)
    prior_ret = np.full(n, np.nan)
    fwd_ret = np.full(n, np.nan)
    reversal = np.full(n, np.nan)

    for i in range(lookback_weeks, n - horizon_weeks):
        asof = pd.Timestamp(dates[i])
        ph, pl = detect_major_pivots_weekly(
            highs[: i + 1], lows[: i + 1], dates[: i + 1], order=pivot_order,
        )
        pivots = pivots_to_records(ph, pl)
        sq = _squaring_at_asof(asof, pivots, tolerance_days)
        conv[i] = sq["convergence"]
        score[i] = sq["squaring_score"]
        conv_sqrt[i] = sq["sqrt_only"]
        conv_price[i] = sq["price_only"]
        conv_range[i] = sq["range_only"]

        if i >= horizon_weeks and closes[i - horizon_weeks] > 0:
            prior_ret[i] = closes[i] / closes[i - horizon_weeks] - 1.0
        if closes[i] > 0:
            fwd_ret[i] = closes[i + horizon_weeks] / closes[i] - 1.0
        if np.isfinite(prior_ret[i]) and prior_ret[i] != 0 and np.isfinite(fwd_ret[i]):
            reversal[i] = float(np.sign(fwd_ret[i]) != np.sign(prior_ret[i]))

    return {
        "convergence": conv,
        "squaring_score": score,
        "conv_sqrt": conv_sqrt,
        "conv_price": conv_price,
        "conv_range": conv_range,
        "prior_ret": prior_ret,
        "fwd_ret": fwd_ret,
        "reversal": reversal,
    }


def _threshold_metrics(
    conv: np.ndarray,
    reversal: np.ndarray,
    fwd_ret: np.ndarray,
    prior_ret: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    eval_mask = (prior_ret != 0) & np.isfinite(prior_ret) & np.isfinite(reversal) & np.isfinite(fwd_ret)
    sig = eval_mask & (conv >= threshold)
    base = eval_mask & ~sig
    n_eval = int(eval_mask.sum())
    n_sig = int(sig.sum())
    hit = float(reversal[sig].mean()) if n_sig else None
    base_rate = float(reversal[base].mean()) if base.any() else None
    lift = hit / base_rate if hit is not None and base_rate and base_rate > 0 else None
    fwd_sig = fwd_ret[sig]
    fwd_base = fwd_ret[base]
    return {
        "threshold": threshold,
        "signal_rate": n_sig / n_eval if n_eval else 0.0,
        "n_signal_weeks": n_sig,
        "hit_rate": hit,
        "base_rate": base_rate,
        "lift": lift,
        "mean_fwd_signal": float(fwd_sig.mean()) if len(fwd_sig) else None,
        "mean_fwd_base": float(fwd_base.mean()) if len(fwd_base) else None,
    }


def _direction_accuracy(score: np.ndarray, fwd_ret: np.ndarray, min_score: float = 0.10) -> float | None:
    mask = np.isfinite(score) & (np.abs(score) > min_score) & np.isfinite(fwd_ret) & (fwd_ret != 0)
    if not mask.any():
        return None
    return float(np.mean(np.sign(fwd_ret[mask]) == np.sign(score[mask])))


def _convergence_lift_at_swings(
    conv: np.ndarray,
    weekly: pd.DataFrame,
    *,
    lookback_weeks: int,
    pivot_order_retro: int,
) -> tuple[float | None, float | None, float | None]:
    highs = weekly["high"].values.astype(float)
    lows = weekly["low"].values.astype(float)
    dates = weekly["week_start_date"].values
    ph_r, pl_r = detect_major_pivots_weekly(highs, lows, dates, order=pivot_order_retro)
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    swing_idxs = [
        date_to_idx[pd.Timestamp(d)]
        for d, _ in ph_r + pl_r
        if pd.Timestamp(d) in date_to_idx
    ]

    near_vals: list[float] = []
    for sidx in swing_idxs:
        if sidx < 3:
            continue
        for b in range(max(lookback_weeks, sidx - 3), sidx):
            if np.isfinite(conv[b]):
                near_vals.append(float(conv[b]))

    all_vals = conv[np.isfinite(conv)]
    non_near = []
    near_set = set()
    for sidx in swing_idxs:
        if sidx < 3:
            continue
        for b in range(max(lookback_weeks, sidx - 3), sidx):
            near_set.add(b)
    for b in range(len(conv)):
        if np.isfinite(conv[b]) and b not in near_set:
            non_near.append(float(conv[b]))

    mean_near = float(np.mean(near_vals)) if near_vals else None
    mean_non = float(np.mean(non_near)) if non_near else None
    lift = mean_near / mean_non if mean_near is not None and mean_non and mean_non > 0 else None
    return mean_near, mean_non, lift


def analyze_symbol(
    symbol: str,
    weekly: pd.DataFrame,
    *,
    lookback_weeks: int,
    horizon_weeks: int,
    pivot_order: int,
    pivot_order_retro: int,
    tolerance_days: int,
    thresholds: list[float],
) -> dict[str, Any] | None:
    weekly = weekly.sort_values("week_start_date").reset_index(drop=True)
    n = len(weekly)
    min_weeks = lookback_weeks + horizon_weeks + 10
    if n < min_weeks:
        print("WARNING: skipping %s — only %d weekly bars (need %d)" % (symbol, n, min_weeks))
        return None

    wf = walk_forward_squaring(
        weekly,
        lookback_weeks=lookback_weeks,
        horizon_weeks=horizon_weeks,
        pivot_order=pivot_order,
        tolerance_days=tolerance_days,
    )
    conv = wf["convergence"]
    score = wf["squaring_score"]
    rev = wf["reversal"]
    fwd = wf["fwd_ret"]
    prior = wf["prior_ret"]

    sens = [_threshold_metrics(conv, rev, fwd, prior, t) for t in thresholds]
    dir_acc = _direction_accuracy(score, fwd)

    _, _, conv_lift = _convergence_lift_at_swings(
        conv, weekly, lookback_weeks=lookback_weeks, pivot_order_retro=pivot_order_retro,
    )

    best = max(sens, key=lambda x: (x.get("lift") or 0))
    print(
        f"{symbol:<6} conv_lift={conv_lift:.3f} dir_acc={dir_acc:.3f} "
        f"best_lift={best.get('lift'):.3f}@t={best['threshold']}"
        if conv_lift is not None and dir_acc is not None and best.get("lift")
        else f"{symbol:<6} partial data",
    )

    return {
        "symbol": symbol,
        "direction_accuracy": dir_acc,
        "convergence_lift_at_swings": conv_lift,
        "mean_conv_sqrt": float(np.mean(wf["conv_sqrt"][np.isfinite(wf["conv_sqrt"])])) if np.isfinite(wf["conv_sqrt"]).any() else None,
        "mean_conv_price": float(np.mean(wf["conv_price"][np.isfinite(wf["conv_price"])])) if np.isfinite(wf["conv_price"]).any() else None,
        "mean_conv_range": float(np.mean(wf["conv_range"][np.isfinite(wf["conv_range"])])) if np.isfinite(wf["conv_range"]).any() else None,
        "sensitivity": sens,
    }


def aggregate_sensitivity(rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for t in thresholds:
        lifts = []
        signal_rates = []
        conv_lifts = []
        dir_accs = []
        for r in rows:
            for s in r["sensitivity"]:
                if s["threshold"] == t:
                    if s.get("lift") is not None:
                        lifts.append(s["lift"])
                    signal_rates.append(s["signal_rate"])
            if r.get("convergence_lift_at_swings") is not None:
                conv_lifts.append(r["convergence_lift_at_swings"])
            if r.get("direction_accuracy") is not None:
                dir_accs.append(r["direction_accuracy"])

        mean_lift = float(np.mean(lifts)) if lifts else None
        mean_signal = float(np.mean(signal_rates)) if signal_rates else None
        mean_conv_lift = float(np.mean(conv_lifts)) if conv_lifts else None
        mean_dir = float(np.mean(dir_accs)) if dir_accs else None

        pass_gate = (
            mean_conv_lift is not None and mean_conv_lift > 1.20
            and mean_lift is not None and mean_lift > 1.15
            and mean_signal is not None and mean_signal >= 0.05
            and mean_dir is not None and mean_dir > 0.55
        )
        table.append({
            "threshold": t,
            "mean_lift": mean_lift,
            "mean_signal_rate": mean_signal,
            "mean_convergence_lift_at_swings": mean_conv_lift,
            "mean_direction_accuracy": mean_dir,
            "passes_go_criteria": pass_gate,
        })
    return table


def aggregate_verdict(sensitivity_table: list[dict[str, Any]]) -> str:
    if any(r["passes_go_criteria"] for r in sensitivity_table):
        return "SQUARING SIGNAL DETECTED"
    return "NO SQUARING SIGNAL"


def print_next_steps(verdict: str) -> None:
    print("\n" + "=" * 72)
    print("NEXT STEPS")
    print("=" * 72)
    if verdict == "SQUARING SIGNAL DETECTED":
        print(
            "Proceed to Experiment 3: weekly IC study with new_cycles_only variant.\n"
            "    python scripts/gann_ic_study.py --timeframe weekly --horizon 3"
        )
    else:
        print(
            "No price-derived squaring signal on this universe.\n"
            "If Experiment 1 also failed → permanently retire cycles component.\n"
            "If Experiment 1 passed → anniversary-only natural cycles may still have value."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gann price-derived squaring event study")
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--start", default="2020-01-02")
    parser.add_argument("--end", default="2026-07-20")
    parser.add_argument("--pivot-order", type=int, default=3)
    parser.add_argument("--pivot-order-retro", type=int, default=5)
    parser.add_argument("--lookback-weeks", type=int, default=26)
    parser.add_argument("--horizon-weeks", type=int, default=3)
    parser.add_argument("--tolerance-days", type=int, default=5)
    parser.add_argument("--threshold-sweep", action="store_true", help="Sweep 0.05–0.20")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    thresholds = [0.05, 0.10, 0.15, 0.20] if args.threshold_sweep else [args.threshold]
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    daily = load_prices(args.cache_dir, args.start, args.end)
    weekly_all = aggregate_to_weekly(daily)

    print("=" * 72)
    print("GANN SQUARING EVENT STUDY — per symbol (weekly bars)")
    print("=" * 72)

    by_symbol: list[dict[str, Any]] = []
    for symbol in symbols:
        sym_weekly = weekly_all[weekly_all["symbol"] == symbol]
        if sym_weekly.empty:
            print("WARNING: skipping %s — no data" % symbol)
            continue
        row = analyze_symbol(
            symbol,
            sym_weekly,
            lookback_weeks=args.lookback_weeks,
            horizon_weeks=args.horizon_weeks,
            pivot_order=args.pivot_order,
            pivot_order_retro=args.pivot_order_retro,
            tolerance_days=args.tolerance_days,
            thresholds=thresholds,
        )
        if row:
            by_symbol.append(row)

    if not by_symbol:
        print("ERROR: no symbols analyzed", file=sys.stderr)
        sys.exit(1)

    sensitivity_table = aggregate_sensitivity(by_symbol, thresholds)
    verdict = aggregate_verdict(sensitivity_table)

    print("\n" + "=" * 72)
    print("SENSITIVITY TABLE")
    print("=" * 72)
    for row in sensitivity_table:
        print(
            f"  t={row['threshold']:.2f} lift={row['mean_lift']:.3f} "
            f"conv_lift={row['mean_convergence_lift_at_swings']:.3f} "
            f"sig_rate={row['mean_signal_rate']:.3f} dir_acc={row['mean_direction_accuracy']:.3f} "
            f"pass={row['passes_go_criteria']}"
        )

    # subtype breakdown
    print("\nSub-type mean convergence:")
    for key, label in [
        ("mean_conv_sqrt", "sqrt_proj"),
        ("mean_conv_price", "price_proj"),
        ("mean_conv_range", "range_proj"),
    ]:
        vals = [r[key] for r in by_symbol if r.get(key) is not None]
        if vals:
            print(f"  {label}: {float(np.mean(vals)):.4f}")

    print(f"\nVERDICT: {verdict}")

    out_dir = Path(args.output or f"runs/gann_squaring_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {**vars(args), "timestamp": datetime.now().isoformat(), "symbols": symbols, "thresholds": thresholds},
        "aggregate": {"verdict": verdict, "sensitivity_table": sensitivity_table},
        "by_symbol": by_symbol,
        "sensitivity_table": sensitivity_table,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results: {out_path}")
    print_next_steps(verdict)


if __name__ == "__main__":
    main()
