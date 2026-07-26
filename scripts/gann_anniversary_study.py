#!/usr/bin/env python3
"""Experiment 1: Gann anniversary cycle event study (weekly bars).

Tests whether 365/730/1095 calendar-day anniversaries of major pivots
predict elevated reversal probability.

Usage::

    python scripts/gann_anniversary_study.py
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
    aggregate_to_weekly,
    anniversary_window_active,
    detect_major_pivots_weekly,
    load_prices,
    pivots_to_records,
)


def _walk_forward_anniversary(
    weekly: pd.DataFrame,
    *,
    lookback_weeks: int,
    horizon_weeks: int,
    pivot_order: int,
    anniversary_days: list[int],
    anniversary_tolerance: int,
) -> dict[str, np.ndarray]:
    dates = weekly["week_start_date"].values
    closes = weekly["adj_close"].values.astype(float)
    highs = weekly["high"].values.astype(float)
    lows = weekly["low"].values.astype(float)
    n = len(weekly)

    in_window = np.zeros(n, dtype=bool)
    prior_ret = np.full(n, np.nan)
    fwd_ret = np.full(n, np.nan)
    reversal = np.full(n, np.nan)

    for i in range(lookback_weeks, n - horizon_weeks):
        asof = pd.Timestamp(dates[i])
        w_highs = highs[: i + 1]
        w_lows = lows[: i + 1]
        w_dates = dates[: i + 1]
        ph, pl = detect_major_pivots_weekly(w_highs, w_lows, w_dates, order=pivot_order)
        pivots = pivots_to_records(ph, pl)

        active = False
        for pdate, _, _ in pivots:
            if anniversary_window_active(asof, pdate, anniversary_days, anniversary_tolerance):
                active = True
                break
        in_window[i] = active

        if i >= horizon_weeks and closes[i - horizon_weeks] > 0:
            prior_ret[i] = closes[i] / closes[i - horizon_weeks] - 1.0
        if closes[i] > 0:
            fwd_ret[i] = closes[i + horizon_weeks] / closes[i] - 1.0

        if np.isfinite(prior_ret[i]) and prior_ret[i] != 0 and np.isfinite(fwd_ret[i]):
            reversal[i] = float(np.sign(fwd_ret[i]) != np.sign(prior_ret[i]))

    return {
        "in_window": in_window,
        "prior_ret": prior_ret,
        "fwd_ret": fwd_ret,
        "reversal": reversal,
    }


def analyze_symbol(
    symbol: str,
    weekly: pd.DataFrame,
    *,
    lookback_weeks: int,
    horizon_weeks: int,
    pivot_order: int,
    pivot_order_retro: int,
    anniversary_days: list[int],
    anniversary_tolerance: int,
) -> dict[str, Any] | None:
    weekly = weekly.sort_values("week_start_date").reset_index(drop=True)
    n = len(weekly)
    min_weeks = lookback_weeks + horizon_weeks + 10
    if n < min_weeks:
        print("WARNING: skipping %s — only %d weekly bars (need %d)" % (symbol, n, min_weeks))
        return None

    wf = _walk_forward_anniversary(
        weekly,
        lookback_weeks=lookback_weeks,
        horizon_weeks=horizon_weeks,
        pivot_order=pivot_order,
        anniversary_days=anniversary_days,
        anniversary_tolerance=anniversary_tolerance,
    )
    in_w = wf["in_window"]
    prior = wf["prior_ret"]
    fwd = wf["fwd_ret"]
    rev = wf["reversal"]

    eval_mask = (prior != 0) & np.isfinite(prior) & np.isfinite(rev) & np.isfinite(fwd)
    if not eval_mask.any():
        print("WARNING: skipping %s — no evaluable weeks" % symbol)
        return None

    win_mask = eval_mask & in_w
    base_mask = eval_mask & ~in_w

    hit_rate = float(rev[win_mask].mean()) if win_mask.any() else None
    base_rate = float(rev[base_mask].mean()) if base_mask.any() else None
    lift = hit_rate / base_rate if hit_rate is not None and base_rate and base_rate > 0 else None

    fwd_win = fwd[win_mask]
    fwd_base = fwd[base_mask]
    mean_fwd_window = float(np.abs(fwd_win).mean()) if len(fwd_win) else None
    mean_fwd_base = float(np.abs(fwd_base).mean()) if len(fwd_base) else None
    mean_fwd_window_signed = float(fwd_win.mean()) if len(fwd_win) else None
    mean_fwd_base_signed = float(fwd_base.mean()) if len(fwd_base) else None

    welch_p = None
    if len(fwd_win) >= 2 and len(fwd_base) >= 2:
        _, welch_p = stats.ttest_ind(fwd_win, fwd_base, equal_var=False)
        welch_p = float(welch_p)

    # Retroactive swings (full history, order=5)
    highs = weekly["high"].values.astype(float)
    lows = weekly["low"].values.astype(float)
    dates = weekly["week_start_date"].values
    ph_r, pl_r = detect_major_pivots_weekly(highs, lows, dates, order=pivot_order_retro)
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    swing_idxs = sorted({
        date_to_idx[pd.Timestamp(d)]
        for d, _ in ph_r + pl_r
        if pd.Timestamp(d) in date_to_idx
    })

    detected = 0
    checked = 0
    for sidx in swing_idxs:
        if sidx < 3:
            continue
        checked += 1
        pre = range(max(lookback_weeks, sidx - 3), sidx)
        if any(in_w[b] for b in pre):
            detected += 1

    detection_rate = detected / checked if checked else None
    false_alarm_rate = float(in_w[eval_mask].mean()) if eval_mask.any() else None

    print(
        f"{symbol:<6} weeks={int(eval_mask.sum()):4d} ann={int(win_mask.sum()):4d} "
        f"lift={lift:.3f} detect={detection_rate:.3f} "
        f"|fwd|_win={mean_fwd_window:.4f} |fwd|_base={mean_fwd_base:.4f}"
        if lift is not None else f"{symbol:<6} insufficient data",
    )

    return {
        "symbol": symbol,
        "n_weeks": int(eval_mask.sum()),
        "n_anniversary_weeks": int(win_mask.sum()),
        "hit_rate": hit_rate,
        "base_rate": base_rate,
        "lift": lift,
        "mean_abs_fwd_window": mean_fwd_window,
        "mean_abs_fwd_base": mean_fwd_base,
        "mean_fwd_window": mean_fwd_window_signed,
        "mean_fwd_base": mean_fwd_base_signed,
        "welch_p": welch_p,
        "detection_rate": detection_rate,
        "n_swings_checked": checked,
        "false_alarm_rate": false_alarm_rate,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    mean_lift = _mean("lift")
    mean_detection = _mean("detection_rate")
    mean_false_alarm = _mean("false_alarm_rate")
    mean_abs_fwd_win = _mean("mean_abs_fwd_window")
    mean_abs_fwd_base = _mean("mean_abs_fwd_base")
    symbols_lift_gt_1 = sum(1 for r in rows if (r.get("lift") or 0) > 1.0)

    abs_fwd_ratio = (
        mean_abs_fwd_win / mean_abs_fwd_base
        if mean_abs_fwd_win is not None and mean_abs_fwd_base and mean_abs_fwd_base > 0
        else None
    )

    detected = (
        mean_lift is not None and mean_lift > 1.15
        and mean_detection is not None and mean_false_alarm is not None
        and mean_false_alarm > 0
        and mean_detection > 1.3 * mean_false_alarm
        and symbols_lift_gt_1 >= 15
        and abs_fwd_ratio is not None and abs_fwd_ratio > 1.10
    )

    return {
        "n_symbols": len(rows),
        "mean_hit_rate": _mean("hit_rate"),
        "mean_base_rate": _mean("base_rate"),
        "mean_lift": mean_lift,
        "mean_abs_fwd_window": mean_abs_fwd_win,
        "mean_abs_fwd_base": mean_abs_fwd_base,
        "abs_fwd_ratio": abs_fwd_ratio,
        "mean_detection_rate": mean_detection,
        "mean_false_alarm_rate": mean_false_alarm,
        "symbols_with_lift_gt_1": symbols_lift_gt_1,
        "frac_symbols_p_lt_05": float(np.mean([
            r["welch_p"] < 0.05 for r in rows if r.get("welch_p") is not None
        ])) if any(r.get("welch_p") is not None for r in rows) else None,
        "verdict": "ANNIVERSARY EFFECT DETECTED" if detected else "NO ANNIVERSARY EFFECT",
    }


def print_next_steps(verdict: str) -> None:
    print("\n" + "=" * 72)
    print("NEXT STEPS")
    print("=" * 72)
    if verdict == "ANNIVERSARY EFFECT DETECTED":
        print(
            "Proceed to Experiment 3 (weekly IC) with anniversary_only variant,\n"
            "or run Experiment 2 (price-derived squaring) in parallel."
        )
    else:
        print(
            "Natural calendar anniversaries show no effect on this universe.\n"
            "Run Experiment 2 (scripts/gann_squaring_event_study.py) — if that\n"
            "also fails, permanently retire the cycles component."
        )
    print("\n    python scripts/gann_squaring_event_study.py --threshold-sweep")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gann anniversary event study")
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--start", default="2020-01-02")
    parser.add_argument("--end", default="2026-07-20")
    parser.add_argument("--anniversary-days", default="365,730,1095")
    parser.add_argument("--anniversary-tolerance", type=int, default=7)
    parser.add_argument("--pivot-order", type=int, default=3)
    parser.add_argument("--pivot-order-retro", type=int, default=5)
    parser.add_argument("--lookback-weeks", type=int, default=26)
    parser.add_argument("--horizon-weeks", type=int, default=3)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    anniversary_days = [int(x) for x in args.anniversary_days.split(",") if x.strip()]
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    daily = load_prices(args.cache_dir, args.start, args.end)
    weekly_all = aggregate_to_weekly(daily)

    print("=" * 72)
    print("GANN ANNIVERSARY STUDY — per symbol (weekly bars)")
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
            anniversary_days=anniversary_days,
            anniversary_tolerance=args.anniversary_tolerance,
        )
        if row:
            by_symbol.append(row)

    if not by_symbol:
        print("ERROR: no symbols analyzed", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(by_symbol)
    print("\n" + "=" * 72)
    print("AGGREGATE")
    print("=" * 72)
    for k, v in agg.items():
        if k == "verdict":
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nVERDICT: {agg['verdict']}")

    out_dir = Path(args.output or f"runs/gann_anniversary_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {**vars(args), "timestamp": datetime.now().isoformat(), "symbols": symbols},
        "aggregate": agg,
        "by_symbol": by_symbol,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results: {out_path}")
    print_next_steps(agg["verdict"])


if __name__ == "__main__":
    main()
