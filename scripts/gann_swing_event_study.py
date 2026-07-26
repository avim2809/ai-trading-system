#!/usr/bin/env python3
"""Gann time-cycle convergence swing event study.

Tests whether cycle convergence predicts *when* price swings occur (timing
hypothesis) rather than cross-sectional direction.

Usage::

    python scripts/gann_swing_event_study.py
    python scripts/gann_swing_event_study.py --convergence-threshold 0.06
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

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META",
    "TSLA", "AVGO", "AMD", "CRM", "NFLX", "ADBE",
    "JPM", "GS", "BAC", "V", "MA",
    "JNJ", "UNH", "LLY",
    "XOM", "CVX",
    "SPY", "QQQ", "IWM",
]

GANN_CALENDAR_CYCLES = [30, 60, 90, 120, 144, 180, 270, 360]


def detect_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    order: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Pivot high/low lists as (bar_index, price) within the input arrays."""
    n = len(highs)
    pivot_highs: list[tuple[int, float]] = []
    pivot_lows: list[tuple[int, float]] = []
    for i in range(order, n - order):
        if highs[i] == max(highs[i - order : i + order + 1]):
            pivot_highs.append((i, float(highs[i])))
        if lows[i] == min(lows[i - order : i + order + 1]):
            pivot_lows.append((i, float(lows[i])))
    return pivot_highs, pivot_lows


def cycle_convergence(
    current_idx: int,
    pivot_highs: list[tuple[int, float]],
    pivot_lows: list[tuple[int, float]],
    tolerance: int,
) -> float:
    """Raw convergence ratio (hits / total_checks), no directional score."""
    gann_cycles_bars = [int(round(c * 252 / 365)) for c in GANN_CALENDAR_CYCLES]
    all_pivots = [(idx, "low") for idx, _ in pivot_lows] + [
        (idx, "high") for idx, _ in pivot_highs
    ]
    hits = 0
    total_checks = 0
    for p_idx, _ in all_pivots:
        for cycle in gann_cycles_bars:
            projected = p_idx + cycle
            total_checks += 1
            if abs(projected - current_idx) <= tolerance:
                hits += 1
    if total_checks == 0:
        return 0.0
    return hits / total_checks


def load_prices(cache_dir: str, start: str, end: str) -> pd.DataFrame:
    from firm.data.cache import ParquetCache

    cache = ParquetCache(cache_dir)
    prices_df = cache.get("combined/prices")
    if prices_df is None or prices_df.empty:
        print(
            "ERROR: No cached price data at combined/prices.\n"
            "Run: python scripts/fetch_data.py --symbols AAPL,MSFT,... "
            f"--start {start} --end {end}",
            file=sys.stderr,
        )
        sys.exit(1)

    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    if "adj_close" not in prices_df.columns and "close" in prices_df.columns:
        prices_df["adj_close"] = prices_df["close"]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    prices_df = prices_df[
        (prices_df["date"] >= start_ts) & (prices_df["date"] <= end_ts)
    ]
    return prices_df


def walk_forward_series(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    pivot_order_live: int,
    tolerance: int,
) -> dict[str, np.ndarray]:
    """Walk-forward convergence and forward-return arrays indexed by bar."""
    n = len(closes)
    conv = np.full(n, np.nan)
    prior_dir = np.zeros(n)
    fwd_ret = np.full(n, np.nan)
    reversal = np.full(n, np.nan)

    for i in range(lookback, n - horizon):
        w_highs = highs[i - lookback + 1 : i + 1]
        w_lows = lows[i - lookback + 1 : i + 1]
        w_closes = closes[i - lookback + 1 : i + 1]
        if len(w_closes) < lookback:
            continue

        ph, pl = detect_pivots(w_highs, w_lows, pivot_order_live)
        local_idx = lookback - 1
        conv[i] = cycle_convergence(local_idx, ph, pl, tolerance)

        if i >= 5 and closes[i - 5] > 0:
            prior_dir[i] = float(np.sign(closes[i] / closes[i - 5] - 1.0))
        else:
            prior_dir[i] = 0.0

        if closes[i] > 0:
            fwd_ret[i] = closes[i + horizon] / closes[i] - 1.0

        if prior_dir[i] != 0 and np.isfinite(fwd_ret[i]):
            reversal[i] = float(np.sign(fwd_ret[i]) != prior_dir[i])

    return {
        "convergence": conv,
        "prior_dir": prior_dir,
        "fwd_ret": fwd_ret,
        "reversal": reversal,
    }


def retro_swing_indices(highs: np.ndarray, lows: np.ndarray, order: int) -> list[int]:
    ph, pl = detect_pivots(highs, lows, order)
    return sorted({idx for idx, _ in ph} | {idx for idx, _ in pl})


def near_any_swing(bar: int, swings: list[int], radius: int) -> bool:
    return any(abs(bar - s) <= radius for s in swings)


def analyze_symbol(
    symbol: str,
    sym_df: pd.DataFrame,
    *,
    lookback: int,
    horizon: int,
    pivot_order_live: int,
    pivot_order_retro: int,
    tolerance: int,
    convergence_threshold: float,
) -> dict[str, Any] | None:
    sym_df = sym_df.sort_values("date").reset_index(drop=True)
    n = len(sym_df)
    min_bars = lookback + horizon + 20
    if n < min_bars:
        print("WARNING: skipping %s — only %d bars (need %d)" % (symbol, n, min_bars))
        return None

    closes = sym_df["adj_close"].values.astype(float)
    highs = sym_df["high"].values.astype(float) if "high" in sym_df.columns else closes
    lows = sym_df["low"].values.astype(float) if "low" in sym_df.columns else closes

    wf = walk_forward_series(
        closes, highs, lows,
        lookback=lookback,
        horizon=horizon,
        pivot_order_live=pivot_order_live,
        tolerance=tolerance,
    )
    conv = wf["convergence"]
    prior_dir = wf["prior_dir"]
    fwd_ret = wf["fwd_ret"]
    reversal = wf["reversal"]

    eval_mask = (
        np.isfinite(conv)
        & (prior_dir != 0)
        & np.isfinite(reversal)
        & np.isfinite(fwd_ret)
    )
    if not eval_mask.any():
        print("WARNING: skipping %s — no evaluable bars" % symbol)
        return None

    signal_mask = eval_mask & (conv >= convergence_threshold)
    base_mask = eval_mask & ~signal_mask

    n_bars = int(eval_mask.sum())
    n_signal_days = int(signal_mask.sum())

    hit_rate = float(reversal[signal_mask].mean()) if n_signal_days else None
    base_rate = float(reversal[base_mask].mean()) if base_mask.any() else None
    lift = (hit_rate / base_rate) if hit_rate is not None and base_rate and base_rate > 0 else None

    fwd_sig = fwd_ret[signal_mask]
    fwd_base = fwd_ret[base_mask]
    mean_fwd_sig = float(fwd_sig.mean()) if len(fwd_sig) else None
    mean_fwd_base = float(fwd_base.mean()) if len(fwd_base) else None

    welch_p = None
    if len(fwd_sig) >= 2 and len(fwd_base) >= 2:
        _, welch_p = stats.ttest_ind(fwd_sig, fwd_base, equal_var=False)
        welch_p = float(welch_p)

    # TEST 2 — backward detection
    swings = retro_swing_indices(highs, lows, pivot_order_retro)
    detected = 0
    checked = 0
    for sidx in swings:
        if sidx < 5:
            continue
        checked += 1
        pre_bars = range(max(lookback, sidx - 5), sidx)
        if any(
            np.isfinite(conv[b]) and conv[b] >= convergence_threshold
            for b in pre_bars
        ):
            detected += 1

    detection_rate = detected / checked if checked else None
    valid_conv = conv[np.isfinite(conv)]
    false_alarm_rate = (
        float((valid_conv >= convergence_threshold).mean())
        if len(valid_conv) else None
    )

    near_mask = np.array([
        near_any_swing(i, swings, pivot_order_retro) if np.isfinite(conv[i]) else False
        for i in range(n)
    ])
    conv_near = conv[near_mask & np.isfinite(conv)]
    conv_non = conv[~near_mask & np.isfinite(conv)]
    mean_conv_near = float(conv_near.mean()) if len(conv_near) else None
    mean_conv_non = float(conv_non.mean()) if len(conv_non) else None

    signal_rate = n_signal_days / n_bars if n_bars else 0.0

    print(
        f"{symbol:<6} n={n_bars:4d} sig={n_signal_days:4d} "
        f"sig_rate={signal_rate:.3f} hit={hit_rate:.3f} base={base_rate:.3f} "
        f"lift={lift:.3f} detect={detection_rate:.3f} "
        f"conv_near={mean_conv_near:.4f} conv_non={mean_conv_non:.4f}"
        if hit_rate is not None and base_rate is not None and lift is not None
        else f"{symbol:<6} insufficient signal/base data",
    )

    return {
        "symbol": symbol,
        "n_bars": n_bars,
        "n_signal_days": n_signal_days,
        "signal_rate": signal_rate,
        "hit_rate": hit_rate,
        "base_rate": base_rate,
        "lift": lift,
        "mean_fwd_sig": mean_fwd_sig,
        "mean_fwd_base": mean_fwd_base,
        "welch_p": welch_p,
        "detection_rate": detection_rate,
        "n_swings_checked": checked,
        "false_alarm_rate": false_alarm_rate,
        "mean_conv_near_swing": mean_conv_near,
        "mean_conv_non_swing": mean_conv_non,
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    p_vals = [r["welch_p"] for r in rows if r.get("welch_p") is not None]
    frac_p = float(np.mean([p < 0.05 for p in p_vals])) if p_vals else None

    mean_conv_near = _mean("mean_conv_near_swing")
    mean_conv_non = _mean("mean_conv_non_swing")
    conv_lift = (
        mean_conv_near / mean_conv_non
        if mean_conv_near is not None and mean_conv_non and mean_conv_non > 0
        else None
    )

    mean_lift = _mean("lift")
    mean_detection = _mean("detection_rate")
    mean_false_alarm = _mean("false_alarm_rate")

    timing_detected = (
        mean_lift is not None and mean_lift > 1.15
        and mean_detection is not None and mean_false_alarm is not None
        and mean_false_alarm > 0
        and mean_detection > 1.5 * mean_false_alarm
        and conv_lift is not None and conv_lift > 1.1
    )

    return {
        "n_symbols": len(rows),
        "test1": {
            "mean_signal_rate": _mean("signal_rate"),
            "mean_hit_rate": _mean("hit_rate"),
            "mean_base_rate": _mean("base_rate"),
            "mean_lift": mean_lift,
            "mean_fwd_sig": _mean("mean_fwd_sig"),
            "mean_fwd_base": _mean("mean_fwd_base"),
            "frac_symbols_p_lt_05": frac_p,
        },
        "test2": {
            "mean_detection_rate": mean_detection,
            "mean_false_alarm_rate": mean_false_alarm,
            "mean_conv_near_swing": mean_conv_near,
            "mean_conv_non_swing": mean_conv_non,
            "convergence_lift_at_swings": conv_lift,
        },
        "verdict": "TIMING VALUE DETECTED" if timing_detected else "NO TIMING VALUE",
    }


def print_next_steps(verdict: str) -> None:
    print("\n" + "=" * 72)
    print("NEXT STEPS")
    print("=" * 72)
    if verdict == "TIMING VALUE DETECTED":
        print(
            "Redesign cycles as a position-sizing gate, not a cross-sectional score.\n"
            "Run sensitivity sweep across thresholds."
        )
    else:
        print(
            "Remove cycles from composite; reallocate 15% weight to angles or\n"
            "sign-corrected retracement."
        )
    print("\nSensitivity sweep:")
    print("    for t in 0.02 0.04 0.06 0.08 0.10; do")
    print("      python scripts/gann_swing_event_study.py --convergence-threshold $t")
    print("    done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gann swing event study")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_UNIVERSE),
        help="Comma-separated symbols",
    )
    parser.add_argument("--start", default="2020-01-02")
    parser.add_argument("--end", default="2026-07-20")
    parser.add_argument("--convergence-threshold", type=float, default=0.04)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--pivot-order-live", type=int, default=5)
    parser.add_argument("--pivot-order-retro", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    prices_df = load_prices(args.cache_dir, args.start, args.end)

    print("=" * 72)
    print("GANN SWING EVENT STUDY — per symbol")
    print("=" * 72)

    by_symbol: list[dict[str, Any]] = []
    for symbol in symbols:
        sym_df = prices_df[prices_df["symbol"] == symbol]
        if sym_df.empty:
            print("WARNING: skipping %s — no price data" % symbol)
            continue
        row = analyze_symbol(
            symbol,
            sym_df,
            lookback=args.lookback,
            horizon=args.horizon,
            pivot_order_live=args.pivot_order_live,
            pivot_order_retro=args.pivot_order_retro,
            tolerance=args.tolerance,
            convergence_threshold=args.convergence_threshold,
        )
        if row is not None:
            by_symbol.append(row)

    if not by_symbol:
        print("ERROR: no symbols analyzed", file=sys.stderr)
        sys.exit(1)

    agg = aggregate_results(by_symbol)
    t1, t2 = agg["test1"], agg["test2"]

    print("\n" + "=" * 72)
    print("TEST 1 — FORWARD HIT RATE")
    print("=" * 72)
    print(f"  mean_signal_rate     : {t1['mean_signal_rate']:.4f}" if t1["mean_signal_rate"] else "  mean_signal_rate     : n/a")
    print(f"  mean_hit_rate        : {t1['mean_hit_rate']:.4f}" if t1["mean_hit_rate"] else "  mean_hit_rate        : n/a")
    print(f"  mean_base_rate       : {t1['mean_base_rate']:.4f}" if t1["mean_base_rate"] else "  mean_base_rate       : n/a")
    print(f"  mean_lift            : {t1['mean_lift']:.4f}" if t1["mean_lift"] else "  mean_lift            : n/a")
    print(f"  mean_fwd_sig         : {t1['mean_fwd_sig']:.6f}" if t1["mean_fwd_sig"] is not None else "  mean_fwd_sig         : n/a")
    print(f"  mean_fwd_base        : {t1['mean_fwd_base']:.6f}" if t1["mean_fwd_base"] is not None else "  mean_fwd_base        : n/a")
    print(f"  frac_symbols_p_lt_05: {t1['frac_symbols_p_lt_05']:.2%}" if t1["frac_symbols_p_lt_05"] is not None else "  frac_symbols_p_lt_05: n/a")

    print("\n" + "=" * 72)
    print("TEST 2 — BACKWARD DETECTION")
    print("=" * 72)
    print(f"  mean_detection_rate       : {t2['mean_detection_rate']:.4f}" if t2["mean_detection_rate"] else "  mean_detection_rate       : n/a")
    print(f"  mean_false_alarm_rate     : {t2['mean_false_alarm_rate']:.4f}" if t2["mean_false_alarm_rate"] else "  mean_false_alarm_rate     : n/a")
    print(f"  mean_conv_near_swing      : {t2['mean_conv_near_swing']:.6f}" if t2["mean_conv_near_swing"] else "  mean_conv_near_swing      : n/a")
    print(f"  mean_conv_non_swing       : {t2['mean_conv_non_swing']:.6f}" if t2["mean_conv_non_swing"] else "  mean_conv_non_swing       : n/a")
    print(f"  convergence_lift_at_swings: {t2['convergence_lift_at_swings']:.4f}" if t2["convergence_lift_at_swings"] else "  convergence_lift_at_swings: n/a")

    print(f"\nVERDICT: {agg['verdict']}")

    out_dir = Path(
        args.output or f"runs/gann_swing_event_study_{datetime.now():%Y%m%d_%H%M%S}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "start": args.start,
            "end": args.end,
            "convergence_threshold": args.convergence_threshold,
            "horizon": args.horizon,
            "pivot_order_live": args.pivot_order_live,
            "pivot_order_retro": args.pivot_order_retro,
            "lookback": args.lookback,
            "tolerance": args.tolerance,
            "cache_dir": args.cache_dir,
        },
        "aggregate": agg,
        "by_symbol": by_symbol,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results: {out_path}")

    print_next_steps(agg["verdict"])


if __name__ == "__main__":
    main()
