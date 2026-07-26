#!/usr/bin/env python3
"""Gann correct cycles study — calendar anniversaries and sqrt(P) squaring.

Implements calendar-day Gann cycle mechanics (no trading-bar conversion) and
runs Experiment 1 (natural calendar cycles) and Experiment 2 (price-derived
sqrt squaring with per-symbol scale calibration).

Usage::

    python scripts/gann_correct_cycles_study.py
    python scripts/gann_correct_cycles_study.py --universe macro
    python scripts/gann_correct_cycles_study.py --symbols USO,SPY,GLD,EURUSD,BTCUSD
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

log = logging.getLogger(__name__)

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META",
    "TSLA", "AVGO", "AMD", "CRM", "NFLX", "ADBE",
    "JPM", "GS", "BAC", "V", "MA",
    "JNJ", "UNH", "LLY",
    "XOM", "CVX",
    "SPY", "QQQ", "IWM",
]

# Oil (USO), S&P 500 (SPY), gold (GLD), EUR/USD, bitcoin — Tiingo tickers.
MACRO_UNIVERSE = ["USO", "SPY", "GLD", "EURUSD", "BTCUSD"]

MACRO_LABELS = {
    "USO": "oil (USO)",
    "SPY": "S&P 500 (SPY)",
    "GLD": "gold (GLD)",
    "EURUSD": "EUR/USD",
    "BTCUSD": "bitcoin (BTCUSD)",
}

NATURAL_CALENDAR_CYCLES_DAYS = [
    7, 14, 21, 30, 45, 49, 60, 90,
    135, 150, 180, 210, 225, 270, 315, 330, 360, 365,
    91, 182, 273, 315, 364, 546,
]

SQRT_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0]
SQRT_SCALES = [1.0, 7.0, 30.0]
DIRECT_SCALES = [1, 5, 10, 50, 100]

WINDOW_TOLERANCE_DAYS = 5
PIVOT_DIRECTION_DAYS = 10
FORWARD_RETURN_DAYS = 10
PIVOT_ORDER = 15
MIN_SWING_PCT = 0.05
N_PERMUTATIONS = 500

TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2022, 12, 31)
HOLDOUT_START = date(2023, 1, 1)
HOLDOUT_END = date(2026, 6, 30)

PivotType = Literal["high", "low"]


def load_prices(cache_dir: str = "data/cache", symbols: list[str] | None = None) -> pd.DataFrame:
    """Load combined price panel; adapt flat or MultiIndex schema."""
    from firm.data.cache import ParquetCache

    cache = ParquetCache(cache_dir)
    prices = cache.get("combined/prices")
    if prices is None or prices.empty:
        log.error("No cached price data at combined/prices")
        sys.exit(1)

    if isinstance(prices.index, pd.MultiIndex):
        prices = prices.reset_index()

    if "symbol" not in prices.columns:
        if "ticker" in prices.columns:
            prices = prices.rename(columns={"ticker": "symbol"})
        else:
            log.error("Cannot detect symbol column in price data")
            sys.exit(1)

    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    if "adj_close" not in prices.columns:
        prices["adj_close"] = prices["close"]
    for col in ("open", "high", "low", "close"):
        if col not in prices.columns:
            prices[col] = prices["adj_close"]
    if "volume" not in prices.columns:
        prices["volume"] = 0.0

    if symbols:
        want = {s.upper() for s in symbols}
        prices = prices[prices["symbol"].str.upper().isin(want)].copy()
    return prices


def detect_pivots(
    df: pd.DataFrame,
    order: int = PIVOT_ORDER,
    min_swing_pct: float = MIN_SWING_PCT,
) -> tuple[list[date], list[float], list[PivotType]]:
    """Sparse major pivots via argrelextrema + minimum swing filter."""
    closes = df["adj_close"].values.astype(float)
    dates = list(df["date"].values)
    n = len(closes)
    if n < 2 * order + 1:
        return [], [], []

    hi_idx = argrelextrema(closes, np.greater_equal, order=order)[0]
    lo_idx = argrelextrema(closes, np.less_equal, order=order)[0]

    raw: list[tuple[int, float, PivotType]] = []
    for i in hi_idx:
        raw.append((int(i), float(closes[i]), "high"))
    for i in lo_idx:
        raw.append((int(i), float(closes[i]), "low"))
    raw.sort(key=lambda x: x[0])

    deduped: list[tuple[int, float, PivotType]] = []
    for item in raw:
        if deduped and deduped[-1][0] == item[0]:
            if abs(item[1] - closes[max(item[0] - 1, 0)]) > abs(
                deduped[-1][1] - closes[max(deduped[-1][0] - 1, 0)],
            ):
                deduped[-1] = item
        else:
            deduped.append(item)

    filtered: list[tuple[int, float, PivotType]] = []
    for idx, (i, price, ptype) in enumerate(deduped):
        prev_p = deduped[idx - 1][1] if idx > 0 else price
        next_p = deduped[idx + 1][1] if idx + 1 < len(deduped) else price
        swing_prev = abs(price - prev_p) / prev_p if prev_p > 0 else 0.0
        swing_next = abs(price - next_p) / next_p if next_p > 0 else 0.0
        if swing_prev >= min_swing_pct and swing_next >= min_swing_pct:
            filtered.append((i, price, ptype))

    pivot_dates = [dates[i] for i, _, _ in filtered]
    pivot_prices = [p for _, p, _ in filtered]
    pivot_types: list[PivotType] = [t for _, _, t in filtered]
    return pivot_dates, pivot_prices, pivot_types


def compute_price_derived_cycles(pivot_price: float) -> list[int]:
    sqrt_p = float(np.sqrt(pivot_price))
    cycles: list[float] = []
    for mult in SQRT_MULTIPLIERS:
        for scale in SQRT_SCALES:
            cycles.append(sqrt_p * mult * scale)
    return [int(round(c)) for c in cycles if 7 <= c <= 1095]


def compute_direct_cycles(pivot_price: float, scale: float) -> list[int]:
    if scale <= 0:
        return []
    days = pivot_price / scale
    return [int(round(days))] if 7 <= days <= 1095 else []


def _date_to_py(d: Any) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    return pd.Timestamp(d).date()


def build_cycle_lists(
    pivot_prices: list[float],
    *,
    mode: Literal["natural", "sqrt"],
    direct_scale: float | None = None,
) -> list[list[int]]:
    out: list[list[int]] = []
    for p in pivot_prices:
        if mode == "natural":
            out.append(list(NATURAL_CALENDAR_CYCLES_DAYS))
        else:
            cycles = compute_price_derived_cycles(p)
            if direct_scale is not None:
                cycles = cycles + compute_direct_cycles(p, direct_scale)
            out.append(sorted(set(cycles)))
    return out


def projection_target_ords(
    pivot_dates: list[date],
    cycle_lists: list[list[int]],
) -> np.ndarray:
    targets: list[int] = []
    for pdate, cycles in zip(pivot_dates, cycle_lists):
        base = pdate.toordinal()
        for c in cycles:
            targets.append(base + int(c))
    return np.array(targets, dtype=np.int32)


def window_mask(
    date_ords: np.ndarray,
    target_ords: np.ndarray,
    tolerance_days: int = WINDOW_TOLERANCE_DAYS,
) -> np.ndarray:
    if target_ords.size == 0:
        return np.zeros(len(date_ords), dtype=bool)
    diff = np.abs(date_ords[:, None].astype(np.int32) - target_ords[None, :])
    return (diff <= tolerance_days).any(axis=1)


def nearest_pivot_direction_ords(
    date_ords: np.ndarray,
    pivot_date_ords: np.ndarray,
    pivot_types: list[PivotType],
    within_days: int = PIVOT_DIRECTION_DAYS,
) -> np.ndarray:
    """Per-bar direction: +1, -1, or 0 (ambiguous / none)."""
    n = len(date_ords)
    direction = np.zeros(n, dtype=np.int8)
    for p_ord, ptype in zip(pivot_date_ords, pivot_types):
        near = np.abs(date_ords - p_ord) <= within_days
        if ptype == "low":
            direction[near] += 1
        else:
            direction[near] -= 1
    out = np.zeros(n, dtype=np.int8)
    out[direction == 1] = 1
    out[direction == -1] = -1
    return out


def confirmation_mask(df: pd.DataFrame, direction: np.ndarray) -> np.ndarray:
    n = len(df)
    confirmed = np.zeros(n, dtype=bool)
    vol = df["volume"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["adj_close"].values.astype(float)
    for i in range(20, n):
        d = int(direction[i])
        if d == 0:
            continue
        vol_ma = float(np.mean(vol[i - 20 : i]))
        if vol_ma > 0 and vol[i] > 1.5 * vol_ma:
            confirmed[i] = True
            continue
        if d > 0 and close[i] > float(np.max(close[i - 5 : i])):
            confirmed[i] = True
            continue
        if d < 0 and close[i] < float(np.min(close[i - 5 : i])):
            confirmed[i] = True
            continue
        ranges = high[i - 5 : i] - low[i - 5 : i]
        bar_range = high[i] - low[i]
        if len(ranges) and float(np.mean(ranges)) > 0 and bar_range > 1.3 * float(np.mean(ranges)):
            confirmed[i] = True
    return confirmed


class SymbolStudy:
    """Precomputed per-symbol arrays for fast event metrics."""

    def __init__(self, symbol: str, df: pd.DataFrame) -> None:
        self.symbol = symbol
        self.df = df
        self.dates = [_date_to_py(d) for d in df["date"].values]
        self.date_ords = np.array([d.toordinal() for d in self.dates], dtype=np.int32)
        self.pivot_dates, self.pivot_prices, self.pivot_types = detect_pivots(df)
        self.pivot_ords = np.array([d.toordinal() for d in self.pivot_dates], dtype=np.int32)

        closes = df["adj_close"].values.astype(float)
        self.direction = nearest_pivot_direction_ords(
            self.date_ords, self.pivot_ords, self.pivot_types,
        )
        self.fwd_hit = np.full(len(df), np.nan)
        for i in range(len(df) - FORWARD_RETURN_DAYS):
            d = int(self.direction[i])
            if d == 0 or closes[i] <= 0:
                continue
            fwd = closes[i + FORWARD_RETURN_DAYS] / closes[i] - 1.0
            if fwd == 0:
                continue
            self.fwd_hit[i] = float(np.sign(fwd) == d)

        self.natural_cycles = build_cycle_lists(self.pivot_prices, mode="natural")
        self.confirm = confirmation_mask(df, self.direction)

    def period_mask(self, start: date | None, end: date | None) -> np.ndarray:
        m = np.ones(len(self.dates), dtype=bool)
        if start:
            m &= self.date_ords >= start.toordinal()
        if end:
            m &= self.date_ords <= end.toordinal()
        m &= np.arange(len(self.dates)) + FORWARD_RETURN_DAYS < len(self.dates)
        m &= self.direction != 0
        m &= np.isfinite(self.fwd_hit)
        return m

    def metrics_from_mask(
        self,
        in_window: np.ndarray,
        period: np.ndarray,
        *,
        require_confirmation: bool = False,
    ) -> dict[str, Any]:
        eval_m = period.copy()
        if require_confirmation:
            in_window = in_window & self.confirm
        in_m = eval_m & in_window
        out_m = eval_m & ~in_window
        hits_in = self.fwd_hit[in_m]
        hits_out = self.fwd_hit[out_m]
        hit_in = float(np.mean(hits_in)) if hits_in.size else None
        hit_out = float(np.mean(hits_out)) if hits_out.size else None
        lift = hit_in / hit_out if hit_in is not None and hit_out and hit_out > 0 else None
        return {
            "hit_rate_in": hit_in,
            "hit_rate_out": hit_out,
            "lift": lift,
            "n_in": int(hits_in.size),
            "n_out": int(hits_out.size),
            "hits_in": hits_in.astype(int).tolist(),
        }

    def metrics_for_cycles(
        self,
        cycle_lists: list[list[int]],
        pivot_dates: list[date] | None = None,
        *,
        start: date | None = None,
        end: date | None = None,
        require_confirmation: bool = False,
    ) -> dict[str, Any]:
        p_dates = pivot_dates if pivot_dates is not None else self.pivot_dates
        targets = projection_target_ords(p_dates, cycle_lists)
        in_win = window_mask(self.date_ords, targets)
        period = self.period_mask(start, end)
        return self.metrics_from_mask(in_win, period, require_confirmation=require_confirmation)

    def convergence_lift(
        self,
        cycle_lists: list[list[int]],
        pivot_dates: list[date] | None = None,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> float:
        p_dates = pivot_dates if pivot_dates is not None else self.pivot_dates
        targets = projection_target_ords(p_dates, cycle_lists)
        in_win = window_mask(self.date_ords, targets)
        period = self.period_mask(start, end)
        false_alarm = float(np.mean(in_win[period])) if period.any() else 0.0
        if false_alarm <= 0:
            return 0.0

        pivot_set = set(p_dates)
        detected = 0
        checked = 0
        for i, d in enumerate(self.dates):
            if d not in pivot_set:
                continue
            if start and d < start:
                continue
            if end and d > end:
                continue
            checked += 1
            lo = max(0, i - 3)
            if in_win[lo:i].any():
                detected += 1
        det_rate = detected / checked if checked else 0.0
        return det_rate / false_alarm


def calibrate_direct_scale(study: SymbolStudy) -> float:
    best_scale = 10.0
    best_lift = -1.0
    for scale in DIRECT_SCALES:
        cycles = build_cycle_lists(study.pivot_prices, mode="sqrt", direct_scale=float(scale))
        m = study.metrics_for_cycles(cycles, start=TRAIN_START, end=TRAIN_END)
        lift = m.get("lift") or 0.0
        if lift > best_lift:
            best_lift = lift
            best_scale = float(scale)
    return best_scale


def pooled_permutation_p(
    studies: list[SymbolStudy],
    cycle_mode: Literal["natural", "sqrt"],
    calibrated_scales: dict[str, float] | None,
    observed_hit_in: float,
    *,
    start: date | None = None,
    end: date | None = None,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(N_PERMUTATIONS):
        perm_hits: list[float] = []
        for study in studies:
            if len(study.pivot_dates) < 2:
                continue
            shuffled = rng.permutation(len(study.pivot_dates))
            s_dates = [study.pivot_dates[i] for i in shuffled]
            if cycle_mode == "natural":
                cycles = study.natural_cycles
                s_cycles = [cycles[i] for i in shuffled]
            else:
                scale = (calibrated_scales or {}).get(study.symbol, 10.0)
                base_cycles = build_cycle_lists(study.pivot_prices, mode="sqrt", direct_scale=scale)
                s_cycles = [base_cycles[i] for i in shuffled]
            m = study.metrics_for_cycles(s_cycles, pivot_dates=s_dates, start=start, end=end)
            perm_hits.extend(m.get("hits_in", []))
        perm_rate = float(np.mean(perm_hits)) if perm_hits else 0.0
        if perm_rate >= observed_hit_in:
            count_ge += 1
    return count_ge / N_PERMUTATIONS


def run_experiment_1(studies: list[SymbolStudy], *, min_symbols_lift: int) -> dict[str, Any]:
    log.info("Running Experiment 1 — calendar anniversaries")
    by_symbol: list[dict[str, Any]] = []
    pooled_hits_in: list[int] = []
    pooled_hits_out: list[int] = []

    for study in studies:
        if len(study.pivot_dates) < 2:
            log.warning("Skipping %s — insufficient pivots", study.symbol)
            continue
        m = study.metrics_for_cycles(study.natural_cycles)
        by_symbol.append({"symbol": study.symbol, **{k: v for k, v in m.items() if k != "hits_in"}})
        pooled_hits_in.extend(m.get("hits_in", []))
        in_win = window_mask(
            study.date_ords, projection_target_ords(study.pivot_dates, study.natural_cycles),
        )
        period = study.period_mask(None, None)
        out_hits = study.fwd_hit[period & ~in_win]
        pooled_hits_out.extend(out_hits.astype(int).tolist())

    hit_in = float(np.mean(pooled_hits_in)) if pooled_hits_in else 0.0
    hit_out = float(np.mean(pooled_hits_out)) if pooled_hits_out else 0.0
    lift = hit_in / hit_out if hit_out > 0 else 0.0
    perm_p = pooled_permutation_p(studies, "natural", None, hit_in)

    symbols_lift_gt_1 = sum(
        1 for r in by_symbol if (r.get("lift") or 0) > 1.0
    )
    passed = (
        lift > 1.05 and perm_p < 0.10 and symbols_lift_gt_1 >= min_symbols_lift
    )
    return {
        "hit_rate_in_window": hit_in,
        "hit_rate_outside": hit_out,
        "lift": lift,
        "permutation_p": perm_p,
        "n_in_window": len(pooled_hits_in),
        "n_outside": len(pooled_hits_out),
        "symbols_with_lift_gt_1": symbols_lift_gt_1,
        "min_symbols_lift_required": min_symbols_lift,
        "by_symbol": by_symbol,
        "verdict": "PASS" if passed else "FAIL",
    }


def run_experiment_2(
    studies: list[SymbolStudy],
    *,
    min_symbols_pass: int,
) -> dict[str, Any]:
    log.info("Running Experiment 2 — sqrt(P) squaring (holdout)")
    calibrated_scales: dict[str, float] = {}
    holdout_by_symbol: list[dict[str, Any]] = []
    symbol_lifts: list[float] = []

    for study in studies:
        if len(study.pivot_dates) < 2:
            continue
        scale = calibrate_direct_scale(study)
        calibrated_scales[study.symbol] = scale
        cycles = build_cycle_lists(study.pivot_prices, mode="sqrt", direct_scale=scale)
        m = study.metrics_for_cycles(cycles, start=HOLDOUT_START, end=HOLDOUT_END)
        conv_lift = study.convergence_lift(cycles, start=HOLDOUT_START, end=HOLDOUT_END)
        symbol_lifts.append(conv_lift)
        holdout_by_symbol.append({
            "symbol": study.symbol,
            "calibrated_scale": scale,
            "convergence_lift": conv_lift,
            **{k: v for k, v in m.items() if k != "hits_in"},
        })

    mean_conv_lift = float(np.mean(symbol_lifts)) if symbol_lifts else 0.0
    n_symbols_lift_gt_12 = sum(1 for x in symbol_lifts if x > 1.2)

    pooled_hits_in: list[int] = []
    for study in studies:
        if study.symbol not in calibrated_scales:
            continue
        cycles = build_cycle_lists(
            study.pivot_prices, mode="sqrt",
            direct_scale=calibrated_scales[study.symbol],
        )
        m = study.metrics_for_cycles(cycles, start=HOLDOUT_START, end=HOLDOUT_END)
        pooled_hits_in.extend(m.get("hits_in", []))
    obs_hit = float(np.mean(pooled_hits_in)) if pooled_hits_in else 0.0
    perm_p = pooled_permutation_p(
        studies, "sqrt", calibrated_scales, obs_hit,
        start=HOLDOUT_START, end=HOLDOUT_END, seed=99,
    )

    passed = mean_conv_lift > 1.20 and n_symbols_lift_gt_12 >= min_symbols_pass
    return {
        "holdout_convergence_lift": mean_conv_lift,
        "n_symbols_lift_gt_1_2": n_symbols_lift_gt_12,
        "n_symbols": len(holdout_by_symbol),
        "min_symbols_pass_required": min_symbols_pass,
        "permutation_p": perm_p,
        "calibrated_scales": calibrated_scales,
        "by_symbol": holdout_by_symbol,
        "verdict": "PASS" if passed else "FAIL",
    }


def print_summary(exp1: dict[str, Any], exp2: dict[str, Any]) -> None:
    min_exp1 = exp1.get("min_symbols_lift_required", 15)
    min_exp2 = exp2.get("min_symbols_pass_required", 12)
    print("=" * 60)
    print("EXPERIMENT 1 — CALENDAR ANNIVERSARIES")
    print(f"  Hit rate in window :  {100 * exp1['hit_rate_in_window']:.1f}%")
    print(f"  Hit rate outside   :  {100 * exp1['hit_rate_outside']:.1f}%")
    print(f"  Lift               :  {exp1['lift']:.2f}")
    print(f"  Permutation p      :  {exp1['permutation_p']:.3f}")
    print(
        f"  Symbols lift>1.0   :  {exp1.get('symbols_with_lift_gt_1', 'n/a')} "
        f"(need {min_exp1})",
    )
    print(f"  VERDICT            :  {exp1['verdict']}  [pass: lift>1.05 AND p<0.10]")
    print()
    print("EXPERIMENT 2 — sqrt(P) SQUARING  (holdout 2023-2026)")
    print(f"  Convergence lift   :  {exp2['holdout_convergence_lift']:.2f}")
    print(
        f"  Symbols lift>1.2   :  {exp2['n_symbols_lift_gt_1_2']} / {exp2['n_symbols']} "
        f"(need {min_exp2})",
    )
    print(f"  Permutation p      :  {exp2['permutation_p']:.3f}")
    print(f"  VERDICT            :  {exp2['verdict']}  [pass: lift>1.20 AND symbols>={min_exp2}]")
    print("=" * 60)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gann correct cycles event study")
    parser.add_argument(
        "--universe",
        choices=("equities", "macro"),
        default="equities",
        help="equities=25-name live stack; macro=oil/SPY/gold/EURUSD/BTC",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated tickers (overrides --universe)",
    )
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: runs/gann_correct_cycles or _macro)",
    )
    return parser.parse_args()


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.universe == "macro":
        return list(MACRO_UNIVERSE)
    return list(DEFAULT_UNIVERSE)


def _pass_thresholds(n_symbols: int) -> tuple[int, int]:
    """Proportional gates from 25-name study (15 lift>1.0, 12 lift>1.2)."""
    min_lift = max(1, math.ceil(n_symbols * 15 / 25))
    min_pass = max(1, math.ceil(n_symbols * 12 / 25))
    return min_lift, min_pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    symbol_list = _resolve_symbols(args)

    prices = load_prices(args.cache_dir, symbols=symbol_list)
    available = sorted(prices["symbol"].str.upper().unique())
    missing = [s for s in symbol_list if s not in available]
    if missing:
        log.error(
            "Missing cached prices for: %s. Fetch with:\n"
            "  python scripts/fetch_data.py --symbols %s "
            "--start 2020-01-01 --end 2026-07-20 --prices-provider tiingo "
            "--no-fundamentals --no-sentiment",
            ",".join(missing),
            ",".join(symbol_list),
        )
        sys.exit(1)

    symbols = [s for s in symbol_list if s in available]
    log.info("Loaded %d rows for %d symbols: %s", len(prices), len(symbols), symbols)

    studies: list[SymbolStudy] = []
    for sym in symbols:
        sdf = prices[prices["symbol"].str.upper() == sym].sort_values("date").reset_index(drop=True)
        if len(sdf) < 2 * PIVOT_ORDER + FORWARD_RETURN_DAYS + 20:
            log.warning("Skipping %s — insufficient history", sym)
            continue
        studies.append(SymbolStudy(sym, sdf))
        label = MACRO_LABELS.get(sym, sym)
        log.info(
            "%s (%s) pivots=%d bars=%d",
            sym, label, len(studies[-1].pivot_dates), len(sdf),
        )

    if not studies:
        log.error("No symbols with sufficient history")
        sys.exit(1)

    min_lift, min_pass = _pass_thresholds(len(studies))
    exp1 = run_experiment_1(studies, min_symbols_lift=min_lift)
    exp2 = run_experiment_2(studies, min_symbols_pass=min_pass)

    default_out = (
        "runs/gann_correct_cycles_macro"
        if args.universe == "macro" and not args.symbols
        else "runs/gann_correct_cycles"
    )
    out_dir = Path(args.output or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "universe": args.universe,
            "symbols": [s.symbol for s in studies],
            "symbol_labels": {s.symbol: MACRO_LABELS.get(s.symbol, s.symbol) for s in studies},
            "train_period": [str(TRAIN_START), str(TRAIN_END)],
            "holdout_period": [str(HOLDOUT_START), str(HOLDOUT_END)],
            "pivot_order": PIVOT_ORDER,
            "min_swing_pct": MIN_SWING_PCT,
            "window_tolerance_days": WINDOW_TOLERANCE_DAYS,
            "n_permutations": N_PERMUTATIONS,
            "pass_thresholds": {"min_symbols_lift_gt_1": min_lift, "min_symbols_lift_gt_1_2": min_pass},
        },
        "experiment_1": exp1,
        "experiment_2": exp2,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Saved results to %s", out_path)

    print_summary(exp1, exp2)


if __name__ == "__main__":
    main()
