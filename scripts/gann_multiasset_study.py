#!/usr/bin/env python3
"""Gann multi-asset cycles study — weekly bars, yfinance data.

Tests natural calendar cycles and price-derived sqrt(P) squaring on
commodities, indices, FX, and crypto. Self-contained; no firm strategy imports.

Usage::

    pip install yfinance   # if not installed
    python scripts/gann_multiasset_study.py
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.signal import argrelextrema

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ASSET_TICKERS: dict[str, list[str]] = {
    "Gold": ["GC=F", "XAUUSD=X"],
    "SP500": ["^GSPC", "SPY"],
    "Oil": ["CL=F"],
    "EURUSD": ["EURUSD=X"],
    "Bitcoin": ["BTC-USD"],
}

CACHE_DIR = Path("data/multiasset_cache")
OUTPUT_PATH = Path("runs/gann_multiasset/results.json")

DATA_START = "2015-01-01"
DATA_END = "2026-07-01"
TRAIN_END = pd.Timestamp("2020-01-01")
HOLDOUT_START = pd.Timestamp("2020-01-01")
HOLDOUT_END = pd.Timestamp("2026-07-01")

NATURAL_CYCLES_WEEKS = [4, 7, 13, 26, 39, 52, 65, 78, 91, 104]
SQRT_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SCALE_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
BITCOIN_HALVING_WEEKS = 208

PIVOT_ORDER = 8
MIN_SWING_PCT = 0.04
WINDOW_WEEKS = 2
N_PERMUTATIONS = 500
N_RANDOM_BASELINE = 2000
FORWARD_WEEKS_EXP3 = 4

PivotType = Literal["high", "low"]
Pivot = tuple[int, float, PivotType]

CACHE_MAX_AGE_HOURS = 24


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _download_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed; run: pip install yfinance")
        return None
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw.rename(columns=str.lower)
        raw = raw.reset_index()
        date_col = "date" if "date" in raw.columns else raw.columns[0]
        raw = raw.rename(columns={date_col: "date"})
        raw["date"] = pd.to_datetime(raw["date"])
        for c in ("open", "high", "low", "close"):
            if c not in raw.columns:
                log.warning("Missing column %s for %s", c, ticker)
                return None
        if "volume" not in raw.columns:
            raw["volume"] = 0.0
        return raw[["date", "open", "high", "low", "close", "volume"]].dropna()
    except Exception as exc:
        log.warning("yfinance download failed for %s (%s)", ticker, exc)
        return None


def load_asset_weekly(name: str, tickers: list[str]) -> pd.DataFrame | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.parquet"
    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_h < CACHE_MAX_AGE_HOURS:
            log.info("Cache hit: %s (%.1fh old)", name, age_h)
            daily = pd.read_parquet(cache_path)
            return resample_weekly(daily)

    for ticker in tickers:
        log.info("Downloading %s via %s …", name, ticker)
        daily = _download_yfinance(ticker, DATA_START, DATA_END)
        if daily is not None and len(daily) > 100:
            daily.to_parquet(cache_path, index=False)
            return resample_weekly(daily)
        log.warning("No data for %s / %s", name, ticker)
    return None


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df = df.set_index("date").sort_index()
    weekly = df.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    weekly = weekly.reset_index()
    return weekly


# ---------------------------------------------------------------------------
# Pivots & confirmation
# ---------------------------------------------------------------------------

def detect_major_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    order: int = PIVOT_ORDER,
    min_swing_pct: float = MIN_SWING_PCT,
) -> list[Pivot]:
    n = len(highs)
    if n < 2 * order + 1:
        return []

    hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]

    raw: list[tuple[int, float, PivotType]] = []
    for i in hi_idx:
        raw.append((int(i), float(highs[i]), "high"))
    for i in lo_idx:
        raw.append((int(i), float(lows[i]), "low"))
    raw.sort(key=lambda x: x[0])

    filtered: list[Pivot] = []
    for idx, (i, price, ptype) in enumerate(raw):
        prev_p = raw[idx - 1][1] if idx > 0 else price
        next_p = raw[idx + 1][1] if idx + 1 < len(raw) else price
        sp = abs(price - prev_p) / prev_p if prev_p > 0 else 0.0
        sn = abs(price - next_p) / next_p if next_p > 0 else 0.0
        if sp >= min_swing_pct and sn >= min_swing_pct:
            filtered.append((i, price, ptype))
    return filtered


def pivot_density_per_year(pivots: list[Pivot], n_bars: int) -> float:
    if n_bars < 2:
        return 0.0
    years = n_bars / 52.0
    return len(pivots) / years if years > 0 else 0.0


def swing_index_set(pivots: list[Pivot]) -> np.ndarray:
    if not pivots:
        return np.array([], dtype=np.int32)
    return np.array(sorted({p[0] for p in pivots}), dtype=np.int32)


def swing_neighborhood_mask(n: int, swing_idxs: np.ndarray, radius: int = WINDOW_WEEKS) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if swing_idxs.size:
        mask[swing_idxs] = True
    structure = np.ones(2 * radius + 1, dtype=bool)
    return binary_dilation(mask, structure=structure)


def compute_atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    for i in range(14, n):
        atr[i] = float(np.mean(tr[i - 13 : i + 1]))
    return atr


def confirmation_mask_weekly(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    vol = df["volume"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    close = df["close"].values.astype(float)
    atr = compute_atr14(high, low, close)

    confirmed = np.zeros(n, dtype=bool)
    for i in range(20, n):
        vol_ma = float(np.mean(vol[max(0, i - 20) : i]))
        volume_spike = vol_ma > 0 and vol[i] > 1.5 * vol_ma
        range_expand = not np.isnan(atr[i]) and (high[i] - low[i]) > 1.5 * atr[i]
        bar_range = high[i] - low[i]
        closes_strong = bar_range > 0 and abs(close[i] - open_[i]) > 0.6 * bar_range
        confirmed[i] = volume_spike or range_expand or closes_strong
    return confirmed


# ---------------------------------------------------------------------------
# Projection helpers (vectorized)
# ---------------------------------------------------------------------------

def build_natural_targets(pivot_idxs: np.ndarray) -> np.ndarray:
    """All projection center indices from natural weekly cycles."""
    if pivot_idxs.size == 0:
        return np.array([], dtype=np.int32)
    cycles = np.array(NATURAL_CYCLES_WEEKS, dtype=np.int32)
    # (n_pivots, n_cycles)
    targets = pivot_idxs[:, None] + cycles[None, :]
    return targets.ravel()


def build_sqrt_targets(
    pivot_idxs: np.ndarray,
    pivot_prices: np.ndarray,
    scale: float,
    multipliers: list[float] | None = None,
) -> np.ndarray:
    if pivot_idxs.size == 0 or scale <= 0:
        return np.array([], dtype=np.int32)
    prices = np.maximum(pivot_prices.astype(float), 1e-9)
    mults = np.array(multipliers or SQRT_MULTIPLIERS, dtype=float)
    weeks = np.sqrt(prices[:, None] / scale) * mults[None, :]
    weeks = np.round(weeks).astype(np.int32)
    weeks = np.clip(weeks, 1, 520)
    targets = pivot_idxs[:, None] + weeks
    return targets.ravel()


def mask_targets_in_range(targets: np.ndarray, n: int) -> np.ndarray:
    valid = (targets >= WINDOW_WEEKS) & (targets < n - WINDOW_WEEKS)
    return targets[valid]


def projection_hit_rate(
    targets: np.ndarray,
    swing_near: np.ndarray,
    gate_near: np.ndarray | None,
    n: int,
) -> tuple[float, int]:
    """Fraction of projection windows containing a swing; optional gate filter."""
    targets = mask_targets_in_range(targets, n)
    if targets.size == 0:
        return 0.0, 0

    if gate_near is not None:
        keep = gate_near[targets]
        targets = targets[keep]
        if targets.size == 0:
            return 0.0, 0

    hits = swing_near[targets].sum()
    return float(hits) / float(targets.size), int(targets.size)


def random_baseline_rate(
    swing_near: np.ndarray,
    n: int,
    rng: np.random.Generator,
    n_samples: int = N_RANDOM_BASELINE,
) -> float:
    lo, hi = WINDOW_WEEKS, n - WINDOW_WEEKS - 1
    if hi <= lo:
        return 0.0
    centers = rng.integers(lo, hi + 1, size=n_samples)
    return float(swing_near[centers].mean())


def permutation_p_lift(
    pivot_idxs: np.ndarray,
    swing_near: np.ndarray,
    gate_near: np.ndarray | None,
    n: int,
    observed_lift: float,
    cycle_weeks: list[int],
    rng: np.random.Generator,
    n_perm: int = N_PERMUTATIONS,
) -> float:
    if pivot_idxs.size == 0 or observed_lift <= 0:
        return 1.0
    cycles = np.array(cycle_weeks, dtype=np.int32)
    count_ge = 0
    for _ in range(n_perm):
        # Shuffle projection offsets while holding pivot positions fixed
        shuffled_cycles = rng.choice(cycles, size=(pivot_idxs.size, cycles.size), replace=True)
        targets = (pivot_idxs[:, None] + shuffled_cycles).ravel()
        hit, _ = projection_hit_rate(targets, swing_near, gate_near, n)
        base = random_baseline_rate(swing_near, n, rng, n_samples=500)
        lift = hit / base if base > 0 else 0.0
        if lift >= observed_lift:
            count_ge += 1
    return count_ge / n_perm


def bar_convergence(
    targets: np.ndarray,
    n: int,
    start_bar: int,
    end_bar: int,
) -> np.ndarray:
    """Per-bar convergence: fraction of projections landing within ±WINDOW of bar."""
    conv = np.zeros(n, dtype=float)
    targets = mask_targets_in_range(targets, n)
    if targets.size == 0:
        return conv

    # For each bar i, count targets with |t - i| <= WINDOW
    for i in range(start_bar, end_bar):
        near = np.abs(targets - i) <= WINDOW_WEEKS
        conv[i] = near.sum() / targets.size
    return conv


def convergence_lift(
    conv: np.ndarray,
    swing_idxs: np.ndarray,
    lookback: int = 3,
    start_bar: int = 0,
    end_bar: int | None = None,
) -> float:
    end_bar = end_bar if end_bar is not None else len(conv)
    near_vals: list[float] = []
    near_set: set[int] = set()
    for sidx in swing_idxs:
        if sidx < lookback or sidx < start_bar or sidx >= end_bar:
            continue
        for b in range(max(start_bar, sidx - lookback), sidx):
            near_vals.append(float(conv[b]))
            near_set.add(b)

    other = [float(conv[b]) for b in range(start_bar, end_bar) if b not in near_set and conv[b] > 0]
    if not near_vals or not other:
        return 0.0
    return float(np.mean(near_vals)) / float(np.mean(other))


def permutation_p_convergence_lift(
    pivot_idxs: np.ndarray,
    pivot_prices: np.ndarray,
    scale: float,
    swing_idxs: np.ndarray,
    n: int,
    start_bar: int,
    end_bar: int,
    observed_lift: float,
    rng: np.random.Generator,
    cycle_weeks: list[int] | None = None,
) -> float:
    count_ge = 0
    if cycle_weeks is not None:
        cycles = np.array(cycle_weeks, dtype=np.int32)
        for _ in range(N_PERMUTATIONS):
            shuffled = rng.choice(cycles, size=(pivot_idxs.size, cycles.size), replace=True)
            targets = (pivot_idxs[:, None] + shuffled).ravel()
            conv = bar_convergence(targets, n, start_bar, end_bar)
            lift = convergence_lift(conv, swing_idxs, start_bar=start_bar, end_bar=end_bar)
            if lift >= observed_lift:
                count_ge += 1
        return count_ge / N_PERMUTATIONS

    mults = np.array(SQRT_MULTIPLIERS)
    for _ in range(N_PERMUTATIONS):
        shuffled_weeks = rng.choice(
            np.arange(1, 105), size=(pivot_idxs.size, mults.size), replace=True,
        )
        targets = (pivot_idxs[:, None] + shuffled_weeks).ravel()
        conv = bar_convergence(targets, n, start_bar, end_bar)
        lift = convergence_lift(conv, swing_idxs, start_bar=start_bar, end_bar=end_bar)
        if lift >= observed_lift:
            count_ge += 1
    return count_ge / N_PERMUTATIONS


def bar_index_for_date(dates: pd.Series, ts: pd.Timestamp) -> int:
    idx = dates.searchsorted(ts)
    return int(min(idx, len(dates) - 1))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_experiment_1(
    name: str,
    weekly: pd.DataFrame,
    pivots: list[Pivot],
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = len(weekly)
    highs = weekly["high"].values.astype(float)
    lows = weekly["low"].values.astype(float)
    pivot_idxs = np.array([p[0] for p in pivots], dtype=np.int32)
    swing_idxs = swing_index_set(pivots)
    swing_near = swing_neighborhood_mask(n, swing_idxs)
    gate = confirmation_mask_weekly(weekly)
    gate_near = binary_dilation(gate, structure=np.ones(2 * WINDOW_WEEKS + 1, dtype=bool))

    targets = build_natural_targets(pivot_idxs)
    hit_ng, n_proj = projection_hit_rate(targets, swing_near, None, n)
    hit_wg, _ = projection_hit_rate(targets, swing_near, gate_near, n)
    base = random_baseline_rate(swing_near, n, rng)
    lift_ng = hit_ng / base if base > 0 else 0.0
    lift_wg = hit_wg / base if base > 0 else 0.0

    p_ng = permutation_p_lift(pivot_idxs, swing_near, None, n, lift_ng, NATURAL_CYCLES_WEEKS, rng)
    p_wg = permutation_p_lift(pivot_idxs, swing_near, gate_near, n, lift_wg, NATURAL_CYCLES_WEEKS, rng)

    passed = lift_wg > 1.15 and p_wg < 0.10
    gate_improves_lift = lift_wg > lift_ng
    return {
        "hit_rate_no_gate": hit_ng,
        "hit_rate_with_gate": hit_wg,
        "base_rate": base,
        "lift_no_gate": lift_ng,
        "lift_with_gate": lift_wg,
        "permutation_p_no_gate": p_ng,
        "permutation_p_with_gate": p_wg,
        "n_projections": n_proj,
        "gate_improves_lift": gate_improves_lift,
        "verdict": "PASS" if passed else "FAIL",
    }


def calibrate_scale(
    weekly: pd.DataFrame,
    pivots: list[Pivot],
    train_start: int,
    train_end: int,
    rng: np.random.Generator,
) -> float:
    best_scale = 1.0
    best_margin = -1.0
    n = len(weekly)
    pivot_idxs = np.array([p[0] for p in pivots if train_start <= p[0] < train_end], dtype=np.int32)
    pivot_prices = np.array(
        [p[1] for p in pivots if train_start <= p[0] < train_end], dtype=float,
    )
    if pivot_idxs.size == 0:
        return 1.0

    swing_idxs = swing_index_set(pivots)
    swing_near = swing_neighborhood_mask(n, swing_idxs)

    for scale in SCALE_GRID:
        targets = build_sqrt_targets(pivot_idxs, pivot_prices, scale)
        hit, _ = projection_hit_rate(targets, swing_near, None, n)
        base = random_baseline_rate(swing_near, n, rng, n_samples=800)
        margin = hit - base
        if margin > best_margin:
            best_margin = margin
            best_scale = scale
    return best_scale


def run_experiment_2(
    name: str,
    weekly: pd.DataFrame,
    pivots: list[Pivot],
    rng: np.random.Generator,
) -> dict[str, Any]:
    dates = weekly["date"]
    n = len(weekly)
    train_start = bar_index_for_date(dates, pd.Timestamp("2015-01-01"))
    train_end = bar_index_for_date(dates, TRAIN_END)
    hold_start = bar_index_for_date(dates, HOLDOUT_START)
    hold_end = bar_index_for_date(dates, HOLDOUT_END)

    pivot_idxs = np.array([p[0] for p in pivots], dtype=np.int32)
    pivot_prices = np.array([p[1] for p in pivots], dtype=float)
    swing_idxs = swing_index_set(pivots)

    best_scale = calibrate_scale(weekly, pivots, train_start, train_end, rng)
    hold_pivot_mask = (pivot_idxs < hold_end) & (pivot_idxs >= train_start - 52)
    h_idxs = pivot_idxs[hold_pivot_mask]
    h_prices = pivot_prices[hold_pivot_mask]

    targets = build_sqrt_targets(h_idxs, h_prices, best_scale)
    conv_ng = bar_convergence(targets, n, hold_start, hold_end)
    lift_ng = convergence_lift(conv_ng, swing_idxs, start_bar=hold_start, end_bar=hold_end)

    gate = confirmation_mask_weekly(weekly)
    gate_near = binary_dilation(gate, structure=np.ones(2 * WINDOW_WEEKS + 1, dtype=bool))
    # Gate-filtered convergence: zero bars without gate in window of any projection
    conv_wg = conv_ng.copy()
    for i in range(hold_start, hold_end):
        if not gate_near[i]:
            conv_wg[i] = 0.0
    lift_wg = convergence_lift(conv_wg, swing_idxs, start_bar=hold_start, end_bar=hold_end)

    p_ng = permutation_p_convergence_lift(
        h_idxs, h_prices, best_scale, swing_idxs, n,
        hold_start, hold_end, lift_ng, rng,
    )
    p_wg = permutation_p_convergence_lift(
        h_idxs, h_prices, best_scale, swing_idxs, n,
        hold_start, hold_end, lift_wg, rng,
    )

    passed = lift_wg > 1.20 and p_wg < 0.10
    return {
        "best_scale": best_scale,
        "convergence_lift_no_gate": lift_ng,
        "convergence_lift_with_gate": lift_wg,
        "permutation_p_no_gate": p_ng,
        "permutation_p_with_gate": p_wg,
        "gate_improves_lift": lift_wg > lift_ng,
        "verdict": "PASS" if passed else "FAIL",
    }


def bitcoin_halving_lift(weekly: pd.DataFrame, pivots: list[Pivot]) -> float:
    dates = weekly["date"]
    n = len(weekly)
    hold_start = bar_index_for_date(dates, HOLDOUT_START)
    hold_end = bar_index_for_date(dates, HOLDOUT_END)
    pivot_idxs = np.array([p[0] for p in pivots], dtype=np.int32)
    if pivot_idxs.size == 0:
        return 0.0
    targets = pivot_idxs + BITCOIN_HALVING_WEEKS
    conv = bar_convergence(targets, n, hold_start, hold_end)
    return convergence_lift(conv, swing_index_set(pivots), start_bar=hold_start, end_bar=hold_end)


def run_experiment_3(
    weekly: pd.DataFrame,
    pivots: list[Pivot],
    best_scale: float,
) -> dict[str, Any]:
    n = len(weekly)
    close = weekly["close"].values.astype(float)
    dates = weekly["date"]
    hold_start = bar_index_for_date(dates, HOLDOUT_START)
    hold_end = bar_index_for_date(dates, HOLDOUT_END) - FORWARD_WEEKS_EXP3

    pivot_idxs = np.array([p[0] for p in pivots], dtype=np.int32)
    pivot_prices = np.array([p[1] for p in pivots], dtype=float)
    pivot_types = [p[2] for p in pivots]

    targets = build_sqrt_targets(pivot_idxs, pivot_prices, best_scale)
    conv = bar_convergence(targets, n, hold_start, hold_end)

    bull_rets: list[float] = []
    bear_rets: list[float] = []
    all_rets: list[float] = []

    type_by_idx = {p[0]: p[2] for p in pivots}
    threshold = np.percentile(conv[hold_start:hold_end][conv[hold_start:hold_end] > 0], 75) if (conv[hold_start:hold_end] > 0).any() else 0.05

    for i in range(hold_start, hold_end):
        if i + FORWARD_WEEKS_EXP3 >= n:
            continue
        fwd = close[i + FORWARD_WEEKS_EXP3] / close[i] - 1.0
        all_rets.append(fwd)
        if conv[i] < threshold:
            continue
        # Direction from nearest triggering pivot
        direction = 0
        for pidx, price, ptype in pivots:
            for mult in SQRT_MULTIPLIERS:
                tw = int(round(np.sqrt(price / best_scale) * mult))
                if abs((pidx + tw) - i) <= WINDOW_WEEKS:
                    direction = 1 if ptype == "low" else -1
                    break
        if direction > 0:
            bull_rets.append(fwd)
        elif direction < 0:
            bear_rets.append(fwd)

    return {
        "mean_fwd_bullish_signals": float(np.mean(bull_rets)) if bull_rets else None,
        "mean_fwd_bearish_signals": float(np.mean(bear_rets)) if bear_rets else None,
        "mean_fwd_all_bars": float(np.mean(all_rets)) if all_rets else None,
        "n_bull_signals": len(bull_rets),
        "n_bear_signals": len(bear_rets),
        "convergence_threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def recommended_next_step(exp1_pass: int, exp2_pass: int) -> str:
    e1 = exp1_pass >= 3
    e2 = exp2_pass >= 3
    if e1 and e2:
        return "Build confirmed-signal strategy on passing assets; proceed to IC and portfolio test"
    if e1 and not e2:
        return "Natural cycles have weak timing value; retest with monthly bars before building directional signal"
    if not e1 and e2:
        return "Price-derived squaring has timing value; build direction signal from pivot type, not price return"
    return "No timing value on any asset class tested; retire Gann cycles research permanently"


def print_summary(
    assets: list[str],
    density: dict[str, float],
    exp1: dict[str, Any],
    exp2: dict[str, Any],
    exp3: dict[str, Any],
    final: dict[str, Any],
) -> None:
    print("=" * 60)
    print("GANN MULTIASSET CYCLES STUDY")
    print("=" * 60)
    print("PIVOT DENSITY (pivots/year)")
    for a in assets:
        d = density.get(a, 0.0)
        flag = "" if 3 <= d <= 10 else "  [WARN: outside 4-8 target]"
        print(f"  {a:<8}: {d:.1f}{flag}")
    print()
    print("EXPERIMENT 1 — NATURAL CALENDAR CYCLES (WEEKLY)")
    print("            no gate         with gate")
    for a in assets:
        r = exp1["per_asset"].get(a, {})
        print(
            f"  {a:<8}: lift={r.get('lift_no_gate', 0):.2f} p={r.get('permutation_p_no_gate', 1):.2f}  "
            f"lift={r.get('lift_with_gate', 0):.2f} p={r.get('permutation_p_with_gate', 1):.2f}  "
            f"[{r.get('verdict', 'FAIL')}]",
        )
    print(f"  Assets passing: {exp1['assets_passing']}/{len(assets)}")
    gates_ok = sum(
        1 for a in assets if exp1["per_asset"].get(a, {}).get("gate_improves_lift")
    )
    if gates_ok < len(assets):
        print(f"  NOTE: confirmation gate improved lift on {gates_ok}/{len(assets)} assets only")
    print()
    print("EXPERIMENT 2 — sqrt(P) SQUARING (holdout 2020-2026)")
    print("            no gate         with gate")
    for a in assets:
        r = exp2["per_asset"].get(a, {})
        print(
            f"  {a:<8}: lift={r.get('convergence_lift_no_gate', 0):.2f} "
            f"p={r.get('permutation_p_no_gate', 1):.2f}  "
            f"lift={r.get('convergence_lift_with_gate', 0):.2f} "
            f"p={r.get('permutation_p_with_gate', 1):.2f}  "
            f"[{r.get('verdict', 'FAIL')}]",
        )
    print(f"  Bitcoin halving cycle lift: {exp2.get('bitcoin_halving_cycle_lift', 0):.2f}")
    print(f"  Assets passing: {exp2['assets_passing']}/{len(assets)}")
    print()
    ran = exp3.get("ran", False)
    print(f"EXPERIMENT 3 — DIRECTION (ran: {'YES' if ran else 'NO'})")
    if ran:
        for a, r in exp3.get("per_asset", {}).items():
            print(
                f"  {a}: bull_fwd={r.get('mean_fwd_bullish_signals')} "
                f"bear_fwd={r.get('mean_fwd_bearish_signals')}",
            )
    else:
        print(f"  {exp3.get('reason_skipped', '')}")
    print()
    print("FINAL VERDICT")
    print(f"  Exp1: {exp1['overall_verdict']}  Exp2: {exp2['overall_verdict']}")
    fv = final.get("cycles_have_timing_value_on_commodities", False)
    verdict_line = (
        "TIMING VALUE DETECTED ON ≥3 ASSETS"
        if fv
        else "NO TIMING VALUE ON ANY ASSET"
    )
    print(f"  → {verdict_line}")
    print(f"  → Recommended next step: {final.get('recommended_next_step', '')}")
    print("=" * 60)

    if not fv:
        print(
            "\nNOTE: Full failure across all assets and both experiments would confirm that\n"
            "Gann's cycle mechanisms do not produce systematic, rule-based, out-of-sample\n"
            "timing value on any of the five tested asset classes. This does not disprove\n"
            "discretionary Gann trading (confirmation gate and pivot selection are applied\n"
            "manually by practitioners). It means the geometric timing mechanism alone,\n"
            "without human filtering, carries no predictive information detectable by a\n"
            "permutation-controlled event study. Recommended action: retire Gann cycle\n"
            "research permanently in this system.",
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(42)

    assets_data: dict[str, pd.DataFrame] = {}
    weekly_bars: dict[str, int] = {}
    density: dict[str, float] = {}
    pivots_by_asset: dict[str, list[Pivot]] = {}

    for name, tickers in ASSET_TICKERS.items():
        weekly = load_asset_weekly(name, tickers)
        if weekly is None or weekly.empty:
            log.warning("Skipping asset %s — no data", name)
            continue
        assets_data[name] = weekly
        weekly_bars[name] = len(weekly)
        highs = weekly["high"].values.astype(float)
        lows = weekly["low"].values.astype(float)
        pivots = detect_major_pivots(highs, lows)
        pivots_by_asset[name] = pivots
        dpy = pivot_density_per_year(pivots, len(weekly))
        density[name] = dpy
        if dpy < 3 or dpy > 10:
            log.warning("%s pivot density %.1f/year (target 4-8)", name, dpy)

    assets = list(assets_data.keys())
    if not assets:
        log.error("No assets loaded; exiting")
        raise SystemExit(1)

    exp1_per: dict[str, Any] = {}
    for name in assets:
        exp1_per[name] = run_experiment_1(name, assets_data[name], pivots_by_asset[name], rng)

    exp1_pass = sum(1 for r in exp1_per.values() if r["verdict"] == "PASS")
    exp1_overall = "PASS" if exp1_pass >= 3 else "FAIL"

    exp2_per: dict[str, Any] = {}
    for name in assets:
        exp2_per[name] = run_experiment_2(name, assets_data[name], pivots_by_asset[name], rng)

    exp2_pass = sum(1 for r in exp2_per.values() if r["verdict"] == "PASS")
    exp2_overall = "PASS" if exp2_pass >= 3 else "FAIL"

    btc_halving = 0.0
    if "Bitcoin" in assets_data:
        btc_halving = bitcoin_halving_lift(assets_data["Bitcoin"], pivots_by_asset["Bitcoin"])

    exp3: dict[str, Any] = {"ran": False, "reason_skipped": "", "per_asset": {}}
    if exp1_overall == "PASS" or exp2_overall == "PASS":
        exp3["ran"] = True
        for name in assets:
            scale = exp2_per[name]["best_scale"]
            exp3["per_asset"][name] = run_experiment_3(assets_data[name], pivots_by_asset[name], scale)
    else:
        exp3["reason_skipped"] = "SKIP EXP 3 — upstream experiments failed"

    timing_value = exp1_pass >= 3 or exp2_pass >= 3
    final = {
        "cycles_have_timing_value_on_commodities": timing_value,
        "recommended_next_step": recommended_next_step(exp1_pass, exp2_pass),
    }

    payload = {
        "run_timestamp": datetime.now().isoformat(),
        "assets": assets,
        "weekly_bars_per_asset": weekly_bars,
        "pivot_density_per_year": {k: round(v, 2) for k, v in density.items()},
        "experiment_1": {
            "per_asset": exp1_per,
            "assets_passing": exp1_pass,
            "overall_verdict": exp1_overall,
        },
        "experiment_2": {
            "training_window": "2015–2020",
            "holdout_window": "2020–2026",
            "per_asset": exp2_per,
            "bitcoin_halving_cycle_lift": round(btc_halving, 4),
            "assets_passing": exp2_pass,
            "overall_verdict": exp2_overall,
        },
        "experiment_3": exp3,
        "final_verdict": final,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Saved %s", OUTPUT_PATH)

    print_summary(assets, density, payload["experiment_1"], payload["experiment_2"], exp3, final)


if __name__ == "__main__":
    main()
