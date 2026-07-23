"""W.D. Gann composite strategy.

Financial intuition:
    William Delbert Gann (1878-1955) developed a suite of geometric and
    cyclical techniques premised on the idea that price and time are
    interchangeable dimensions and that markets move in predictable
    geometric patterns.  This module quantifies five of his core methods
    and combines them into a single composite signal per symbol.

    1. **Gann Angles** – trend direction from geometric price/time lines
       drawn off significant pivot points (1x1, 2x1, 1x2 slopes).
    2. **Square of Nine** – support/resistance proximity derived from the
       Gann spiral (cardinal/diagonal levels around the current price).
    3. **Time Cycles** – convergence of Gann's key calendar-day cycles
       (30, 60, 90, 120, 144, 180, 270, 360) projected from recent
       pivots to the current date.
    4. **Swing Indicator** – Gann's bar-by-bar swing chart tracking
       consecutive higher-highs/higher-lows (upswing) vs lower-highs/
       lower-lows (downswing).
    5. **Retracement Levels** – position within Gann's 8-part price
       grid (0/8 through 8/8); the 4/8 (50 %) level is the key pivot.

Data inputs:
    OHLC + adjusted-close prices from PitView.prices() with a lookback
    long enough to detect pivots and compute ranges (default 120+ days).

Signal logic:
    Each sub-method produces a raw score in [-1, +1].  The composite is
    a weighted average (configurable via ``sub_weights``), then cross-
    sectionally z-scored across the universe and clipped to [-3, 3].

Portfolio construction approach:
    Long the highest composite scores, short the lowest.  Gann signals
    tend to be shorter-horizon (5-21 d), so pair with a weekly or
    bi-weekly rebalance.

Risk notes:
    Gann techniques are geometry-based heuristics, not grounded in
    risk-factor theory.  They can produce spurious signals in trendless,
    choppy markets.  To mitigate this:

    1. A built-in **trend-strength filter** (directional-movement ratio,
       similar to ADX) dampens scores when the market lacks a clear trend.
       Controlled via the ``trend_filter_lookback`` and
       ``trend_filter_threshold`` params.
    2. A **confidence floor** (``min_confidence``) suppresses signals
       whose composite confidence falls below a threshold, preventing
       low-conviction noise from entering the pipeline.
    3. Signals are emitted with ``horizon="10d"`` and are best paired
       with weekly rebalancing and strict position-size / drawdown
       limits from the Risk Manager agent.
    4. Always combine with fundamental or factor overlays for
       diversification; do not rely on Gann signals in isolation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    order: int = 5,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (pivot_highs, pivot_lows) as lists of (index, price).

    A pivot high at bar *i* means highs[i] is the max in the window
    [i-order, i+order].  Similarly for pivot lows.
    """
    n = len(highs)
    pivot_highs: list[tuple[int, float]] = []
    pivot_lows: list[tuple[int, float]] = []
    for i in range(order, n - order):
        if highs[i] == max(highs[i - order : i + order + 1]):
            pivot_highs.append((i, float(highs[i])))
        if lows[i] == min(lows[i - order : i + order + 1]):
            pivot_lows.append((i, float(lows[i])))
    return pivot_highs, pivot_lows


def _gann_angles(
    current_price: float,
    current_idx: int,
    pivot_highs: list[tuple[int, float]],
    pivot_lows: list[tuple[int, float]],
    unit: float,
) -> tuple[float, float]:
    """Compute Gann-angle score and confidence.

    Draws a 1x1 ascending line from the most recent pivot low and a 1x1
    descending line from the most recent pivot high.  *unit* converts bars
    to price units (the "price per bar" for a 45-degree line).

    Returns (score in [-1,1], confidence in [0,1]).
    """
    score = 0.0
    votes = 0
    weight_sum = 0.0

    # Ascending line from latest pivot low
    if pivot_lows:
        idx_low, p_low = pivot_lows[-1]
        bars = current_idx - idx_low
        if bars > 0:
            for speed, w in [(1.0, 1.0), (2.0, 0.5), (0.5, 0.5)]:
                line_price = p_low + speed * unit * bars
                diff = (current_price - line_price) / (unit * bars) if unit * bars > 0 else 0.0
                score += np.clip(diff, -1.0, 1.0) * w
                weight_sum += w
                votes += 1

    # Descending line from latest pivot high
    if pivot_highs:
        idx_high, p_high = pivot_highs[-1]
        bars = current_idx - idx_high
        if bars > 0:
            for speed, w in [(1.0, 1.0), (2.0, 0.5), (0.5, 0.5)]:
                line_price = p_high - speed * unit * bars
                diff = (current_price - line_price) / (unit * bars) if unit * bars > 0 else 0.0
                score += np.clip(diff, -1.0, 1.0) * w
                weight_sum += w
                votes += 1

    if weight_sum == 0:
        return 0.0, 0.0
    score /= weight_sum
    confidence = min(abs(score), 1.0) * min(votes / 6.0, 1.0)
    return float(np.clip(score, -1.0, 1.0)), float(confidence)


def _square_of_nine_levels(price: float, n_levels: int = 3) -> tuple[list[float], list[float]]:
    """Compute nearest Square-of-Nine support and resistance levels.

    The spiral formula places perfect squares on the cardinal cross.
    Returns (supports_below, resistances_above), each sorted by
    distance from *price*.
    """
    if price <= 0:
        return [], []

    sqrt_p = math.sqrt(price)
    base = int(sqrt_p)

    candidates: list[float] = []
    for offset in range(-n_levels - 1, n_levels + 2):
        val = base + offset
        if val > 0:
            candidates.append(float(val * val))
            # Diagonal levels sit at half-integer squares
            candidates.append(float((val + 0.5) ** 2))

    supports = sorted([c for c in candidates if c < price], key=lambda x: price - x)
    resistances = sorted([c for c in candidates if c > price], key=lambda x: x - price)
    return supports[:n_levels], resistances[:n_levels]


def _sq9_score(
    current_price: float,
    atr: float,
    price_direction: float,
) -> tuple[float, float]:
    """Score based on proximity to Square-of-Nine levels.

    Bullish near support when price rising, bearish near resistance
    when price stalling.  Normalized by ATR.
    """
    if atr <= 0 or current_price <= 0:
        return 0.0, 0.0

    supports, resistances = _square_of_nine_levels(current_price)
    if not supports and not resistances:
        return 0.0, 0.0

    dist_to_support = (current_price - supports[0]) / atr if supports else 999.0
    dist_to_resist = (resistances[0] - current_price) / atr if resistances else 999.0

    # Closer to support + rising -> bullish; closer to resistance + falling -> bearish
    if dist_to_support < dist_to_resist:
        proximity = max(0.0, 1.0 - dist_to_support / 3.0)
        raw = proximity * max(price_direction, 0.0)
    else:
        proximity = max(0.0, 1.0 - dist_to_resist / 3.0)
        raw = -proximity * max(-price_direction, 0.0)

    # If direction is ambiguous, score damps toward zero
    if abs(raw) < 0.01:
        raw = 0.0

    return float(np.clip(raw, -1.0, 1.0)), float(min(abs(raw), 1.0))


def _time_cycles(
    current_idx: int,
    total_bars: int,
    pivot_highs: list[tuple[int, float]],
    pivot_lows: list[tuple[int, float]],
    tolerance: int = 3,
    last_direction: float = 0.0,
) -> tuple[float, float]:
    """Count how many Gann time cycles converge on the current bar.

    Projects key cycle lengths (30, 60, 90, 120, 144, 180, 270, 360 cal
    days ≈ bars * 252/365) from each detected pivot.  When multiple
    cycles cluster near the current bar a reversal is more likely.
    """
    gann_cycles_bars = [
        int(round(c * 252 / 365)) for c in [30, 60, 90, 120, 144, 180, 270, 360]
    ]

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
        return 0.0, 0.0

    convergence = hits / max(total_checks, 1)
    # More convergence -> higher chance of turning point; direction = opposite of last move
    score = convergence * (-last_direction) if last_direction != 0 else 0.0
    score = float(np.clip(score, -1.0, 1.0))
    confidence = min(convergence * 3.0, 1.0)
    return score, float(confidence)


def _swing_indicator(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 2,
) -> tuple[float, float]:
    """Gann swing chart: consecutive higher-highs/higher-lows = upswing.

    Returns (score in [-1,1], confidence in [0,1]).
    """
    n = len(highs)
    if n < period + 1:
        return 0.0, 0.0

    consecutive_up = 0
    consecutive_down = 0

    for i in range(n - 1, max(n - 1 - 20, 0), -1):
        if i < 1:
            break
        hh = highs[i] > highs[i - 1]
        hl = lows[i] > lows[i - 1]
        lh = highs[i] < highs[i - 1]
        ll = lows[i] < lows[i - 1]

        if hh and hl:
            if consecutive_down > 0:
                break
            consecutive_up += 1
        elif lh and ll:
            if consecutive_up > 0:
                break
            consecutive_down += 1
        else:
            break

    if consecutive_up >= period:
        score = min(consecutive_up / 10.0, 1.0)
        confidence = min(consecutive_up / 8.0, 1.0)
        return float(score), float(confidence)
    elif consecutive_down >= period:
        score = -min(consecutive_down / 10.0, 1.0)
        confidence = min(consecutive_down / 8.0, 1.0)
        return float(score), float(confidence)

    return 0.0, 0.0


def _trend_strength(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 14,
) -> float:
    """Compute a 0-1 trend-strength ratio (directional movement based).

    Uses the ratio of net directional movement to total directional
    movement over the lookback window — conceptually similar to ADX but
    cheaper to compute.  Returns 0.0 for no trend, 1.0 for a perfectly
    directional market.
    """
    n = len(closes)
    if n < lookback + 1:
        return 0.5  # insufficient data -> neutral assumption

    plus_dm_sum = 0.0
    minus_dm_sum = 0.0

    for i in range(n - lookback, n):
        up_move = float(highs[i] - highs[i - 1])
        down_move = float(lows[i - 1] - lows[i])

        if up_move > down_move and up_move > 0:
            plus_dm_sum += up_move
        if down_move > up_move and down_move > 0:
            minus_dm_sum += down_move

    total_dm = plus_dm_sum + minus_dm_sum
    if total_dm == 0:
        return 0.0

    return abs(plus_dm_sum - minus_dm_sum) / total_dm


def _retracement_levels(
    current_price: float,
    range_high: float,
    range_low: float,
) -> tuple[float, float]:
    """Gann 8-part retracement: score based on position within the grid.

    The 4/8 (50 %) level is the key pivot.  Above 4/8 = bearish pressure,
    below = bullish opportunity.  Extremes (near 0/8 or 8/8) carry less
    conviction because they may indicate exhaustion.
    """
    span = range_high - range_low
    if span <= 0:
        return 0.0, 0.0

    position = (current_price - range_low) / span  # 0.0 to 1.0
    position = float(np.clip(position, 0.0, 1.0))

    # Center around 0.5 (the 4/8 pivot); invert so below midpoint = bullish
    raw = -(position - 0.5) * 2.0  # [-1, +1]; +1 at 0/8, -1 at 8/8

    # Confidence peaks near 3/8 and 5/8 (the action zones) and fades at extremes
    dist_from_mid = abs(position - 0.5)
    confidence = float(np.clip(1.0 - abs(dist_from_mid - 0.125) * 4.0, 0.2, 1.0))

    return float(np.clip(raw, -1.0, 1.0)), confidence


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

@register("gann")
class GannStrategy(BaseStrategy):
    """Composite Gann strategy combining angles, Square of Nine, time
    cycles, swing indicator, and retracement levels."""

    def __init__(self, params: dict | None = None):
        super().__init__("gann", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        pivot_lookback: int = self.params.get("pivot_lookback", 60)
        pivot_order: int = self.params.get("pivot_order", 5)
        cycle_tolerance: int = self.params.get("cycle_tolerance_days", 3)
        swing_period: int = self.params.get("swing_period", 2)
        retracement_lookback: int = self.params.get("retracement_lookback", 120)
        trend_filter_lookback: int = self.params.get("trend_filter_lookback", 14)
        trend_filter_threshold: float = self.params.get("trend_filter_threshold", 0.15)
        min_confidence: float = self.params.get("min_confidence", 0.05)
        default_sub_weights = {
            "angles": 0.25,
            "sq9": 0.15,
            "cycles": 0.15,
            "swing": 0.25,
            "retracement": 0.20,
        }
        sub_weights: dict[str, float] = self.params.get("sub_weights", default_sub_weights)

        universe = pit_view.universe
        if not universe:
            return []

        needed_days = max(pivot_lookback, retracement_lookback) + 30
        prices_df = pit_view.prices(symbols=universe, lookback_days=needed_days)
        if prices_df.empty:
            return []

        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        raw_scores: dict[str, float] = {}
        raw_confs: dict[str, float] = {}
        meta_all: dict[str, dict] = {}

        for symbol in universe:
            sym_df = (
                prices_df[prices_df["symbol"] == symbol]
                .sort_values("date")
                .reset_index(drop=True)
            )
            if len(sym_df) < pivot_lookback:
                continue

            highs = sym_df["high"].values if "high" in sym_df.columns else sym_df["adj_close"].values
            lows = sym_df["low"].values if "low" in sym_df.columns else sym_df["adj_close"].values
            closes = sym_df["adj_close"].values if "adj_close" in sym_df.columns else sym_df["close"].values

            current_price = float(closes[-1])
            current_idx = len(closes) - 1
            if current_price <= 0:
                continue

            pivot_highs, pivot_lows = _detect_pivots(highs, lows, order=pivot_order)

            # ATR for normalization
            true_ranges = []
            for i in range(1, min(15, len(closes))):
                tr = max(
                    float(highs[-i] - lows[-i]),
                    abs(float(highs[-i] - closes[-i - 1])),
                    abs(float(lows[-i] - closes[-i - 1])),
                )
                true_ranges.append(tr)
            atr = float(np.mean(true_ranges)) if true_ranges else 1.0
            if atr <= 0:
                atr = 1.0

            # Recent direction for sq9 and cycles
            if len(closes) >= 6:
                ret_5d = (closes[-1] / closes[-6]) - 1.0
                direction = float(np.sign(ret_5d))
            else:
                direction = 0.0

            # --- 1. Gann Angles ---
            angle_unit = atr  # 1 ATR per bar as the 45-degree unit
            a_score, a_conf = _gann_angles(
                current_price, current_idx, pivot_highs, pivot_lows, angle_unit
            )

            # --- 2. Square of Nine ---
            s_score, s_conf = _sq9_score(current_price, atr, direction)

            # --- 3. Time Cycles ---
            c_score, c_conf = _time_cycles(
                current_idx, len(closes), pivot_highs, pivot_lows,
                tolerance=cycle_tolerance, last_direction=direction,
            )

            # --- 4. Swing Indicator ---
            sw_score, sw_conf = _swing_indicator(highs, lows, period=swing_period)

            # --- 5. Retracement Levels ---
            retrace_start = max(0, len(highs) - retracement_lookback)
            range_high = float(np.max(highs[retrace_start:]))
            range_low = float(np.min(lows[retrace_start:]))
            r_score, r_conf = _retracement_levels(current_price, range_high, range_low)

            # --- Trend-strength filter (risk mitigation) ---
            trend_str = _trend_strength(
                highs, lows, closes, lookback=trend_filter_lookback
            )
            # Dampen composite when trend is weak (choppy market)
            trend_dampener = 1.0
            if trend_str < trend_filter_threshold:
                trend_dampener = trend_str / max(trend_filter_threshold, 1e-9)

            # --- Composite ---
            components = {
                "angles": (a_score, a_conf),
                "sq9": (s_score, s_conf),
                "cycles": (c_score, c_conf),
                "swing": (sw_score, sw_conf),
                "retracement": (r_score, r_conf),
            }

            total_w = 0.0
            composite = 0.0
            conf_weighted = 0.0
            for key, (sc, co) in components.items():
                w = sub_weights.get(key, 0.0)
                composite += sc * w
                conf_weighted += co * w
                total_w += w

            if total_w > 0:
                composite /= total_w
                conf_weighted /= total_w
            else:
                composite = 0.0
                conf_weighted = 0.0

            # Apply trend dampener to score and confidence
            composite *= trend_dampener
            conf_weighted *= trend_dampener

            # Confidence floor: skip symbols with negligible conviction
            if conf_weighted < min_confidence:
                continue

            raw_scores[symbol] = composite
            raw_confs[symbol] = conf_weighted
            meta_all[symbol] = {
                "angle_score": a_score,
                "sq9_score": s_score,
                "cycle_score": c_score,
                "swing_score": sw_score,
                "retracement_score": r_score,
                "atr": atr,
                "direction_5d": direction,
                "trend_strength": trend_str,
                "trend_dampener": trend_dampener,
            }

        if not raw_scores:
            return []

        signals: list[Signal] = []
        for symbol in sorted(raw_scores):
            sc = float(np.clip(raw_scores[symbol], -1.0, 1.0))
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="gann",
                    score=sc,
                    confidence=float(np.clip(raw_confs.get(symbol, 0.0), 0.0, 1.0)),
                    horizon="10d",
                    asof=pit_view.asof,
                    meta=meta_all.get(symbol, {}),
                )
            )
        return signals
