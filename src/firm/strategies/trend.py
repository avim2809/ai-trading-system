"""Time-series trend-following strategy.

Financial intuition:
    When an asset's price sits above its long-term moving average it is in
    an uptrend; below signals a downtrend.  The score uses *continuous*
    crossover strength — how far the fast MA sits above/below the slow MA —
    so cross-sectional ranking reflects trend conviction, not a binary sign.

    Volatility scaling belongs in portfolio construction (e.g. risk-agent vol
    targeting), not in the cross-sectional alpha score: embedding ``1/vol``
    in the signal made rankings track inverse vol and produced negative IC.

Data inputs:
    Adjusted close prices from PitView.prices() with lookback sufficient for
    the slow moving-average window plus a volatility estimation window.

Signal logic:
    1. Compute fast MA (default 50 d) and slow MA (default 200 d) per symbol.
    2. Score = (fast MA − slow MA) / slow MA  (signed trend strength).
    3. Direction and annualized vol are kept in ``meta`` for diagnostics/sizing.

Portfolio construction approach:
    Each signal encodes direction and conviction via crossover strength.
    Consumers z-score across the universe in the analyst layer.

Risk notes:
    Trend strategies suffer during choppy, range-bound markets.  Works best
    when combined with a mean-reversion overlay for diversification.
"""

from __future__ import annotations

import logging

import numpy as np

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

log = logging.getLogger(__name__)

# MA spread at which confidence saturates (15% fast-above-slow ≈ strong trend).
_CONFIDENCE_SCALE = 0.15


@register("trend")
class TrendStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("trend", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        fast_window: int = self.params.get("fast_window", 50)
        slow_window: int = self.params.get("slow_window", 200)
        vol_lookback: int = self.params.get("vol_lookback", 60)

        universe = pit_view.universe
        if not universe:
            return []

        needed_days = slow_window + vol_lookback + 10
        prices_df = pit_view.prices(symbols=universe, lookback_days=needed_days)
        if prices_df.empty:
            return []

        pivot = (
            prices_df.pivot_table(index="date", columns="symbol", values="adj_close")
            .sort_index()
        )
        if len(pivot) < slow_window:
            return []

        fast_ma = pivot.rolling(window=fast_window, min_periods=fast_window).mean().iloc[-1]
        slow_ma = pivot.rolling(window=slow_window, min_periods=slow_window).mean().iloc[-1]
        direction = np.sign(fast_ma - slow_ma)

        daily_ret = pivot.pct_change()
        vol = daily_ret.iloc[-vol_lookback:].std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)

        # Continuous crossover strength — do not divide by vol here (hurts IC).
        strength = (fast_ma - slow_ma) / slow_ma
        raw_score = strength.dropna()

        if len(raw_score) < 2:
            return []

        log.debug(
            "trend: %d symbols, score range [%.4f, %.4f]",
            len(raw_score), float(raw_score.min()), float(raw_score.max()),
        )

        signals: list[Signal] = []
        for symbol in raw_score.index:
            raw = float(raw_score[symbol])
            d = float(direction.get(symbol, 0))
            v = float(vol.get(symbol, np.nan))
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="trend",
                    score=raw,
                    confidence=min(abs(raw) / _CONFIDENCE_SCALE, 1.0),
                    horizon="21d",
                    asof=pit_view.asof,
                    meta={
                        "direction": d,
                        "annualized_vol": v,
                        "strength": raw,
                        "fast_ma": float(fast_ma.get(symbol, np.nan)),
                        "slow_ma": float(slow_ma.get(symbol, np.nan)),
                    },
                )
            )
        return signals
