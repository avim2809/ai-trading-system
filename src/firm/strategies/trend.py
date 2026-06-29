"""Time-series trend-following strategy.

Financial intuition:
    When an asset's price sits above its long-term moving average it is in
    an uptrend; below signals a downtrend.  Scaling the binary signal by
    inverse volatility allocates more risk budget to lower-vol (higher
    Sharpe) trends, following the managed-momentum literature (Barroso &
    Santa-Clara, 2015).

Data inputs:
    Adjusted close prices from PitView.prices() with lookback sufficient for
    the slow moving-average window plus a volatility estimation window.

Signal logic:
    1. Compute fast MA (default 50 d) and slow MA (default 200 d) per symbol.
    2. Direction = +1 if fast MA > slow MA, else -1.
    3. Realized volatility = annualized std of daily returns over vol_lookback.
    4. Signal score = direction / volatility (inverse-vol scaled), then
       normalized across the universe to roughly [-1, 1].

Portfolio construction approach:
    Each signal encodes direction and conviction (inverse vol).  Consumers
    size positions proportionally to the score.

Risk notes:
    Trend strategies suffer during choppy, range-bound markets.  Works best
    when combined with a mean-reversion overlay for diversification.
"""

from __future__ import annotations

import numpy as np

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register


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

        raw_score = direction / vol
        raw_score = raw_score.dropna()

        if len(raw_score) < 2:
            return []

        mean = raw_score.mean()
        std = raw_score.std()
        if std == 0 or np.isnan(std):
            z_scores = raw_score * 0.0
        else:
            z_scores = ((raw_score - mean) / std).clip(-3, 3)

        signals: list[Signal] = []
        for symbol in z_scores.index:
            z = float(z_scores[symbol])
            d = float(direction.get(symbol, 0))
            v = float(vol.get(symbol, np.nan))
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="trend",
                    score=z,
                    confidence=min(abs(z) / 3.0, 1.0),
                    horizon="21d",
                    asof=pit_view.asof,
                    meta={
                        "direction": d,
                        "annualized_vol": v,
                        "fast_ma": float(fast_ma.get(symbol, np.nan)),
                        "slow_ma": float(slow_ma.get(symbol, np.nan)),
                    },
                )
            )
        return signals
