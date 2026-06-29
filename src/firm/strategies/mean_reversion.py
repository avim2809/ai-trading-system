"""Short-term mean-reversion (reversal) strategy.

Financial intuition:
    Over very short horizons (1-5 days) stocks that have moved sharply tend
    to partially reverse, driven by liquidity provision, over-reaction, and
    bid-ask bounce effects (Lehmann, 1990; Lo & MacKinlay, 1990).

Data inputs:
    Adjusted close prices from PitView.prices() with a short lookback
    (~20 trading days is enough for computing 1-5 day returns and z-scores).

Signal logic:
    1. Compute trailing N-day return for each symbol (default N=5).
    2. Cross-sectionally z-score these returns.
    3. Negate the z-score: recent losers get positive scores (buy),
       recent winners get negative scores (sell).
    4. Cap extreme z-scores to ±zscore_cap.

Portfolio construction approach:
    Long the biggest recent losers, short the biggest recent winners.
    Market-neutral by construction when the z-score distribution is
    symmetric.

Risk notes:
    Reversals can fail spectacularly during momentum-driven crashes or
    when a stock gaps on genuine fundamental news.  Position sizing should
    be modest and a stop-loss is advisable.
"""

from __future__ import annotations

import numpy as np

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register


@register("mean_reversion")
class MeanReversionStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("mean_reversion", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        lookback_days: int = self.params.get("lookback_days", 5)
        zscore_cap: float = self.params.get("zscore_cap", 3.0)

        universe = pit_view.universe
        if not universe:
            return []

        prices_df = pit_view.prices(symbols=universe, lookback_days=lookback_days + 10)
        if prices_df.empty:
            return []

        pivot = (
            prices_df.pivot_table(index="date", columns="symbol", values="adj_close")
            .sort_index()
        )
        if len(pivot) < lookback_days + 1:
            return []

        recent_ret = (pivot.iloc[-1] / pivot.iloc[-(lookback_days + 1)]) - 1.0
        recent_ret = recent_ret.dropna()

        if len(recent_ret) < 3:
            return []

        mean = recent_ret.mean()
        std = recent_ret.std()
        if std == 0 or np.isnan(std):
            return []

        z_scores = (recent_ret - mean) / std
        reversal_scores = -z_scores
        reversal_scores = reversal_scores.clip(-zscore_cap, zscore_cap)

        signals: list[Signal] = []
        for symbol, score in reversal_scores.items():
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="mean_reversion",
                    score=float(score),
                    confidence=min(abs(float(score)) / zscore_cap, 1.0),
                    horizon="5d",
                    asof=pit_view.asof,
                    meta={
                        "raw_return": float(recent_ret.get(symbol, np.nan)),
                        "z_score_before_negate": float(z_scores.get(symbol, np.nan)),
                    },
                )
            )
        return signals
