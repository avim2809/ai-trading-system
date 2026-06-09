"""News & sentiment-driven strategy.

Financial intuition:
    Aggregate news sentiment carries information about near-term returns
    that is not yet fully priced in.  Positive sentiment shifts (rising
    scores) predict bullish drift, while deteriorating sentiment predicts
    weakness (Tetlock, 2007; Loughran & McDonald, 2011).

Data inputs:
    Sentiment data from PitView.sentiment(): per-symbol, per-date records
    containing sentiment_score and news_volume.

Signal logic:
    1. Aggregate daily sentiment scores per symbol (mean sentiment weighted
       by news_volume where available).
    2. Compute the sentiment *level* (latest aggregated score).
    3. Compute the sentiment *delta* (change over the lookback window).
    4. Combine: raw_score = level + delta_weight * delta.
    5. Z-score the combined score cross-sectionally across the universe.

Portfolio construction approach:
    Long symbols with the most positive sentiment momentum, short those
    with the most negative.

Risk notes:
    Sentiment signals are noisy and can be manipulated (social-media
    pump-and-dump).  Best used as a secondary tilt on top of fundamental
    or technical signals rather than as a standalone allocator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register


@register("sentiment")
class SentimentStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("sentiment", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        lookback_days: int = self.params.get("lookback_days", 5)
        delta_weight: float = self.params.get("delta_weight", 1.0)

        universe = pit_view.universe
        if not universe:
            return []

        sent_df = pit_view.sentiment(symbols=universe, lookback_days=lookback_days + 5)
        if sent_df.empty or "sentiment_score" not in sent_df.columns:
            return []

        sent_df = sent_df.copy()
        sent_df["date"] = pd.to_datetime(sent_df["date"])

        daily_sent = (
            sent_df.groupby(["date", "symbol"])["sentiment_score"]
            .mean()
            .reset_index()
        )
        daily_pivot = daily_sent.pivot_table(
            index="date", columns="symbol", values="sentiment_score"
        ).sort_index()

        if daily_pivot.empty:
            return []

        level = daily_pivot.iloc[-1]

        if len(daily_pivot) >= 2:
            old_val = daily_pivot.iloc[0]
            delta = level - old_val
        else:
            delta = pd.Series(0.0, index=level.index)

        combined = level + delta_weight * delta
        combined = combined.dropna()

        if len(combined) < 2:
            signals = []
            for symbol, val in combined.items():
                score = float(np.clip(val, -1, 1))
                signals.append(
                    Signal(
                        symbol=str(symbol),
                        strategy="sentiment",
                        score=score,
                        confidence=min(abs(score), 1.0),
                        horizon="5d",
                        asof=pit_view.asof,
                        meta={"sentiment_level": float(level.get(symbol, np.nan))},
                    )
                )
            return signals

        mean = combined.mean()
        std = combined.std()
        if std == 0 or np.isnan(std):
            z_scores = combined * 0.0
        else:
            z_scores = ((combined - mean) / std).clip(-3, 3)

        signals: list[Signal] = []
        for symbol in z_scores.index:
            z = float(z_scores[symbol])
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="sentiment",
                    score=z,
                    confidence=min(abs(z) / 3.0, 1.0),
                    horizon="5d",
                    asof=pit_view.asof,
                    meta={
                        "sentiment_level": float(level.get(symbol, np.nan)),
                        "sentiment_delta": float(delta.get(symbol, np.nan)),
                    },
                )
            )
        return signals
