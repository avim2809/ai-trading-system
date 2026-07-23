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

        # Volume-weighted mean, as the docstring promises: a symbol with one
        # wildly positive article shouldn't score identically to one with
        # 50 moderately positive articles the same day. Falls back to a
        # plain mean if news_volume isn't present (e.g. a minimal fixture).
        if "news_volume" in sent_df.columns:
            vol = pd.to_numeric(sent_df["news_volume"], errors="coerce").fillna(1.0).clip(lower=1.0)
            sent_df = sent_df.assign(_vol=vol, _weighted=sent_df["sentiment_score"] * vol)
            grouped = sent_df.groupby(["date", "symbol"])
            daily_sent = (
                (grouped["_weighted"].sum() / grouped["_vol"].sum())
                .reset_index(name="sentiment_score")
            )
        else:
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

        # "old" must be ~lookback_days before asof, not just whichever row
        # happens to be first in the fetched buffer (lookback_days + 5) —
        # the buffer's actual span varies with data sparsity around
        # weekends/holidays, so the delta window used to silently drift
        # away from the lookback_days parameter it's supposed to measure.
        target_old_date = pd.Timestamp(pit_view.asof) - pd.Timedelta(days=lookback_days)
        older_rows = daily_pivot.index[daily_pivot.index <= target_old_date]
        if len(older_rows) > 0:
            old_val = daily_pivot.loc[older_rows[-1]]
            delta = level - old_val
        else:
            delta = pd.Series(0.0, index=level.index)

        combined = level + delta_weight * delta
        combined = combined.dropna()

        # Fewer than 3 symbols can't be meaningfully z-scored cross-
        # sectionally (matches momentum.py/mean_reversion.py's own "< 3"
        # convention) — emit nothing rather than falling back to a
        # differently-scaled raw score. The strategy used to mix two
        # incompatible scales (raw [-1,1] here vs. a z-score clipped to
        # [-3,3] below) depending on how many symbols happened to have
        # data that day, breaking comparability for anything downstream
        # that combines/normalizes signals assuming one stable scale.
        if len(combined) < 3:
            return []

        signals: list[Signal] = []
        for symbol in combined.index:
            raw = float(combined[symbol])
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="sentiment",
                    score=raw,
                    confidence=min(abs(raw), 1.0),
                    horizon="5d",
                    asof=pit_view.asof,
                    meta={
                        "sentiment_level": float(level.get(symbol, np.nan)),
                        "sentiment_delta": float(delta.get(symbol, np.nan)),
                    },
                )
            )
        return signals
