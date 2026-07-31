"""Danelfin AI-score strategy — a genuine paid REST API, not a scraper.

Financial intuition:
    Danelfin's AI Score (1-10, ML-driven composite of price/fundamental/
    technical/sentiment/news signals) is their own published headline
    metric: per their marketing (unverified independently — treat as a
    claim, not evidence), US stocks scoring 10/10 have historically
    outperformed the market by ~+21% (3-month annualized alpha) while
    1/10-scored stocks underperformed by ~-33%. This strategy tests that
    claim directly against this project's own walk-forward harness rather
    than taking it at face value.

Data inputs:
    PitView.ai_scores(): AI_SCORE_COLS (date, symbol, ai_score,
    fundamental_score, technical_score, sentiment_score, low_risk_score).
    Backed by Danelfin's ``/ranking`` endpoint (firm.data.providers.danelfin
    .DanelfinProvider.get_ai_scores) — genuinely historical (~2016-present,
    verified live), unlike this project's other Investing.com-adjacent
    endpoints, so this strategy can be honestly backtested rather than only
    run in shadow mode.

Signal logic:
    1. level = latest ai_score, centered at the scale's midpoint (5.5) so
       a "5" (Danelfin's own neutral/hold boundary per their /v3/trading-
       parameters signal mapping) contributes ~zero rather than a large
       positive raw value.
    2. delta = change in ai_score over the lookback window (is Danelfin
       getting more/less bullish on this name), weighted at 0.5 (not 1.0)
       so combined stays within [-9, 9] even in the worst case (level
       bounded [-4.5, 4.5], delta bounded [-9, 9]) — comfortably inside
       this project's raw-score sanity convention.
    3. Emit raw combined = level + delta_weight * delta — analysts
       z-score cross-sectionally once (matches every other strategy's
       convention; see momentum.py).

Portfolio construction approach:
    Long the highest-AI-Score / improving names, short the lowest /
    deteriorating ones.

Risk notes:
    A single third-party ML score with no visibility into its own training
    data or decay — a black box, not a re-derivable factor. Treat any
    positive backtest result with real skepticism (small-N vendor claims
    are a classic overfitting trap) and see docs/investing_pro_integration.md
    for how this project's promotion-gate discipline handles that.
"""

from __future__ import annotations

import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

_MIDPOINT = 5.5  # center of Danelfin's 1-10 scale


@register("danelfin_ai_score")
class DanelfinAiScoreStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("danelfin_ai_score", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        lookback_days: int = self.params.get("lookback_days", 30)
        # level is bounded [-4.5, 4.5]; a raw day-to-day delta over the
        # 1-10 scale is bounded [-9, 9] in the worst case (score swinging
        # from 1 to 10). 0.5 keeps combined's worst case at 4.5+4.5=9.0,
        # comfortably inside this project's [-10, 10] raw-score sanity
        # convention even under adversarial/noisy data — see
        # investing_analyst_ratings.py's identical by-construction bounding.
        delta_weight: float = self.params.get("delta_weight", 0.5)

        universe = pit_view.universe
        if not universe:
            return []

        scores_df = pit_view.ai_scores(symbols=universe, lookback_days=lookback_days + 5)
        if scores_df.empty or "ai_score" not in scores_df.columns:
            return []

        scores_df = scores_df.copy()
        scores_df["date"] = pd.to_datetime(scores_df["date"])
        # aiscore is nominally daily but can repeat a stale value across a
        # short gap — last-value-per-day per symbol, matching sentiment.py's
        # groupby/pivot pattern.
        pivot = scores_df.pivot_table(
            index="date", columns="symbol", values="ai_score", aggfunc="last"
        ).sort_index()
        if pivot.empty:
            return []

        level = pivot.iloc[-1] - _MIDPOINT

        # "old" must be ~lookback_days before asof, not just the first row
        # of the fetched buffer — mirrors sentiment.py's/
        # investing_analyst_ratings.py's identical fix for the same
        # drift-with-data-sparsity failure mode.
        target_old_date = pd.Timestamp(pit_view.asof) - pd.Timedelta(days=lookback_days)
        older_rows = pivot.index[pivot.index <= target_old_date]
        if len(older_rows) > 0:
            delta = pivot.iloc[-1] - pivot.loc[older_rows[-1]]
        else:
            delta = pd.Series(0.0, index=level.index)

        combined = (level + delta_weight * delta).dropna()
        if len(combined) < 3:
            return []

        latest_row_by_symbol = (
            scores_df.sort_values("date").groupby("symbol").last()
        )

        signals: list[Signal] = []
        for symbol in combined.index:
            raw = float(combined[symbol])
            extra = (
                latest_row_by_symbol.loc[symbol]
                if symbol in latest_row_by_symbol.index else None
            )
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="danelfin_ai_score",
                    score=raw,
                    # combined's worst-case magnitude is 9.0 (see the
                    # delta_weight comment above) — scale confidence to that.
                    confidence=min(abs(raw) / 9.0, 1.0),
                    horizon="60d",
                    asof=pit_view.asof,
                    meta={
                        "ai_score_level": float(level.get(symbol, float("nan"))) + _MIDPOINT,
                        "ai_score_delta": float(delta.get(symbol, float("nan"))),
                        "fundamental_score": float(extra["fundamental_score"])
                        if extra is not None and pd.notna(extra.get("fundamental_score")) else float("nan"),
                        "technical_score": float(extra["technical_score"])
                        if extra is not None and pd.notna(extra.get("technical_score")) else float("nan"),
                        "sentiment_score": float(extra["sentiment_score"])
                        if extra is not None and pd.notna(extra.get("sentiment_score")) else float("nan"),
                        "low_risk_score": float(extra["low_risk_score"])
                        if extra is not None and pd.notna(extra.get("low_risk_score")) else float("nan"),
                    },
                )
            )
        return signals
