"""Analyst rating-consensus strategy (FMP backtest / Investing.com Pro live).

Financial intuition:
    Analyst rating consensus (strong buy/buy/hold/sell/strong sell counts)
    aggregates professional analysts' views on a stock. Both the *level*
    (net bullish tilt) and its *trend* (analysts turning more/less bullish
    over time) carry information — this mirrors the well-documented
    analyst-recommendation-revision literature (e.g. Womack 1996).

Data inputs:
    PitView.estimates(): ANALYST_RATINGS_COLS (date, symbol, strong_buy,
    buy, hold, sell, strong_sell). Backtests source this from FMP's
    /stable/grades-historical (a genuine ~8-year monthly history — verified
    live 2026-07-31; FMP's price-target-consensus/analyst-estimates lack
    real point-in-time history under this plan tier, see
    firm.data.providers.fmp.FMPProvider.get_analyst_ratings). Live can
    instead be fed by Investing.com Pro's own "Analyst Ratings" column
    (Phase 2b, firm.data.investing) — same schema either way, so this
    strategy is unchanged by which one is wired into the "estimates"
    provider key.

Signal logic:
    1. net_score = (2*strong_buy + buy - 2*strong_sell - sell) / total
       analysts covering the name — naturally bounded in [-2, 2]; scaling by
       total avoids treating a mild tilt from 40 analysts the same as an
       identical raw tilt from 3.
    2. level = latest net_score; delta = change over the lookback window
       (the rating-revision trend).
    3. Emit raw combined = level + delta_weight * delta — analysts z-score
       cross-sectionally once (see momentum.py's identical convention);
       this strategy does not z-score its own output.

Portfolio construction approach:
    Long the most net-bullish / improving-consensus names, short the most
    net-bearish / deteriorating ones.

Risk notes:
    Monthly-cadence data — a slow-moving tilt, not a timing signal. The
    live-fed (Investing.com) variant only ever sees "today's" consensus,
    not genuine history, unlike the FMP-backed backtest — see
    docs/investing_pro_integration.md for how that gap is handled
    (shadow-mode / proxy-backtest discipline before any live weight).
"""

from __future__ import annotations

import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

_RATING_COLS = ["strong_buy", "buy", "hold", "sell", "strong_sell"]


@register("investing_analyst_ratings")
class InvestingAnalystRatingsStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("investing_analyst_ratings", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        lookback_days: int = self.params.get("lookback_days", 365)
        delta_weight: float = self.params.get("delta_weight", 1.0)

        universe = pit_view.universe
        if not universe:
            return []

        est_df = pit_view.estimates(symbols=universe, lookback_days=lookback_days)
        if est_df.empty or not set(_RATING_COLS).issubset(est_df.columns):
            return []

        est_df = est_df.copy()
        est_df["date"] = pd.to_datetime(est_df["date"])
        counts = est_df[_RATING_COLS].fillna(0)
        total = counts.sum(axis=1)

        # A name with zero analyst coverage that month can't produce a
        # meaningful consensus tilt.
        covered = total > 0
        est_df, counts, total = est_df.loc[covered], counts.loc[covered], total.loc[covered]
        if est_df.empty:
            return []

        est_df["net_score"] = (
            2 * counts["strong_buy"] + counts["buy"]
            - 2 * counts["strong_sell"] - counts["sell"]
        ) / total
        est_df["total_analysts"] = total

        pivot = est_df.pivot_table(
            index="date", columns="symbol", values="net_score", aggfunc="last"
        ).sort_index()
        if pivot.empty:
            return []
        coverage_pivot = est_df.pivot_table(
            index="date", columns="symbol", values="total_analysts", aggfunc="last"
        ).sort_index()

        level = pivot.iloc[-1]
        latest_coverage = coverage_pivot.iloc[-1]

        # "old" must be ~lookback_days before asof, not just the first row
        # in the fetched buffer — mirrors sentiment.py's identical fix for
        # the same drift-with-data-sparsity failure mode.
        target_old_date = pd.Timestamp(pit_view.asof) - pd.Timedelta(days=lookback_days)
        older_rows = pivot.index[pivot.index <= target_old_date]
        if len(older_rows) > 0:
            delta = level - pivot.loc[older_rows[-1]]
        else:
            delta = pd.Series(0.0, index=level.index)

        combined = (level + delta_weight * delta).dropna()
        if len(combined) < 3:
            return []

        signals: list[Signal] = []
        for symbol in combined.index:
            raw = float(combined[symbol])
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="investing_analyst_ratings",
                    score=raw,
                    # net_score (and hence combined, absent an extreme delta)
                    # is naturally bounded in [-2, 2] — scale confidence to
                    # that range rather than an arbitrary constant.
                    confidence=min(abs(raw) / 2.0, 1.0),
                    horizon="60d",
                    asof=pit_view.asof,
                    meta={
                        "rating_level": float(level.get(symbol, float("nan"))),
                        "rating_delta": float(delta.get(symbol, float("nan"))),
                        "num_analysts": float(latest_coverage.get(symbol, float("nan"))),
                    },
                )
            )
        return signals
