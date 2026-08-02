"""Danelfin market-percentile strategy — rank vs. the whole market, not just
this project's own fixed universe.

Financial intuition:
    `danelfin_ai_score` already reads each universe symbol's ai_score, but
    the downstream signal-combination/analyst layer z-scores it only across
    this project's own ~25-name fixed universe. That answers "is this a top
    score relative to an arbitrarily chosen small set", not "is this
    actually a top score relative to the whole market" — a real, different
    question. This strategy answers the second one directly: each universe
    symbol's ai_score percentile rank within a broad cross-sectional
    population (many symbols across many sectors, not just the universe).

Data inputs:
    PitView.market_percentile(): MARKET_PERCENTILE_COLS (date, symbol,
    sector, ai_score) — a population SNAPSHOT for one date, backed by
    firm.data.danelfin_market_percentile.fetch_market_percentile_pool
    (bulk historical /ranking mode, genuinely historical — unlike
    live_signals/best_stocks). Real cost: one full snapshot costs ~66+
    Danelfin API calls (11 sectors x ~6 low_risk values, +pagination).

Status as of 2026-08-02 — built and tested, NOT YET LIVE-WIRED:
    Nothing populates PitView.market_percentile() with real data yet in
    either backtests or live trading (see firm.live.data_feed's
    LivePitViewAdapter.market_percentile and
    firm.backtest.firm_strategy's PitViewAdapter.market_percentile — both
    correctly wired to read from the PIT store, but nothing writes to that
    store's market_percentile field). This strategy therefore emits zero
    signals in current backtests and live cycles alike — a graceful no-op,
    the same posture as a strategy with no data available yet. Wiring a
    real data source (a live per-cycle fetch, or a cached historical
    dataset for backtesting) is a real cost/time decision — not made
    unilaterally here — see docs/investing_pro_integration.md for the
    open question and estimated cost.

Signal logic:
    1. Compute percentile = ai_score.rank(pct=True) across the ENTIRE
       population snapshot (0 = lowest ai_score in the population, 1 =
       highest) — a genuine cross-sectional percentile, not scaled relative
       to the small fixed universe.
    2. For each universe symbol present in the population, map percentile
       to a raw score centered at 0 (percentile 0.5 -> raw 0), scaled so
       percentile 0 and 1 map to +/- _SCALE (comfortably inside the
       project's [-10, 10] raw-score sanity convention).
    3. Confidence scales with population size (a percentile computed
       against a tiny/degenerate population is less trustworthy than one
       against hundreds of symbols) and is bounded [0, 1].

Portfolio construction approach:
    Long AND short — unlike danelfin_best_stocks_signal (long-only, a
    curated buy-list has no natural "sell" reading), a LOW population
    percentile here is a genuine bearish cross-sectional signal: this name
    scores worse than most of the market on Danelfin's own composite,
    not merely "outside a curated top-N list."

Risk notes:
    Same black-box-vendor caveat as every Danelfin strategy in this project.
    A minimum population size guard avoids a percentile computed against
    too few symbols to be meaningful.
"""

from __future__ import annotations

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

_SCALE = 9.0  # percentile 0/1 -> raw -9/+9, comfortably inside [-10, 10]
_MIN_POPULATION = 30  # below this, a percentile rank isn't meaningful
_MIN_CONFIDENCE_POPULATION = 200  # population size at which confidence saturates near 1.0


@register("danelfin_market_percentile")
class DanelfinMarketPercentileStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("danelfin_market_percentile", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        universe = pit_view.universe
        if not universe:
            return []

        pool = pit_view.market_percentile()
        if pool.empty or "ai_score" not in pool.columns:
            return []

        pool = pool.dropna(subset=["ai_score"])
        if len(pool) < _MIN_POPULATION:
            return []

        population_size = len(pool)
        pool = pool.copy()
        pool["percentile"] = pool["ai_score"].rank(pct=True)

        universe_rows = pool[pool["symbol"].isin(universe)]
        if universe_rows.empty:
            return []

        confidence = min(population_size / _MIN_CONFIDENCE_POPULATION, 1.0)

        signals: list[Signal] = []
        for _, row in universe_rows.iterrows():
            percentile = float(row["percentile"])
            raw = (percentile - 0.5) * 2.0 * _SCALE
            raw = max(-10.0, min(10.0, raw))

            signals.append(
                Signal(
                    symbol=str(row["symbol"]),
                    strategy="danelfin_market_percentile",
                    score=raw,
                    confidence=confidence,
                    horizon="60d",
                    asof=pit_view.asof,
                    meta={
                        "percentile": percentile,
                        "ai_score": float(row["ai_score"]),
                        "sector": row.get("sector"),
                        "population_size": population_size,
                    },
                )
            )
        return signals
