"""Danelfin best-stocks-signal strategy — real /v3/beststocks Top-25 membership.

Financial intuition:
    Danelfin's ``/v3/beststocks`` is their own curated, real Top-25 "Best
    Stocks" list (rank 1 = highest conviction) — not a reconstruction of
    it (see docs/danelfin_best_stocks_arm.md's "Important caveat" section
    for why this project's own sector-ranking reimplementation only
    matched their real output ~25-30% and should not be trusted as a
    proxy). This strategy reads the real list directly: a universe symbol
    appearing in it is treated as a strong, high-conviction buy signal.

Data inputs:
    PitView.best_stocks(): BEST_STOCKS_COLS, backed by
    firm.data.providers.danelfin.DanelfinProvider.get_best_stocks.

Cannot be backtested — same reasoning as danelfin_live_signals.py:
    ``/v3/beststocks`` has no historical dates (confirmed live — tested
    whether it secretly accepts a ``date`` param the way ``/ranking``
    does; it rejects it as an unknown parameter). pit_view.best_stocks()
    always returns an empty frame in backtests, so this strategy always
    emits zero signals there — expected, not a bug. Enabling this live is
    a live-only, unbacktested-by-construction judgment call.

Signal logic:
    Fixed high raw score (near this project's [-10, 10] sanity ceiling)
    for any universe symbol present in today's real Top-25, scaled
    slightly by rank so #1 scores a touch higher than #25 — not scaled by
    ``ai_score`` itself, since ``danelfin_ai_score`` already covers that
    axis; this strategy's entire signal is "Danelfin's own curated
    conviction list says buy this, right now." No signal (not a sell) for
    symbols absent from the list — being outside a curated Top-25 says
    nothing about being bearish. Confidence is fixed high (this is
    Danelfin's own top-conviction output, not a derived heuristic), never
    scaled down by anything else in this strategy.

Portfolio construction approach:
    Long only, and only ever the names Danelfin itself currently ranks in
    its real Best Stocks list.

Risk notes:
    Same black-box-vendor caveat as every other Danelfin strategy here,
    with the same "no backtest evidence at all" caveat as
    danelfin_live_signals — treat as a high-conviction but unvalidated
    input. This strategy alone will rarely fire against a small, mostly
    US-mega-cap fixed universe (Danelfin's real list skews toward
    sector-rotating small/mid-cap and international names) — see
    firm.live.danelfin_universe_sync for the separate, explicitly
    risk-reviewed mechanism that can dynamically grow the universe to
    give this strategy more surface to act on.
"""

from __future__ import annotations

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

_BASE_SCORE = 8.0  # near this project's [-10, 10] sanity ceiling
_RANK_BONUS_MAX = 1.5  # rank 1 gets +1.5, rank 25 gets ~0
_CONFIDENCE = 0.9


@register("danelfin_best_stocks_signal")
class DanelfinBestStocksSignalStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("danelfin_best_stocks_signal", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        universe = pit_view.universe
        if not universe:
            return []

        df = pit_view.best_stocks(symbols=universe)
        if df.empty or "symbol" not in df.columns:
            return []

        signals: list[Signal] = []
        for _, row in df.iterrows():
            rank = row.get("rank")
            rank_bonus = _RANK_BONUS_MAX * (1.0 - (float(rank) - 1) / 24.0) if rank else 0.0
            raw = min(_BASE_SCORE + rank_bonus, 10.0)

            signals.append(
                Signal(
                    symbol=str(row["symbol"]),
                    strategy="danelfin_best_stocks_signal",
                    score=raw,
                    confidence=_CONFIDENCE,
                    horizon="60d",
                    asof=pit_view.asof,
                    meta={
                        "rank": float(rank) if rank else float("nan"),
                        "ai_score": float(row["ai_score"]) if "ai_score" in row and row["ai_score"] is not None else float("nan"),
                        "sector": row.get("sector"),
                        "country": row.get("country"),
                    },
                )
            )
        return signals
