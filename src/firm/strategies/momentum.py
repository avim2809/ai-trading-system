"""Cross-sectional momentum strategy.

Financial intuition:
    Securities that have performed well over the past 6-12 months tend to
    continue outperforming in the near term (Jegadeesh & Titman, 1993).
    The most recent month is skipped to avoid the well-documented short-term
    reversal effect.

Data inputs:
    Adjusted close prices from PitView.prices() with ~260 trading-day
    lookback (≈12 months).

Signal logic:
    1. Compute cumulative return over [skip+1 .. lookback] months for each
       symbol (12-1 month momentum by default).
    2. Cross-sectionally z-score these returns across the universe.
    3. The z-score *is* the signal score — positive values indicate relative
       strength, negative values indicate relative weakness.

Portfolio construction approach:
    Downstream consumers should go long the top decile (highest scores) and
    short the bottom decile (lowest scores), creating a dollar-neutral
    long/short portfolio.

Risk notes:
    Momentum strategies are exposed to sharp, infrequent "momentum crashes"
    (reversals) that tend to occur during market stress or regime changes.
    Combine with a trend filter or volatility-managed overlay.
"""

from __future__ import annotations

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register


@register("momentum")
class MomentumStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("momentum", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        lookback_months: int = self.params.get("lookback_months", 12)
        skip_months: int = self.params.get("skip_months", 1)
        top_pct: float = self.params.get("top_pct", 0.1)
        bottom_pct: float = self.params.get("bottom_pct", 0.1)

        lookback_days = lookback_months * 21
        skip_days = skip_months * 21

        universe = pit_view.universe
        if not universe:
            return []

        prices_df = pit_view.prices(symbols=universe, lookback_days=lookback_days + 5)
        if prices_df.empty:
            return []

        pivot = (
            prices_df.pivot_table(index="date", columns="symbol", values="adj_close")
            .sort_index()
        )
        if len(pivot) < lookback_days:
            lookback_days = len(pivot)
        if lookback_days <= skip_days + 21:
            return []

        end_idx = len(pivot) - skip_days if skip_days > 0 else len(pivot)
        start_idx = max(0, end_idx - (lookback_days - skip_days))

        prices_start = pivot.iloc[start_idx]
        prices_end = pivot.iloc[end_idx - 1]
        cum_ret = (prices_end / prices_start) - 1.0
        cum_ret = cum_ret.dropna()

        if len(cum_ret) < 3:
            return []

        # Emit raw cumulative returns; analysts z-score cross-sectionally once.
        n = len(cum_ret)
        top_n = max(1, int(n * top_pct))
        bottom_n = max(1, int(n * bottom_pct))
        sorted_ret = cum_ret.sort_values(ascending=False)
        top_symbols = set(sorted_ret.head(top_n).index)
        bottom_symbols = set(sorted_ret.tail(bottom_n).index)

        signals: list[Signal] = []
        for symbol, raw in cum_ret.items():
            raw_f = float(raw)
            if symbol in top_symbols or symbol in bottom_symbols:
                confidence = min(abs(raw_f) / 0.30, 1.0)
            else:
                confidence = min(abs(raw_f) / 0.50, 0.5)

            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="momentum",
                    score=raw_f,
                    confidence=confidence,
                    horizon="21d",
                    asof=pit_view.asof,
                    meta={
                        "cum_return": raw_f,
                        "in_top_decile": symbol in top_symbols,
                        "in_bottom_decile": symbol in bottom_symbols,
                    },
                )
            )
        return signals
