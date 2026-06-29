"""Statistical arbitrage (pairs trading) strategy.

Financial intuition:
    Pairs of stocks that are economically linked (same sector, supply chain)
    tend to maintain a stable long-run price relationship.  When the spread
    between a pair deviates significantly from its historical mean, a
    convergence trade (long the undervalued leg, short the overvalued leg)
    harvests the reversion.

Data inputs:
    Adjusted close prices from PitView.prices() for the universe.  Pairs
    are selected by highest rolling correlation (or can be pre-specified).

Signal logic:
    1. Identify top-correlated pairs from the universe (or use a
       pre-configured pair list).
    2. For each pair, estimate a hedge ratio via OLS regression
       (statsmodels) of asset B on asset A over the lookback window.
    3. Compute the spread = price_B - hedge_ratio * price_A.
    4. Z-score the spread using its rolling mean and std.
    5. If z > +entry_z → short B / long A; if z < -entry_z → long B / short A.
    6. If |z| < exit_z → flatten (score ≈ 0).

Portfolio construction approach:
    Each pair produces two opposing signals (one per leg).  Position sizing
    should ensure dollar-neutrality within each pair.

Risk notes:
    Pairs can "break" permanently due to structural changes (M&A,
    regulatory shifts).  A stop-loss on divergence beyond a maximum z-score
    is essential.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

logger = logging.getLogger(__name__)


def _ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Compute OLS hedge ratio (beta) of y on x using statsmodels."""
    try:
        import statsmodels.api as sm

        x_const = sm.add_constant(x.values)
        model = sm.OLS(y.values, x_const).fit()
        return float(model.params[1])
    except Exception:
        if x.std() == 0:
            return 1.0
        cov = np.cov(y.values, x.values)
        return float(cov[0, 1] / cov[1, 1])


@register("stat_arb")
class StatArbStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("stat_arb", params)

    def _find_top_pairs(
        self, pivot: pd.DataFrame, max_pairs: int
    ) -> list[tuple[str, str]]:
        """Return the *max_pairs* most-correlated symbol pairs."""
        corr = pivot.corr()
        pairs: list[tuple[float, str, str]] = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                c = corr.loc[a, b]
                if not np.isnan(c):
                    pairs.append((abs(c), a, b))
        pairs.sort(reverse=True)
        return [(a, b) for _, a, b in pairs[:max_pairs]]

    def generate(self, pit_view: PitView) -> list[Signal]:
        lookback_days: int = self.params.get("lookback_days", 60)
        entry_z: float = self.params.get("entry_z", 2.0)
        exit_z: float = self.params.get("exit_z", 0.5)
        max_pairs: int = self.params.get("max_pairs", 5)
        predefined_pairs: list[tuple[str, str]] | None = self.params.get(
            "predefined_pairs", None
        )

        universe = pit_view.universe
        if not universe or len(universe) < 2:
            return []

        prices_df = pit_view.prices(symbols=universe, lookback_days=lookback_days + 10)
        if prices_df.empty:
            return []

        pivot = (
            prices_df.pivot_table(index="date", columns="symbol", values="adj_close")
            .sort_index()
            .dropna(axis=1, how="any")
        )
        if pivot.shape[1] < 2 or len(pivot) < 20:
            return []

        if predefined_pairs is not None:
            valid = set(pivot.columns)
            pairs = [(a, b) for a, b in predefined_pairs if a in valid and b in valid]
        else:
            pairs = self._find_top_pairs(pivot, max_pairs)

        if not pairs:
            return []

        signals: list[Signal] = []
        for sym_a, sym_b in pairs:
            series_a = pivot[sym_a]
            series_b = pivot[sym_b]

            hedge_ratio = _ols_hedge_ratio(series_b, series_a)
            spread = series_b - hedge_ratio * series_a

            spread_mean = spread.mean()
            spread_std = spread.std()
            if spread_std == 0 or np.isnan(spread_std):
                continue

            z = float((spread.iloc[-1] - spread_mean) / spread_std)

            if abs(z) >= entry_z:
                score_a = z
                score_b = -z
                confidence = min(abs(z) / 4.0, 1.0)
            elif abs(z) < exit_z:
                score_a = 0.0
                score_b = 0.0
                confidence = 0.1
            else:
                score_a = z * 0.5
                score_b = -z * 0.5
                confidence = min(abs(z) / 4.0, 0.5)

            meta = {
                "pair": f"{sym_a}/{sym_b}",
                "hedge_ratio": hedge_ratio,
                "spread_z": z,
            }
            signals.append(
                Signal(
                    symbol=sym_a,
                    strategy="stat_arb",
                    score=float(np.clip(score_a, -5, 5)),
                    confidence=confidence,
                    horizon="5d",
                    asof=pit_view.asof,
                    meta=meta,
                )
            )
            signals.append(
                Signal(
                    symbol=sym_b,
                    strategy="stat_arb",
                    score=float(np.clip(score_b, -5, 5)),
                    confidence=confidence,
                    horizon="5d",
                    asof=pit_view.asof,
                    meta=meta,
                )
            )

        return signals
