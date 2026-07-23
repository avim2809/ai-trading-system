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
       (statsmodels) of log(asset B) on log(asset A) over the lookback.
    3. Optionally require Engle-Granger cointegration on the spread.
    4. Compute spread z-score using mean/std of the spread **excluding**
       the current bar (no look-ahead in the normalisation).
    5. Net opposing pair legs into **one signal per symbol**.
    6. If |z| >= entry_z → short rich / long cheap; elif |z| < exit_z → flat.

Portfolio construction approach:
    Each symbol receives a single net score.  Position sizing should ensure
    dollar-neutrality within each pair at execution time.

Risk notes:
    Pairs can "break" permanently due to structural changes (M&A,
    regulatory shifts).  A stop-loss on divergence beyond a maximum z-score
    is essential.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

logger = logging.getLogger(__name__)


def _ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Compute OLS hedge ratio (beta) of log(y) on log(x)."""
    y_log = np.log(y.replace(0, np.nan).dropna())
    x_log = np.log(x.reindex(y_log.index).replace(0, np.nan).dropna())
    common = y_log.index.intersection(x_log.index)
    if len(common) < 10:
        return 1.0
    y_log = y_log.loc[common]
    x_log = x_log.loc[common]
    try:
        import statsmodels.api as sm

        x_const = sm.add_constant(x_log.values)
        model = sm.OLS(y_log.values, x_const).fit()
        return float(model.params[1])
    except Exception:
        if x_log.std() == 0:
            return 1.0
        cov = np.cov(y_log.values, x_log.values)
        return float(cov[0, 1] / cov[1, 1])


def _spread_is_cointegrated(spread: pd.Series, max_pvalue: float) -> bool:
    """Engle-Granger step 1: ADF test on the spread level."""
    clean = spread.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 20:
        return False
    try:
        from statsmodels.tsa.stattools import adfuller

        pvalue = float(adfuller(clean, autolag="AIC")[1])
        return pvalue <= max_pvalue
    except Exception:
        logger.debug("ADF cointegration test failed", exc_info=True)
        return False


@register("stat_arb")
class StatArbStrategy(BaseStrategy):
    #: Surfaced by ``GET /api/strategies`` for backtest UI param fields.
    default_params: dict = {
        "lookback_days": 60,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "max_pairs": 5,
        "require_cointegration": True,
        "coint_pvalue": 0.10,
    }

    def __init__(self, params: dict | None = None):
        super().__init__("stat_arb", params)

    def _find_top_pairs(
        self, pivot: pd.DataFrame, max_pairs: int
    ) -> list[tuple[str, str]]:
        """Return the *max_pairs* most-correlated symbol pairs (log returns)."""
        log_ret = np.log(pivot / pivot.shift(1)).dropna(how="all")
        corr = log_ret.corr()
        pairs: list[tuple[float, str, str]] = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
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
        require_cointegration: bool = self.params.get("require_cointegration", True)
        coint_pvalue: float = self.params.get("coint_pvalue", 0.10)
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
            .dropna(axis=1, how="all")
        )
        # Require most recent observations; allow a few interior gaps per symbol.
        min_obs = max(20, int(len(pivot) * 0.85))
        pivot = pivot.dropna(axis=1, thresh=min_obs)
        if pivot.shape[1] < 2 or len(pivot) < 20:
            return []

        if predefined_pairs is not None:
            valid = set(pivot.columns)
            pairs = [(a, b) for a, b in predefined_pairs if a in valid and b in valid]
        else:
            pairs = self._find_top_pairs(pivot, max_pairs)

        if not pairs:
            return []

        # Accumulate pair legs, then net to one score per symbol.
        leg_scores: dict[str, list[float]] = defaultdict(list)
        leg_conf: dict[str, list[float]] = defaultdict(list)
        leg_meta: dict[str, list[dict]] = defaultdict(list)

        for sym_a, sym_b in pairs:
            series_a = pivot[sym_a]
            series_b = pivot[sym_b]

            hedge_ratio = _ols_hedge_ratio(series_b, series_a)
            spread = np.log(series_b.replace(0, np.nan)) - hedge_ratio * np.log(
                series_a.replace(0, np.nan)
            )
            spread = spread.dropna()
            if len(spread) < 10:
                continue

            if require_cointegration and not _spread_is_cointegrated(spread, coint_pvalue):
                continue

            hist = spread.iloc[:-1]
            if hist.empty:
                continue
            spread_mean = hist.mean()
            spread_std = hist.std()
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
            for sym, sc in ((sym_a, score_a), (sym_b, score_b)):
                leg_scores[sym].append(float(np.clip(sc, -5, 5)))
                leg_conf[sym].append(confidence)
                leg_meta[sym].append(meta)

        if not leg_scores:
            return []

        signals: list[Signal] = []
        for symbol in sorted(leg_scores):
            scores = leg_scores[symbol]
            confs = leg_conf[symbol]
            net_score = float(np.mean(scores))
            net_conf = float(np.mean(confs))
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="stat_arb",
                    score=net_score,
                    confidence=net_conf,
                    horizon="5d",
                    asof=pit_view.asof,
                    meta={
                        "net_score": net_score,
                        "pair_count": len(scores),
                        "pairs": leg_meta[symbol],
                    },
                )
            )
        return signals
