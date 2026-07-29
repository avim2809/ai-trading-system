"""Market-level regime detector used by the risk overlay.

Fits a single Gaussian HMM on a *market proxy* series and caches it, refitting
only every ``retrain_frequency`` calls (the same retrain-cadence pattern used
by :mod:`firm.strategies.ml_prediction`).  Decoding between refits is cheap, so
the detector adds negligible per-bar cost to the risk pipeline.

All data is read through the supplied :class:`~firm.strategies.base.PitView`,
so the detector inherits the no-look-ahead guarantee of the point-in-time store.
"""

from __future__ import annotations

import logging

import pandas as pd

from firm.regime.ensemble import DEFAULT_SEEDS, EnsembleRegimeModel
from firm.regime.features import compute_regime_features
from firm.regime.model import GaussianRegimeModel, RegimeState, RegimeUnavailable

log = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detect the prevailing market regime from a broad proxy series.

    ``ensemble=True`` swaps the single :class:`~firm.regime.model.GaussianRegimeModel`
    for a :class:`~firm.regime.ensemble.EnsembleRegimeModel` (majority vote
    over several independently-seeded HMM fits) behind the exact same
    ``fit``/``classify`` interface — every downstream consumer (``RiskAgent``'s
    ``regime_overlay``, ``strategy_regime_weights``) is unaffected by which
    one is active. Off by default: unchanged behaviour unless explicitly
    opted in.
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback_days: int = 504,
        retrain_frequency: int = 21,
        benchmark_symbol: str | None = None,
        min_data_points: int = 120,
        random_state: int = 42,
        ensemble: bool = False,
        ensemble_seeds: tuple[int, ...] = DEFAULT_SEEDS,
    ) -> None:
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.retrain_frequency = max(1, retrain_frequency)
        self.benchmark_symbol = benchmark_symbol
        self.min_data_points = min_data_points
        self.random_state = random_state
        self.ensemble = ensemble
        self.ensemble_seeds = ensemble_seeds
        self._model: GaussianRegimeModel | EnsembleRegimeModel | None = None
        self._bars_since_fit = 0
        self._warned_unavailable = False

    def _market_proxy(self, pit_view) -> pd.DataFrame:
        """Build an OHLCV (or close-only) proxy for the market.

        Prefers the configured benchmark symbol; falls back to the
        equal-weight average close of the universe when the benchmark is not
        present in the loaded data.
        """
        if self.benchmark_symbol:
            bench = pit_view.prices(
                symbols=[self.benchmark_symbol], lookback_days=self.lookback_days
            )
            if bench is not None and not bench.empty:
                return bench.sort_values("date")

        prices = pit_view.prices(lookback_days=self.lookback_days)
        if prices is None or prices.empty:
            return pd.DataFrame()
        pivot = (
            prices.pivot_table(index="date", columns="symbol", values="adj_close")
            .sort_index()
        )
        if pivot.empty:
            return pd.DataFrame()
        proxy = pivot.mean(axis=1).dropna()
        return pd.DataFrame({"date": proxy.index, "close": proxy.values})

    def detect(self, pit_view) -> RegimeState | None:
        """Return the current :class:`RegimeState`, or ``None`` if unavailable.

        ``None`` is returned (and the caller treats the overlay as a no-op)
        when there is insufficient history or ``hmmlearn`` is not installed.
        """
        proxy = self._market_proxy(pit_view)
        if proxy.empty:
            return None

        features = compute_regime_features(proxy)
        if len(features) < self.min_data_points:
            return None

        X = features.values
        needs_fit = self._model is None or self._bars_since_fit >= self.retrain_frequency
        if needs_fit:
            try:
                model: GaussianRegimeModel | EnsembleRegimeModel
                if self.ensemble:
                    model = EnsembleRegimeModel(
                        n_states=self.n_states, seeds=self.ensemble_seeds
                    ).fit(X)
                else:
                    model = GaussianRegimeModel(
                        n_states=self.n_states, random_state=self.random_state
                    ).fit(X)
                self._model = model
                self._bars_since_fit = 0
            except RegimeUnavailable as exc:
                if not self._warned_unavailable:
                    log.warning("Market regime detection unavailable: %s", exc)
                    self._warned_unavailable = True
                return None
            except Exception:
                # Numerical / convergence failure: reuse the last good model if
                # we have one, otherwise give up for this bar.
                log.warning("Market regime refit failed; reusing prior model", exc_info=True)
                if self._model is None:
                    return None

        self._bars_since_fit += 1
        try:
            return self._model.classify(X)
        except Exception:
            log.warning("Market regime decode failed", exc_info=True)
            return None
