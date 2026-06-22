"""HMM-based market-regime detection strategy (per-symbol).

Financial intuition:
    Financial markets are non-stationary: they cycle between structurally
    different behavioural states ("regimes") that a single homogeneous model
    treats incorrectly.  A Gaussian Hidden Markov Model infers the *hidden*
    regime that generated the observed (stationarised) market data and adapts
    positioning accordingly.  A 3-state Gaussian HMM has been shown to
    outperform double-moving-average strategies in both total return and
    drawdown control (Chen, Yi & Zhao, 2020).

Data inputs:
    Per-symbol OHLCV from PitView.prices() (<= asof, so strictly no
    look-ahead), stationarised into log returns, 5-day cumulative log returns,
    14-period ATR and a 20-day volume-spike ratio.

Signal logic:
    1. For each symbol, fit a Gaussian HMM (Baum-Welch) on its feature history
       using only data up to ``asof``.  Fitting is cached per symbol and
       refit every ``retrain_frequency`` days (walk-forward).
    2. Decode the current regime with the forward posterior and label states
       Bull / Chop / Bear by mean log return.
    3. Emit a directional signal: +confidence in Bull, -confidence in Bear,
       a damped value in Chop.  Downstream ``zscore_signals`` re-normalises the
       scores cross-sectionally across the universe.

Portfolio construction approach:
    Long symbols in a confident Bull regime, short symbols in a confident Bear
    regime, lightly weight / avoid choppy names.  Confidence is the HMM
    posterior probability of the active regime.

Risk notes:
    The HMM needs several candles of new data before it updates its estimate,
    creating a lag on sudden regime shifts.  Using the posterior (rather than
    the hard-decoded state) means partial probability updates feed through to
    sizing before full regime confirmation.  A complementary market-level
    overlay (firm.agents.risk.RiskAgent) scales gross exposure by the broad
    regime to catch fast moves the per-name signal lags.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.regime.features import compute_regime_features
from firm.regime.model import BEAR, BULL, GaussianRegimeModel, RegimeUnavailable
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

logger = logging.getLogger(__name__)


@register("regime_hmm")
class HMMRegimeStrategy(BaseStrategy):
    """Per-symbol Gaussian-HMM regime detector emitting directional signals."""

    #: Surfaced by the /api/strategies endpoint so the UI renders editable
    #: parameter fields. Also the single source of truth for runtime defaults.
    default_params: dict = {
        "n_states": 3,
        "lookback_days": 252,
        "retrain_frequency": 21,
        "min_data_points": 60,
        "chop_damping": 0.1,
        "horizon": "5d",
    }

    def __init__(self, params: dict | None = None):
        super().__init__("regime_hmm", params)
        # Lazily-built per-symbol models + last-fit timestamps for walk-forward
        # caching (mirrors the retrain cadence of MLPredictionStrategy).
        self._models: dict[str, GaussianRegimeModel] = {}
        self._last_fit: dict[str, pd.Timestamp] = {}
        self._unavailable_logged = False

    def generate(self, pit_view: PitView) -> list[Signal]:
        p = {**self.default_params, **(self.params or {})}
        n_states = int(p["n_states"])
        lookback_days = int(p["lookback_days"])
        retrain_frequency = int(p["retrain_frequency"])
        min_data_points = int(p["min_data_points"])
        chop_damping = float(p["chop_damping"])
        horizon = str(p["horizon"])

        universe = pit_view.universe
        if not universe:
            return []

        prices_df = pit_view.prices(symbols=universe, lookback_days=lookback_days + 25)
        if prices_df.empty:
            return []

        asof_ts = pd.Timestamp(pit_view.asof)
        signals: list[Signal] = []

        for symbol, sym_df in prices_df.groupby("symbol"):
            features = compute_regime_features(sym_df)
            if len(features) < max(min_data_points, n_states + 1):
                continue

            X = features.values
            model = self._get_model(str(symbol), X, asof_ts, n_states, retrain_frequency)
            if model is None:
                # hmmlearn unavailable or fit failed for this symbol — skip it.
                continue

            try:
                state = model.classify(X)
            except Exception:
                logger.debug("Regime decode failed for %s", symbol, exc_info=True)
                continue

            if state.label == BULL:
                score = state.confidence
            elif state.label == BEAR:
                score = -state.confidence
            else:  # Chop — damped, near-neutral
                score = chop_damping * (state.confidence - 1.0 / n_states)

            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="regime_hmm",
                    score=float(score),
                    confidence=float(state.confidence),
                    horizon=horizon,
                    asof=pit_view.asof,
                    meta={
                        "regime": state.label,
                        "state_idx": state.state_idx,
                        "posterior": state.posterior,
                        "n_states": n_states,
                    },
                )
            )

        return signals

    def _get_model(
        self,
        symbol: str,
        X: np.ndarray,
        asof_ts: pd.Timestamp,
        n_states: int,
        retrain_frequency: int,
    ) -> GaussianRegimeModel | None:
        """Return a fitted model for *symbol*, refitting on the retrain cadence.

        Returns ``None`` when no usable model can be produced (hmmlearn missing
        or a per-symbol fit failure), so the caller skips that symbol rather
        than aborting the whole bar.
        """
        last = self._last_fit.get(symbol)
        cached = self._models.get(symbol)
        needs_fit = (
            cached is None
            or last is None
            or (asof_ts - last).days >= retrain_frequency
        )
        if not needs_fit:
            return cached

        try:
            model = GaussianRegimeModel(n_states=n_states).fit(X)
        except RegimeUnavailable as exc:
            if not self._unavailable_logged:
                logger.warning("regime_hmm disabled: %s", exc)
                self._unavailable_logged = True
            return None
        except Exception:
            # Numerical / convergence failure for this symbol: fall back to the
            # last good model if we have one, else skip the symbol this bar.
            logger.debug("HMM fit failed for %s; using cached model if any", symbol, exc_info=True)
            return cached

        self._models[symbol] = model
        self._last_fit[symbol] = asof_ts
        return model
