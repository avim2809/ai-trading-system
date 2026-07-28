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
    4. Bull/Bear signals are additionally scaled by how *statistically
       distinguishable* the labelled state's mean return is from its
       nearest-ranked neighbour (``RegimeState.separation``, a pooled-std
       effect size computed at fit time).  A thin, noise-level margin between
       two states is a known driver of label-switching (the same underlying
       state gets called "Bull" one retrain and "Chop"/"Bear" the next), so a
       low-separation label is damped toward neutral rather than traded at
       full confidence — see ``min_state_separation``/``separation_damping_floor``.

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

    A per-symbol, ~1yr-window, 3-state full-covariance Gaussian HMM is prone
    to overfitting/instability between retrains, and states are labelled
    purely by *realized* mean return within the fit window — i.e. "Bull"
    means "the cluster with the best recent returns", not a genuine forward
    forecast. This makes the strategy structurally closer to a
    momentum/trend-echo signal than an independent regime forecast, and it
    showed a negative Sharpe in 6/6 historical diagnostic windows before the
    separation-based damping above was added (see
    ``docs/portfolio_construction_diagnosis.md``). If negative-Sharpe drag
    persists after this fix, consider a generic per-strategy rolling-Sharpe
    circuit breaker at the signal-combination layer instead of further
    strategy-specific tuning.
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
        # Minimum pooled-std effect size (RegimeState.separation) required
        # before a Bull/Bear label is trusted at full confidence. Below this,
        # the label is statistically indistinguishable from its neighbour
        # (label-switching risk) and gets damped toward neutral. Calibrated
        # against the diagnostic backtest windows in
        # docs/portfolio_construction_diagnosis.md.
        "min_state_separation": 0.5,
        # Floor on the damping multiplier applied when separation is below
        # min_state_separation, so a directional (if low-conviction) signal
        # is still emitted rather than fully zeroed.
        "separation_damping_floor": 0.15,
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
        min_state_separation = float(p["min_state_separation"])
        separation_damping_floor = float(p["separation_damping_floor"])

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

            damping = 1.0
            if state.label in (BULL, BEAR):
                if min_state_separation > 0:
                    damping = max(
                        separation_damping_floor,
                        min(1.0, state.separation / min_state_separation),
                    )
                    if damping < 1.0:
                        logger.debug(
                            "regime_hmm %s: %s label separation=%.3f < threshold=%.3f, "
                            "damping signal by %.2fx (label-switching risk)",
                            symbol, state.label, state.separation,
                            min_state_separation, damping,
                        )
                score = damping * state.confidence if state.label == BULL else -damping * state.confidence
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
                        "separation": state.separation,
                        "separation_damping": damping,
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
