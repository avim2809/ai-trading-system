"""Hidden Markov Model market-regime detection.

This package centralises the Gaussian-HMM regime logic so that it can be
shared by two consumers:

* :mod:`firm.strategies.regime_hmm` — a per-symbol regime strategy that emits
  directional cross-sectional :class:`~firm.contracts.models.Signal` objects.
* :class:`firm.regime.detector.MarketRegimeDetector` — a market-level detector
  used by :class:`firm.agents.risk.RiskAgent` to scale gross exposure.

The mathematical foundation (stationary features, Gaussian emissions,
Baum-Welch fit, Viterbi decode, forward posterior, mean-return labelling and
Laplace smoothing of the transition matrix) follows Chen, Yi & Zhao (2020),
*Trading Strategy for Market Situation Estimation Based on Hidden Markov Model*.
"""

from __future__ import annotations

from firm.regime.features import (
    REGIME_FEATURES,
    apply_laplace_smoothing,
    compute_regime_features,
)
from firm.regime.model import (
    BEAR,
    BULL,
    CHOP,
    GaussianRegimeModel,
    RegimeState,
    RegimeUnavailable,
    hmm_available,
)

__all__ = [
    "REGIME_FEATURES",
    "apply_laplace_smoothing",
    "compute_regime_features",
    "GaussianRegimeModel",
    "RegimeState",
    "RegimeUnavailable",
    "hmm_available",
    "BULL",
    "BEAR",
    "CHOP",
]
