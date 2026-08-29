"""Monte Carlo robustness analysis for strategy return streams.

Bootstrap-resamples a return series to characterise the *distribution* of
outcomes a headline backtest number hides: how bad drawdowns get, the
probability of a loss over a holding period, and a confidence interval on
forward returns.

Ported from the external trading-suite ``backtesting-frameworks`` /
``strategy-audit`` skills and adapted to firm's pandas-based metrics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from firm.eval.metrics import sharpe_ratio

log = logging.getLogger(__name__)


class MonteCarloAnalyzer:
    """Bootstrap simulation for strategy robustness.

    Parameters
    ----------
    n_simulations:
        Number of bootstrap paths.
    confidence:
        Confidence level for interval / worst-case percentiles (e.g. 0.95).
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.n_simulations = n_simulations
        self.confidence = confidence
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core resampler
    # ------------------------------------------------------------------

    def bootstrap_returns(
        self,
        returns: pd.Series | np.ndarray,
        n_periods: int | None = None,
    ) -> np.ndarray:
        """Resample returns with replacement.

        Returns an array of shape ``(n_simulations, n_periods)``.
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.zeros((self.n_simulations, n_periods or 0))
        if n_periods is None:
            n_periods = arr.size
        idx = self._rng.integers(0, arr.size, size=(self.n_simulations, n_periods))
        return arr[idx]

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------

    def analyze_drawdowns(self, returns: pd.Series | np.ndarray) -> dict[str, float]:
        """Distribution of maximum drawdown across bootstrap paths.

        Drawdowns are returned as negative fractions (``-0.2`` = 20% peak-to-
        trough). ``worst_Npct`` is the worst-case at the configured confidence.
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return {}
        sims = self.bootstrap_returns(arr)
        equity = np.cumprod(1.0 + sims, axis=1)
        running_max = np.maximum.accumulate(equity, axis=1)
        drawdowns = (equity - running_max) / running_max
        max_dd = drawdowns.min(axis=1)
        pct = int(self.confidence * 100)
        return {
            "expected_max_dd": float(np.mean(max_dd)),
            "median_max_dd": float(np.median(max_dd)),
            f"worst_{pct}pct": float(np.percentile(max_dd, (1 - self.confidence) * 100)),
            "worst_case": float(max_dd.min()),
        }

    def probability_of_loss(
        self,
        returns: pd.Series | np.ndarray,
        holding_periods: list[int] | None = None,
    ) -> dict[int, float]:
        """Probability the cumulative return is negative over each holding period."""
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return {}
        if holding_periods is None:
            holding_periods = [21, 63, 126, 252]
        out: dict[int, float] = {}
        for period in holding_periods:
            if period > arr.size:
                continue
            sims = self.bootstrap_returns(arr, period)
            total = np.prod(1.0 + sims, axis=1) - 1.0
            out[period] = float((total < 0).mean())
        return out

    def confidence_interval(
        self,
        returns: pd.Series | np.ndarray,
        periods: int = 252,
    ) -> dict[str, float]:
        """Confidence interval for the cumulative return over ``periods`` steps."""
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return {}
        sims = self.bootstrap_returns(arr, periods)
        total = np.prod(1.0 + sims, axis=1) - 1.0
        lower = (1 - self.confidence) / 2
        upper = 1 - lower
        return {
            "expected": float(total.mean()),
            "lower_bound": float(np.percentile(total, lower * 100)),
            "upper_bound": float(np.percentile(total, upper * 100)),
            "std": float(total.std(ddof=1)),
        }

    def sharpe_confidence_interval(
        self,
        returns: pd.Series | np.ndarray,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> dict[str, float]:
        """Bootstrap confidence interval for the annualized Sharpe ratio.

        Resamples via :meth:`bootstrap_returns` and computes the annualized
        Sharpe per path (vectorized, matching ``firm.eval.metrics.
        sharpe_ratio``'s formula), then reports the same two-sided
        percentile interval as :meth:`confidence_interval` — e.g.
        ``MonteCarloAnalyzer(confidence=0.90).sharpe_confidence_interval(...)``
        gives a 90% CI whose ``lower_bound`` is the 5th percentile.
        ``point_estimate`` is the actual (non-bootstrapped) Sharpe of
        ``returns`` itself, for comparison against the bootstrap distribution.
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return {}
        sims = self.bootstrap_returns(arr)
        daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess = sims - daily_rf
        std = sims.std(axis=1, ddof=1)
        safe_std = np.where(std < 1e-14, 1.0, std)
        sharpes = np.where(
            std < 1e-14, 0.0, excess.mean(axis=1) / safe_std * np.sqrt(periods_per_year),
        )
        lower = (1 - self.confidence) / 2
        upper = 1 - lower
        return {
            "expected": float(sharpes.mean()),
            "lower_bound": float(np.percentile(sharpes, lower * 100)),
            "upper_bound": float(np.percentile(sharpes, upper * 100)),
            "std": float(sharpes.std(ddof=1)),
            "point_estimate": sharpe_ratio(
                pd.Series(arr),
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),
        }

    def summary(self, returns: pd.Series | np.ndarray) -> dict[str, object]:
        """Convenience roll-up of all three analyses for a report block."""
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            log.debug(
                "MonteCarloAnalyzer.summary: too few observations (%d); "
                "returning empty", arr.size,
            )
            return {}
        log.debug(
            "MonteCarloAnalyzer.summary: %d obs x %d simulations (confidence=%.2f)",
            arr.size, self.n_simulations, self.confidence,
        )
        return {
            "n_simulations": self.n_simulations,
            "confidence": self.confidence,
            "drawdowns": self.analyze_drawdowns(arr),
            "probability_of_loss": {
                str(k): v for k, v in self.probability_of_loss(arr).items()
            },
            "confidence_interval": self.confidence_interval(arr),
        }
