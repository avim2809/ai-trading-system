"""Statistical-significance testing for confirmed chart patterns against a
matched-volatility null (Part A of the 2026-09-27 false-positive-rate fix).

The scorer rebalance (``scorer.py``) fixed two components that never
discriminated noise from signal, but a heuristic quality_score -- however
well-calibrated on average -- is still just one number computed the same
way for every symbol, with no notion of "is 75 unusually good FOR THIS
SYMBOL right now, or is this symbol just naturally noisy." This module
answers that with an actual hypothesis test: build a null distribution of
"best confirmed quality_score" from many surrogate price paths that share
the real symbol's own realized return distribution and typical intrabar
range (so the null is matched to its actual volatility/wick behavior) but
have no real structure (an i.i.d. bootstrap resample of daily returns,
plus an independently-shuffled volume series to break any real
price-volume co-movement) -- then a candidate's p-value is simply how rare
its own score is against that symbol-specific null.

This is a permutation/surrogate-data test, not a parametric one: no
assumption that returns are normally distributed, just that resampling the
symbol's own realized returns with replacement destroys any genuine
multi-bar structure (trend, pattern, momentum) while preserving its
marginal return distribution (fat tails and all) and typical volatility.
Reuses ``firm.eval.robustness.MonteCarloAnalyzer.bootstrap_returns`` for
the resampling step rather than reimplementing it.

Deliberately NOT wired into the live default path yet
(``significance_test_enabled: False`` in
``firm.strategies.pattern_recognition.PatternRecognitionStrategy.default_params``)
-- like every other new pattern_recognition knob, it needs its own
walk-forward re-validation before being trusted live, and this is also
computationally heavier per-symbol than the rest of the scanner (order
``n_draws`` extra ``scan_symbol`` calls), so it's designed to be
cached/amortized (see ``cached_null_score_distribution``) rather than
recomputed every cycle.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from firm.eval.robustness import MonteCarloAnalyzer
from firm.patterns.scanner import scan_symbol

log = logging.getLogger(__name__)

DEFAULT_N_DRAWS = 200
DEFAULT_SEED = 42


def _synthesize_surrogate_ohlcv(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    sampled_returns: np.ndarray,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One surrogate OHLCV frame from one row of bootstrap-resampled log
    returns, matching the real symbol's own typical intrabar wick range
    and volume distribution (shuffled, to break any real price-volume
    co-movement -- exactly the coincidence a real breakout's volume
    confirmation depends on, which a genuine null must not accidentally
    preserve).
    """
    n = len(close)
    surrogate_close = float(close[0]) * np.exp(np.concatenate([[0.0], np.cumsum(sampled_returns)]))
    # Real bars' own average half-range (as a fraction of close) -- used so
    # the surrogate's high/low spread matches this symbol's actual
    # intrabar behavior instead of an arbitrary fixed wick.
    with np.errstate(divide="ignore", invalid="ignore"):
        half_range_pct = np.where(close > 0, (high - low) / (2.0 * close), 0.0)
    half_range_pct = half_range_pct[np.isfinite(half_range_pct)]
    typical_half_range = float(np.median(half_range_pct)) if half_range_pct.size else 0.002
    typical_half_range = max(typical_half_range, 1e-4)

    surrogate_high = surrogate_close * (1.0 + typical_half_range)
    surrogate_low = surrogate_close * (1.0 - typical_half_range)
    surrogate_volume = rng.permutation(volume) if len(volume) == n else np.full(n, float(np.mean(volume)))

    return pd.DataFrame({
        "high": surrogate_high,
        "low": surrogate_low,
        "close": surrogate_close,
        "volume": surrogate_volume,
    })


def build_null_score_distribution(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    zigzag_pct: float = 0.03,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """The null distribution: ``n_draws`` independent surrogate price paths
    (bootstrap-resampled from this symbol's own realized log returns),
    each scanned the same way real data would be, recording the BEST
    (max) confirmed ``quality_score`` per draw (0.0 if nothing confirmed).

    This is deliberately "best across all 9 detectors/17 pattern names per
    draw," matching how a real scan takes the single best match per
    symbol per cycle (``PatternRecognitionStrategy.generate``) -- the null
    must reflect the same "best of many tries" selection the real
    candidate went through, or the comparison isn't apples-to-apples (this
    is exactly the multiple-comparisons concern Part A's FDR control also
    addresses at the cross-sectional level; this module addresses it at
    the per-symbol, per-detector-family level).
    """
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    volume = np.asarray(volume, dtype=float)
    n = len(close)
    if n < 21:
        log.debug("build_null_score_distribution: only %d bars, too short to bootstrap -- returning empty null", n)
        return np.zeros(0)

    log_returns = np.diff(np.log(np.clip(close, 1e-9, None)))
    analyzer = MonteCarloAnalyzer(n_simulations=n_draws, seed=seed)
    sampled_returns = analyzer.bootstrap_returns(log_returns, n_periods=n - 1)  # (n_draws, n-1)

    rng = np.random.default_rng(seed)
    null_scores = np.zeros(n_draws)
    for i in range(n_draws):
        surrogate = _synthesize_surrogate_ohlcv(high, low, close, volume, sampled_returns[i], rng=rng)
        try:
            matches = scan_symbol(surrogate, min_score=0.0, zigzag_pct=zigzag_pct)
        except Exception:
            log.debug("build_null_score_distribution: scan_symbol failed on surrogate draw %d", i, exc_info=True)
            continue
        null_scores[i] = max((m.quality_score for m in matches), default=0.0)
    return null_scores


def benjamini_hochberg_accept(p_values: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure: which of ``p_values`` are
    "discoveries" while controlling the False Discovery Rate at level
    ``q``. Returns a boolean array the same length/order as ``p_values``.

    This is the cross-sectional half of Part A's false-positive fix, and
    a DIFFERENT multiple-comparisons problem from the one
    :func:`build_null_score_distribution` already controls for: that
    null's "best score across all 9 detectors/17 pattern names per
    surrogate draw" already accounts for the WITHIN-symbol multiplicity
    (a single symbol trying many detectors/windows and keeping the best
    hit). Scanning ~25-35 symbols every cycle and independently accepting
    anything with ``p <= 0.05`` is a SEPARATE, ACROSS-symbol multiplicity
    problem: even if not one of them has a real pattern, ~5% of 30
    symbols (1-2 names) would clear an uncorrected p<0.05 bar by chance,
    every single cycle. BH control bounds the *expected proportion* of
    such false discoveries among everything accepted, rather than
    bounding each symbol's error rate in isolation.

    Standard step-up procedure: sort p-values ascending, find the largest
    rank ``k`` (1-indexed) with ``p_(k) <= (k / m) * q`` (``m`` = number of
    tests), and accept every hypothesis at or below that rank.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * q
    passing = sorted_p <= thresholds
    if not np.any(passing):
        return np.zeros(m, dtype=bool)
    # Largest passing rank -- every hypothesis at or below it is accepted
    # (the defining step-up property: a later, larger p-value can still be
    # "accepted" as long as SOME later-or-equal rank clears its own
    # threshold).
    max_passing_rank = np.max(np.where(passing)[0])
    accept_sorted = np.zeros(m, dtype=bool)
    accept_sorted[: max_passing_rank + 1] = True
    accept = np.zeros(m, dtype=bool)
    accept[order] = accept_sorted
    return accept


def pattern_p_value(quality_score: float, null_scores: np.ndarray) -> float | None:
    """Fraction of the null distribution at least as extreme as
    ``quality_score`` -- the standard ``(1 + count) / (1 + n)`` permutation
    p-value estimator (never exactly 0, so a downstream ``-log10(p)``-style
    use never divides by zero). ``None`` if the null distribution is empty
    (too little history to bootstrap -- see
    :func:`build_null_score_distribution`).
    """
    if null_scores.size == 0:
        return None
    count_at_least_as_extreme = int(np.sum(null_scores >= quality_score))
    return (1.0 + count_at_least_as_extreme) / (1.0 + null_scores.size)


@lru_cache(maxsize=256)
def _cached_null_score_distribution(
    high_bytes: bytes, low_bytes: bytes, close_bytes: bytes, volume_bytes: bytes,
    shape: tuple[int, ...], n_draws: int, zigzag_pct: float, seed: int,
) -> np.ndarray:
    high = np.frombuffer(high_bytes, dtype=np.float64).reshape(shape)
    low = np.frombuffer(low_bytes, dtype=np.float64).reshape(shape)
    close = np.frombuffer(close_bytes, dtype=np.float64).reshape(shape)
    volume = np.frombuffer(volume_bytes, dtype=np.float64).reshape(shape)
    return build_null_score_distribution(high, low, close, volume, n_draws=n_draws, zigzag_pct=zigzag_pct, seed=seed)


def cached_null_score_distribution(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
    *, n_draws: int = DEFAULT_N_DRAWS, zigzag_pct: float = 0.03, seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Same as :func:`build_null_score_distribution`, memoized on the exact
    input arrays (+ parameters) via an LRU cache -- this is the entry
    point live callers should use, so a symbol whose scanned window hasn't
    changed since the last cycle (the common case: most cycles append at
    most one new bar) doesn't re-run ``n_draws`` extra ``scan_symbol``
    calls every time. Cache key is the raw array bytes, so it naturally
    misses (and recomputes) the moment the window actually changes.
    """
    high = np.ascontiguousarray(high, dtype=np.float64)
    low = np.ascontiguousarray(low, dtype=np.float64)
    close = np.ascontiguousarray(close, dtype=np.float64)
    volume = np.ascontiguousarray(volume, dtype=np.float64)
    return _cached_null_score_distribution(
        high.tobytes(), low.tobytes(), close.tobytes(), volume.tobytes(),
        close.shape, n_draws, zigzag_pct, seed,
    )
