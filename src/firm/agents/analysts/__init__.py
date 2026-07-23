"""Analyst agents – produce SignalSets from raw data.

Shared utilities for cross-sectional normalization live here so that all
analyst implementations stay DRY.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from firm.contracts.models import Signal

log = logging.getLogger(__name__)


def zscore_signals(signals: list[Signal]) -> list[Signal]:
    """Cross-sectional z-score normalisation, applied per strategy.

    Groups signals by ``strategy`` name, then within each group replaces
    ``score`` with its z-score across the symbols in that group.  Groups
    with fewer than two signals or zero standard deviation are passed
    through unchanged.
    """
    by_strategy: dict[str, list[Signal]] = {}
    for sig in signals:
        by_strategy.setdefault(sig.strategy, []).append(sig)

    result: list[Signal] = []
    for strat_signals in by_strategy.values():
        scores = [s.score for s in strat_signals]
        n = len(scores)
        if n < 2:
            result.extend(strat_signals)
            continue
        mean = sum(scores) / n
        # Sample std (ddof=1) to match the pandas convention strategies use
        # for their own z-scoring, so the two normalization stages are
        # consistent rather than mixing population and sample variance.
        var = sum((x - mean) ** 2 for x in scores) / (n - 1)
        std = var**0.5
        if std < 1e-10:
            result.extend(strat_signals)
            continue
        for s in strat_signals:
            result.append(
                Signal(
                    symbol=s.symbol,
                    strategy=s.strategy,
                    score=(s.score - mean) / std,
                    confidence=s.confidence,
                    horizon=s.horizon,
                    asof=s.asof,
                    meta=s.meta,
                )
            )
    return result


def combine_signals_by_symbol(
    signals: list[Signal],
    weight_by_confidence: bool = True,
) -> dict[str, float]:
    """Aggregate per-symbol scores across strategies.

    Returns ``{symbol: combined_score}``.  If *weight_by_confidence* is
    ``True`` the combination is a confidence-weighted average; otherwise
    a simple mean.
    """
    buckets: dict[str, list[Signal]] = {}
    for s in signals:
        buckets.setdefault(s.symbol, []).append(s)

    combined: dict[str, float] = {}
    for sym, sigs in buckets.items():
        if weight_by_confidence:
            total_conf = sum(s.confidence for s in sigs)
            if total_conf > 0:
                combined[sym] = sum(s.score * s.confidence for s in sigs) / total_conf
            else:
                combined[sym] = sum(s.score for s in sigs) / len(sigs)
        else:
            combined[sym] = sum(s.score for s in sigs) / len(sigs)
    return combined


# ---------------------------------------------------------------------------
# Optimal alpha combination (inverse-covariance / independent-edge weighting)
# ---------------------------------------------------------------------------

def optimal_signal_weights(
    strategy_returns: pd.DataFrame,
) -> tuple[pd.Series, float]:
    """Minimum-variance / independent-edge weights across strategies.

    Given a ``periods × strategies`` frame of historical per-strategy returns,
    weight each strategy by the inverse of its return *covariance* with the
    stack: ``w ∝ Σ⁻¹·1`` where ``Σ = D·C·D`` (``D`` = diag of per-strategy
    vols, ``C`` = correlation). This simultaneously (a) down-weights
    high-variance strategies (inverse-variance) and (b) splits weight between
    redundant/correlated strategies instead of double-counting them
    (independent-edge, via the pseudo-inverse of the correlation matrix).

    Returns ``(weights, effective_n)`` where ``weights`` are L1-normalised
    (``sum |w| = 1``) and ``effective_n`` is the participation ratio
    ``1 / Σ wᵢ²`` — the number of *effectively independent* strategies, which
    falls below the raw count as signals become correlated. Negative
    (hedging) weights are clipped to zero so signal directions are never
    flipped; falls back to equal weights when history is unusable.
    """
    cols = list(strategy_returns.columns)
    n = len(cols)
    if n == 0:
        return pd.Series(dtype=float), 0.0
    if n == 1:
        return pd.Series([1.0], index=cols, name="weight"), 1.0

    R = strategy_returns.astype(float)
    sigma = R.std(axis=0, ddof=1)
    valid = [c for c in cols if np.isfinite(sigma[c]) and sigma[c] > 0]
    if len(valid) < 2:
        w = pd.Series(1.0 / n, index=cols, name="weight")
        return w, float(n)

    Rv = R[valid]
    sig = sigma[valid]
    Z = (Rv - Rv.mean(axis=0)) / sig
    C = np.corrcoef(Z.to_numpy(), rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    inv_sigma = (1.0 / sig).to_numpy()
    # w ∝ Σ⁻¹·1 = D⁻¹ C⁻¹ D⁻¹ · 1  (pinv keeps it defined under duplicates).
    raw = inv_sigma * (np.linalg.pinv(C, rcond=1e-10) @ inv_sigma)
    raw = np.clip(raw, 0.0, None)  # no sign flips from hedging weights

    weights = pd.Series(0.0, index=cols, name="weight")
    denom = float(np.abs(raw).sum())
    if denom <= 0:
        w = pd.Series(1.0 / n, index=cols, name="weight")
        return w, float(n)
    for c, r in zip(valid, raw):
        weights[c] = r / denom

    w2 = float((weights.to_numpy() ** 2).sum())
    effective_n = (1.0 / w2) if w2 > 0 else 0.0
    return weights, effective_n


def combine_signals_optimal(
    signals: list[Signal],
    strategy_returns: dict[str, pd.Series] | None = None,
) -> dict[str, float]:
    """Aggregate per-symbol scores with optimal (inverse-covariance) weights.

    For each symbol, weights the contributing strategies by
    :func:`optimal_signal_weights` using their historical return series in
    *strategy_returns*. When fewer than two of the symbol's strategies have
    usable history, falls back to the confidence-weighted mean
    (:func:`combine_signals_by_symbol`) so behaviour degrades gracefully.

    Returns ``{symbol: combined_score}``.
    """
    strategy_returns = strategy_returns or {}
    buckets: dict[str, list[Signal]] = {}
    for s in signals:
        buckets.setdefault(s.symbol, []).append(s)

    combined: dict[str, float] = {}
    n_optimal = 0
    n_fallback = 0
    for sym, sigs in buckets.items():
        strats = [s.strategy for s in sigs]
        hist = {
            st: strategy_returns[st]
            for st in strats
            if st in strategy_returns and len(strategy_returns[st].dropna()) >= 2
        }
        if len(hist) < 2:
            combined.update(combine_signals_by_symbol(sigs, weight_by_confidence=True))
            n_fallback += 1
            continue

        frame = pd.DataFrame(hist).dropna(how="all")
        weights, _ = optimal_signal_weights(frame)
        # Strategies without history contribute at zero weight; if that drops
        # everything, fall back to the confidence-weighted mean.
        score = 0.0
        wsum = 0.0
        for s in sigs:
            w = float(weights.get(s.strategy, 0.0))
            score += w * s.score
            wsum += abs(w)
        if wsum <= 0:
            combined.update(combine_signals_by_symbol(sigs, weight_by_confidence=True))
            n_fallback += 1
        else:
            combined[sym] = score
            n_optimal += 1
    log.debug(
        "combine_signals_optimal: %d symbols optimally weighted, %d fell back to "
        "confidence-weighted mean", n_optimal, n_fallback,
    )
    return combined
