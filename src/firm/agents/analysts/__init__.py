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


def zscore_signals(signals: list[Signal], demean: bool = True) -> list[Signal]:
    """Cross-sectional score normalisation, applied per strategy.

    Groups signals by ``strategy`` name, then within each group rescales
    ``score`` by the cross-sectional dispersion (sample std, ddof=1) across
    the symbols in that group.  Groups with fewer than two signals or zero
    standard deviation are passed through unchanged.

    ``demean`` (default ``True``, unchanged prior behavior) subtracts the
    cross-sectional mean before dividing by std -- a true z-score, forcing
    every strategy's output to mean 0 across the universe *every bar*, which
    destroys aggregate level/direction information: a strategy that
    genuinely reads the whole universe as bullish (all raw scores positive)
    gets rescaled into "half the universe is above average, half below"
    regardless. That's a real, confirmed contributor to live turnover/
    whipsaw -- mid-ranked names whose demeaned score sits near zero flip
    sign on noise (day-to-day drift in *other* names' raw scores, via the
    shared mean) even when their own signal didn't meaningfully change.
    ``demean=False`` divides by the same std but skips the mean subtraction,
    so a one-sided universe stays one-sided (a genuine net-long or net-short
    read is preserved) while cross-strategy scale stays comparable. Opt-in
    (see ``zscore_demean`` in TechnicalAnalyst/FundamentalAnalyst/
    SentimentAnalyst) pending a backtest A/B against the true-z-score default.
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
        # Always computed around the true mean (a measure of dispersion),
        # regardless of whether that mean is then subtracted from the output.
        var = sum((x - mean) ** 2 for x in scores) / (n - 1)
        std = var**0.5
        if std < 1e-10:
            result.extend(strat_signals)
            continue
        offset = mean if demean else 0.0
        for s in strat_signals:
            result.append(
                Signal(
                    symbol=s.symbol,
                    strategy=s.strategy,
                    score=(s.score - offset) / std,
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

def _valid_correlation_matrix(
    strategy_returns: pd.DataFrame,
) -> tuple[list[str], pd.Series, np.ndarray] | None:
    """Shared prep for :func:`optimal_signal_weights` and
    :func:`hrp_signal_weights`: the columns with usable (finite, nonzero-std)
    history, their per-strategy vols, and the correlation matrix among them.

    Returns ``None`` when fewer than 2 columns have usable history — callers
    fall back to equal weights in that case.
    """
    cols = list(strategy_returns.columns)
    R = strategy_returns.astype(float)
    sigma = R.std(axis=0, ddof=1)
    valid = [c for c in cols if np.isfinite(sigma[c]) and sigma[c] > 0]
    if len(valid) < 2:
        return None
    Rv = R[valid]
    sig = sigma[valid]
    Z = (Rv - Rv.mean(axis=0)) / sig
    C = np.corrcoef(Z.to_numpy(), rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    return valid, sig, C


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

    See :func:`hrp_signal_weights` for an alternative that avoids inverting
    the correlation matrix at all — worth A/B-ing against this when the
    strategy count is high relative to history length (this system's usual
    situation), where ``pinv(C)`` can be numerically unstable.
    """
    cols = list(strategy_returns.columns)
    n = len(cols)
    if n == 0:
        return pd.Series(dtype=float), 0.0
    if n == 1:
        return pd.Series([1.0], index=cols, name="weight"), 1.0

    prep = _valid_correlation_matrix(strategy_returns)
    if prep is None:
        w = pd.Series(1.0 / n, index=cols, name="weight")
        return w, float(n)
    valid, sig, C = prep

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


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (HRP) combination — alternative to `optimal`
# ---------------------------------------------------------------------------

def _hrp_quasi_diagonal_order(link: np.ndarray, n_items: int) -> list[int]:
    """Leaf order implied by a scipy ``linkage`` matrix.

    Repeatedly expands the root's children until only original leaf indices
    (``< n_items``) remain, preserving left-to-right dendrogram order so
    that hierarchically similar items end up adjacent in the returned list
    — this is what lets recursive bisection split "outward" along the
    correlation structure instead of an arbitrary index order.
    """
    link = link.astype(int)
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n_items:
        expanded: list[int] = []
        for idx in order:
            if idx >= n_items:
                child_row = link[idx - n_items]
                expanded.append(int(child_row[0]))
                expanded.append(int(child_row[1]))
            else:
                expanded.append(idx)
        order = expanded
    return order


def _hrp_cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    """Variance of the inverse-variance portfolio over a cluster's members."""
    sub_cov = cov[np.ix_(items, items)]
    inv_diag = 1.0 / np.diag(sub_cov)
    ivp = inv_diag / inv_diag.sum()
    return float(ivp @ sub_cov @ ivp)


def _hrp_recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Top-down HRP weight allocation (López de Prado).

    Starting from the full quasi-diagonal ``order``, repeatedly bisect each
    cluster in half and split its weight between the two halves inversely
    proportional to each half's own inverse-variance portfolio variance —
    so a correlated block of strategies is down-weighted as a unit rather
    than double-counted strategy-by-strategy, without ever inverting a
    covariance/correlation matrix.
    """
    n = len(order)
    w = np.ones(n)
    clusters = [list(range(n))]
    while clusters:
        clusters = [
            c[start:end]
            for c in clusters
            if len(c) > 1
            for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
        ]
        for i in range(0, len(clusters) - 1, 2):
            left, right = clusters[i], clusters[i + 1]
            left_items = [order[p] for p in left]
            right_items = [order[p] for p in right]
            var_left = _hrp_cluster_variance(cov, left_items)
            var_right = _hrp_cluster_variance(cov, right_items)
            total = var_left + var_right
            alpha = (1.0 - var_left / total) if total > 0 else 0.5
            for p in left:
                w[p] *= alpha
            for p in right:
                w[p] *= 1.0 - alpha
    out = np.zeros(n)
    for pos, orig_idx in enumerate(order):
        out[orig_idx] = w[pos]
    return out


def hrp_signal_weights(
    strategy_returns: pd.DataFrame,
) -> tuple[pd.Series, float]:
    """Hierarchical Risk Parity weights across strategies.

    An alternative to :func:`optimal_signal_weights` that never inverts the
    correlation matrix: strategies are hierarchically clustered by
    correlation distance (``d = sqrt(0.5 * (1 - correlation))``), then
    weight is allocated top-down by recursively bisecting the resulting tree
    and splitting each split inversely to the two halves' inverse-variance
    portfolio variance. This avoids the numerical instability
    ``optimal_signal_weights``'s ``pinv(Σ)`` can hit with a small, highly
    correlated strategy set (this system's usual situation: ~10-12
    strategies, short history) — HRP degrades gracefully instead of
    producing wildly swinging weights from an ill-conditioned inverse.

    Returns ``(weights, effective_n)`` in the same shape as
    :func:`optimal_signal_weights` — weights are always non-negative and
    already sum to 1 (no clipping/re-normalising needed) — and falls back to
    equal weights under the same degenerate conditions (fewer than 2
    strategies with usable history).
    """
    cols = list(strategy_returns.columns)
    n = len(cols)
    if n == 0:
        return pd.Series(dtype=float), 0.0
    if n == 1:
        return pd.Series([1.0], index=cols, name="weight"), 1.0

    prep = _valid_correlation_matrix(strategy_returns)
    if prep is None:
        w = pd.Series(1.0 / n, index=cols, name="weight")
        return w, float(n)
    valid, sig, C = prep

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    m = len(valid)
    cov = np.outer(sig.to_numpy(), sig.to_numpy()) * C
    dist = np.sqrt(np.clip(0.5 * (1.0 - C), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    order = _hrp_quasi_diagonal_order(link, m)
    hrp_w = _hrp_recursive_bisection(cov, order)

    weights = pd.Series(0.0, index=cols, name="weight")
    for i, c in enumerate(valid):
        weights[c] = float(hrp_w[i])

    w2 = float((weights.to_numpy() ** 2).sum())
    effective_n = (1.0 / w2) if w2 > 0 else 0.0
    return weights, effective_n


def combine_signals_hrp(
    signals: list[Signal],
    strategy_returns: dict[str, pd.Series] | None = None,
) -> dict[str, float]:
    """Aggregate per-symbol scores with Hierarchical Risk Parity weights.

    Mirrors :func:`combine_signals_optimal`'s structure exactly (bucket by
    symbol, build a per-symbol return frame, fall back to the
    confidence-weighted mean when fewer than two of the symbol's strategies
    have usable history) but weights via :func:`hrp_signal_weights` instead
    of the inverse-covariance method.

    Returns ``{symbol: combined_score}``.
    """
    strategy_returns = strategy_returns or {}
    buckets: dict[str, list[Signal]] = {}
    for s in signals:
        buckets.setdefault(s.symbol, []).append(s)

    combined: dict[str, float] = {}
    n_hrp = 0
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
        weights, _ = hrp_signal_weights(frame)
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
            n_hrp += 1
    log.debug(
        "combine_signals_hrp: %d symbols HRP-weighted, %d fell back to "
        "confidence-weighted mean", n_hrp, n_fallback,
    )
    return combined


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
