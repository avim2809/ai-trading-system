"""Analyst agents – produce SignalSets from raw data.

Shared utilities for cross-sectional normalization live here so that all
analyst implementations stay DRY.
"""

from __future__ import annotations

from firm.contracts.models import Signal


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
        var = sum((x - mean) ** 2 for x in scores) / n
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
