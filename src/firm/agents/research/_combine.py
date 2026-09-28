"""Shared net-score computation for the bull/bear researchers.

Keeps the two researchers symmetric: both derive a single signed "net" score
per symbol here, so the bull (net > 0) and bear (net < 0) sides always agree
on the sign and magnitude.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.analysts import (
    combine_signals_by_symbol,
    combine_signals_hrp,
    combine_signals_optimal,
    combine_signals_optimal_robust,
    combine_signals_random_weights,
)
from firm.agents.research._circuit_breaker import apply_circuit_breaker
from firm.agents.research._regime_weights import apply_strategy_regime_weights

log = logging.getLogger(__name__)


def net_scores_for_blackboard(
    blackboard: Any,
    ctx: Any,
    config: dict[str, Any] | None,
) -> dict[str, float]:
    """Return ``{symbol: net_signed_score}`` for every symbol on the board.

    Default method is the confidence-weighted mean (historical behaviour).
    When ``config['signal_combination']['method'] == 'optimal'`` **and** the
    context exposes per-strategy return history (``ctx.strategy_returns``), the
    optimal inverse-covariance combination is used instead — it down-weights
    correlated/redundant strategies. ``method == 'hrp'`` uses Hierarchical Risk
    Parity instead (same return-history requirement) — an alternative that
    never inverts the correlation matrix, worth A/B-ing against ``optimal``
    when the strategy count is high relative to history length (see
    :func:`firm.agents.analysts.hrp_signal_weights`). Both degrade to the
    confidence-weighted mean per symbol whenever history is too thin.

    Before either combination runs, an opt-in per-strategy circuit breaker
    (``config['strategy_circuit_breaker']``, disabled by default — see
    :mod:`firm.agents.research._circuit_breaker`) damps the raw score of any
    strategy with a persistently, materially negative trailing Sharpe. An
    opt-in per-strategy regime-conditional multiplier
    (``config['strategy_regime_weights']``, disabled by default — see
    :mod:`firm.agents.research._regime_weights`) scales raw scores by market
    regime before combination. Both are independent of and complementary to
    ``optimal``'s inverse-covariance weighting, which has no notion of a
    strategy's edge sign or regime fit.
    """
    signals = [
        sig
        for symbol in blackboard.get_all_symbols()
        for sig in blackboard.get_signals_by_symbol(symbol)
    ]
    if not signals:
        return {}

    cfg = config or {}
    strategy_returns = getattr(ctx, "strategy_returns", None)
    signals = apply_circuit_breaker(
        signals, strategy_returns, cfg.get("strategy_circuit_breaker")
    )
    signals = apply_strategy_regime_weights(
        signals, getattr(ctx, "market_regime", None), cfg.get("strategy_regime_weights"),
    )

    combo = cfg.get("signal_combination") or {}
    method = combo.get("method", "confidence")
    estimator = combo.get("estimator", "legacy")

    if method == "optimal" and estimator == "robust":
        # Opt-in fix for the legacy path's lock-out/cadence/unshrunk-inverse
        # defects (see combine_signals_optimal_robust). Needs no history to
        # run: strategies without it get the neutral equal share.
        source = combo.get("returns_source", "attribution")
        if source == "standalone":
            returns = getattr(ctx, "strategy_signal_returns", None)
        elif source == "attribution":
            returns = strategy_returns
        else:
            raise ValueError(
                f"signal_combination.returns_source must be 'attribution' or "
                f"'standalone', got {source!r}"
            )
        log.debug(
            "net_scores: robust optimal combination (returns_source=%s) over %d signals",
            source, len(signals),
        )
        return combine_signals_optimal_robust(
            signals, returns,
            min_obs=int(combo.get("min_obs", 20)),
            lookback_days=int(combo.get("lookback_days", 126)),
        )
    if method == "optimal" and estimator != "legacy":
        raise ValueError(
            f"signal_combination.estimator must be 'legacy' or 'robust', got {estimator!r}"
        )
    if method == "random_weights":
        log.debug("net_scores: RESEARCH placebo random_weights over %d signals", len(signals))
        return combine_signals_random_weights(
            signals, seed=int(combo.get("seed", 0)), asof=getattr(ctx, "now", None),
        )

    if method == "optimal" and strategy_returns:
        log.debug(
            "net_scores: optimal (inverse-covariance) combination over %d signals",
            len(signals),
        )
        return combine_signals_optimal(signals, strategy_returns)
    if method == "hrp" and strategy_returns:
        log.debug(
            "net_scores: HRP (hierarchical risk parity) combination over %d signals",
            len(signals),
        )
        return combine_signals_hrp(signals, strategy_returns)
    if method in ("optimal", "hrp"):
        log.debug(
            "net_scores: '%s' requested but no strategy_returns in context; "
            "falling back to confidence-weighted mean (%d signals)",
            method, len(signals),
        )
    return combine_signals_by_symbol(signals, weight_by_confidence=True)
