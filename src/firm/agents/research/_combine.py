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
    combine_signals_optimal,
)

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
    correlated/redundant strategies. It degrades to the confidence-weighted
    mean per symbol whenever history is too thin.
    """
    signals = [
        sig
        for symbol in blackboard.get_all_symbols()
        for sig in blackboard.get_signals_by_symbol(symbol)
    ]
    if not signals:
        return {}

    cfg = config or {}
    combo = cfg.get("signal_combination") or {}
    method = combo.get("method", "confidence")
    strategy_returns = getattr(ctx, "strategy_returns", None)

    if method == "optimal" and strategy_returns:
        log.debug(
            "net_scores: optimal (inverse-covariance) combination over %d signals",
            len(signals),
        )
        return combine_signals_optimal(signals, strategy_returns)
    if method == "optimal":
        log.debug(
            "net_scores: 'optimal' requested but no strategy_returns in context; "
            "falling back to confidence-weighted mean (%d signals)",
            len(signals),
        )
    return combine_signals_by_symbol(signals, weight_by_confidence=True)
