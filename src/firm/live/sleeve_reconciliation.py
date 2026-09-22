"""Corrects sleeved-mode holdings once a cycle's real broker fill is known.

Sleeve books (``Orchestrator._sleeve_portfolios``) are credited from a
virtual execution pass at decision time -- each sleeve assumes it got its
full decided quantity filled, at the decision-time price, immediately. The
real, broker-side order for a symbol is the *net* across every contributing
sleeve that cycle, so a partial fill, a cancellation, or a rejection is
never fed back into any sleeve's book: see
docs/claude-memory/project_vps_migration_sep22.md and
/root/.claude/plans/compressed-mapping-frost.md for the full incident this
was built for.

The broker's real fill for a symbol is a single number covering every
sleeve that contributed to that cycle's net order -- there is no way to
recover, in general, "which sleeve's shares actually filled" when more than
one sleeve holds the same symbol. This module does NOT attempt that. It
applies one deliberate, safe compromise instead: scale every contributing
sleeve's decided quantity by the same realized-vs-decided ratio. This never
guesses which sleeve was "right" -- it treats every contributor
proportionally -- and it is exactly correct (not just a safe approximation)
in the single-sleeve case, which degenerates to blended mode's own
self-heal for that one sleeve.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.portfolio.state import PortfolioState

log = logging.getLogger(__name__)


def correction_key(cycle_id: int, symbol: str) -> str:
    """Idempotency key for LiveStateStore.save_corrected_fills -- a
    correction must apply exactly once per (cycle_id, symbol), even if the
    order-reconciliation job re-scans the same now-terminal order later."""
    return f"{cycle_id}:{symbol}"


def apply_realized_fill(
    sleeve_portfolios: dict[str, PortfolioState],
    sleeve_decisions: dict[str, dict[str, Any]],
    symbol: str,
    real_filled_qty: float,
    avg_fill_price: float,
) -> dict[str, float]:
    """Apportion one symbol's real fill across this cycle's contributing
    sleeves, in place, correcting each sleeve's earlier hypothetical credit.

    ``sleeve_decisions`` is one cycle's ``Blackboard.sleeve_decisions``
    (persisted verbatim to ``cycle_history.json``) -- ``{strategy: {...,
    "fills": [{"symbol", "shares", "price"}, ...]}}``. Only sleeves with a
    nonzero decided quantity for ``symbol`` this cycle are touched.

    Returns ``{strategy: correction_applied}`` (signed shares) for logging
    -- empty if no sleeve decided anything for this symbol this cycle
    (``net_decision_qty == 0``), in which case nothing is touched.

    Cash correction uses each sleeve's own decision-time price to reverse
    its original hypothetical debit, NOT ``avg_fill_price`` applied to the
    share delta -- a cancelled order reports ``avg_fill_price=0.0`` (no fill
    happened), and ``correction * 0.0`` would silently lose the entire cash
    reversal for a full cancellation instead of returning the cash the
    original (nonzero decision-price) hypothetical debit took out.
    """
    decided_by_strategy: dict[str, dict[str, float]] = {}
    for strategy, decision in sleeve_decisions.items():
        for fill in decision.get("fills") or []:
            if fill.get("symbol") == symbol:
                entry = decided_by_strategy.setdefault(strategy, {"qty": 0.0, "dollar": 0.0})
                shares = float(fill["shares"])
                entry["qty"] += shares
                entry["dollar"] += shares * float(fill["price"])

    net_decision_qty = sum(e["qty"] for e in decided_by_strategy.values())
    if net_decision_qty == 0:
        return {}

    fill_ratio = real_filled_qty / net_decision_qty
    applied: dict[str, float] = {}
    for strategy, entry in decided_by_strategy.items():
        portfolio = sleeve_portfolios.get(strategy)
        if portfolio is None:
            continue
        decided_qty, decided_dollar = entry["qty"], entry["dollar"]
        realized_qty = decided_qty * fill_ratio
        holdings_correction = realized_qty - decided_qty
        # Original debit already applied was `-decided_dollar`; the correct
        # debit is `-realized_qty * avg_fill_price` -- credit back the
        # difference.
        cash_correction = decided_dollar - realized_qty * avg_fill_price
        if holdings_correction == 0 and cash_correction == 0:
            continue
        portfolio.holdings[symbol] = portfolio.holdings.get(symbol, 0.0) + holdings_correction
        if portfolio.holdings[symbol] == 0:
            del portfolio.holdings[symbol]
        portfolio.cash += cash_correction
        applied[strategy] = holdings_correction

    if applied:
        log.info(
            "Sleeve fill correction for %s: real_filled=%.4f decided=%.4f "
            "ratio=%.4f corrections=%s",
            symbol, real_filled_qty, net_decision_qty, fill_ratio, applied,
        )
    return applied
