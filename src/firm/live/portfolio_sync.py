"""Portfolio reconciliation – syncs PortfolioState with real broker positions.

On every cycle the engine should call :func:`sync_portfolio_from_broker`
to ensure the internal book matches the broker's ground truth.  Discrepancies
(manual trades, partial fills, corporate actions) are logged as warnings.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from firm.brokers.base import Broker, BrokerPosition
from firm.portfolio.state import PortfolioState

log = logging.getLogger(__name__)


def sync_portfolio_from_broker(
    broker: Broker,
    portfolio: PortfolioState,
    prices: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Reconcile *portfolio* with real broker positions and account cash.

    Returns a list of discrepancy dicts (empty when perfectly in sync).
    """
    discrepancies: list[dict[str, Any]] = []

    account = broker.get_account()
    broker_cash = account.get("cash", 0.0)
    if abs(broker_cash - portfolio.cash) > 0.01:
        discrepancies.append({
            "type": "cash_mismatch",
            "expected": portfolio.cash,
            "actual": broker_cash,
            "diff": broker_cash - portfolio.cash,
        })
        log.warning(
            "Cash mismatch: internal=%.2f broker=%.2f (diff=%.2f)",
            portfolio.cash,
            broker_cash,
            broker_cash - portfolio.cash,
        )
        portfolio.cash = broker_cash

    broker_positions = broker.get_positions()
    broker_map: dict[str, BrokerPosition] = {p.symbol: p for p in broker_positions}

    # Pending (in-flight) quantity per symbol, signed by side.  An order that
    # was submitted last cycle but has not yet settled at the broker explains a
    # gap between internal and broker positions; without this, the reconciler
    # would "snap" internal state back to the pre-fill value and the next cycle
    # would re-submit the same order.
    pending, pending_ok = _pending_quantities(broker)
    if not pending_ok:
        # In-flight orders are unknown: position mismatches below may be
        # explained by unsettled orders we can't see, so callers should treat
        # the reconciliation as degraded rather than authoritative.
        discrepancies.append({
            "type": "open_orders_unavailable",
            "detail": "Could not fetch open orders; in-flight view is incomplete.",
        })

    all_symbols = (
        set(portfolio.holdings.keys()) | set(broker_map.keys()) | set(pending.keys())
    )
    for sym in sorted(all_symbols):
        internal_qty = portfolio.holdings.get(sym, 0.0)
        broker_pos = broker_map.get(sym)
        broker_qty = broker_pos.quantity if broker_pos else 0.0
        # Expected broker quantity once in-flight orders settle.
        expected_qty = broker_qty + pending.get(sym, 0.0)

        # If internal already matches the post-settlement view, the difference
        # is explained by an in-flight order; leave internal state untouched.
        if abs(internal_qty - expected_qty) <= 0.001:
            continue

        if abs(internal_qty - broker_qty) > 0.001:
            discrepancies.append({
                "type": "position_mismatch",
                "symbol": sym,
                "expected": internal_qty,
                "actual": broker_qty,
                "diff": broker_qty - internal_qty,
            })
            log.warning(
                "Position mismatch %s: internal=%.4f broker=%.4f",
                sym,
                internal_qty,
                broker_qty,
            )
            if broker_qty == 0.0:
                portfolio.holdings.pop(sym, None)
            else:
                portfolio.holdings[sym] = broker_qty

    portfolio.holdings = {s: q for s, q in portfolio.holdings.items() if q != 0}

    if prices:
        portfolio.record_snapshot(datetime.utcnow(), prices)

    return discrepancies


def _pending_quantities(broker: Broker) -> tuple[dict[str, float], bool]:
    """Return ``(per-symbol signed unfilled quantity, ok)``.

    ``ok`` is ``False`` when open orders could not be fetched, so the caller
    can flag the reconciliation as degraded instead of trusting an empty
    in-flight view.
    """
    pending: dict[str, float] = {}
    try:
        open_orders = broker.get_open_orders()
    except Exception:
        log.warning("Could not fetch open orders for reconciliation", exc_info=True)
        return pending, False
    for o in open_orders:
        remaining = max(0.0, o.quantity - o.filled_quantity)
        if remaining <= 0:
            continue
        sign = 1.0 if o.side == "buy" else -1.0
        pending[o.symbol] = pending.get(o.symbol, 0.0) + sign * remaining
    return pending, True
