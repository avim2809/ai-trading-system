"""Reconciles persisted order-history status against the broker's ground truth.

``TradeHistoryStore.record_orders()`` writes each order's status once, at
submission time. An order that fills or gets cancelled asynchronously
afterward — the common case for IBKR, where fills arrive as a later
callback — never gets that update: the record stays frozen at "pending"
forever even though the order genuinely settled seconds later. This module
closes that gap by polling the broker for the current status of every
locally non-terminal order and correcting the persisted record in place.
"""

from __future__ import annotations

import logging

from firm.brokers.base import Broker, BrokerError
from firm.live.trade_history import TradeHistoryStore

log = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"filled", "cancelled", "rejected"})


def reconcile_order_statuses(
    trade_history: TradeHistoryStore,
    broker: Broker,
    *,
    max_orders: int = 200,
) -> int:
    """Poll the broker for every locally non-terminal order and correct it.

    Returns the number of records updated. Best-effort: a broker error for
    one order fails that order alone rather than aborting the whole pass —
    an order the broker no longer knows about (e.g. from before a process
    restart wiped its in-session trade cache) is left as-is rather than
    guessed at.
    """
    pending = [
        o
        for o in trade_history.list_orders(limit=max_orders)
        if o.get("status") not in _TERMINAL_STATUSES and o.get("order_id")
    ]
    if not pending:
        return 0

    updated = 0
    for order in pending:
        order_id = str(order["order_id"])
        try:
            live_status = broker.get_order_status(order_id)
        except BrokerError:
            log.debug("Order %s not known to broker — leaving status as-is", order_id)
            continue
        except Exception:
            log.warning(
                "Could not reconcile order %s — unexpected broker error",
                order_id, exc_info=True,
            )
            continue

        if (
            live_status.status == order.get("status")
            and live_status.filled_quantity == order.get("filled_quantity")
        ):
            continue

        if trade_history.update_order_status(
            order_id,
            status=live_status.status,
            filled_quantity=live_status.filled_quantity,
            avg_fill_price=live_status.avg_fill_price,
        ):
            updated += 1
            log.info(
                "Reconciled order %s (%s): %s -> %s (filled=%.4f @ %.4f)",
                order_id, order.get("symbol"), order.get("status"),
                live_status.status, live_status.filled_quantity, live_status.avg_fill_price,
            )
    return updated
