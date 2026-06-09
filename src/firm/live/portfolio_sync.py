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

    all_symbols = set(portfolio.holdings.keys()) | set(broker_map.keys())
    for sym in sorted(all_symbols):
        internal_qty = portfolio.holdings.get(sym, 0.0)
        broker_pos = broker_map.get(sym)
        broker_qty = broker_pos.quantity if broker_pos else 0.0

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
