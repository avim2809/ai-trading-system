"""Broker ABC and shared data types for live/paper trading."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from firm.time_utils import utcnow

log = logging.getLogger(__name__)


class BrokerError(Exception):
    """Raised when a broker operation fails."""


@dataclass
class OrderRequest:
    """Instruction to place an order with a broker."""

    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None
    strategy: str = "composite"
    time_in_force: str = "day"
    # Client-side idempotency token.  When set, it is forwarded to the broker
    # (Alpaca ``client_order_id`` / IBKR ``orderRef``) so a retried or
    # overlapping cycle that re-submits the same order is deduplicated by the
    # broker instead of double-filling.
    client_order_id: str | None = None


@dataclass
class OrderStatus:
    """Tracks the lifecycle of a submitted order."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: Literal["pending", "filled", "partial", "cancelled", "rejected"] = "pending"
    timestamp: datetime = field(default_factory=utcnow)


@dataclass
class BrokerPosition:
    """A single position as reported by the broker."""

    symbol: str
    quantity: float
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


class Broker(ABC):
    """Abstract base for all broker integrations."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker."""

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the broker connection."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the broker connection is active."""

    def reconnect(self) -> None:
        """Tear down and re-establish the connection.

        Concrete default (not abstract) so every broker gets mid-session
        recovery for free without each implementation duplicating the same
        disconnect-then-connect dance: this is what
        ``LiveTradingEngine`` calls when a cycle's first broker call raises
        ``BrokerError`` (e.g. IB Gateway dropped the socket, or its
        mandatory daily restart happened mid-session — see
        ``docs/PROJECT_CONTEXT.md`` "Broker & host failover"). ``disconnect()``
        failures are swallowed (the connection may already be dead — that's
        exactly why we're here) but a failing ``connect()`` propagates so the
        caller knows the reconnect attempt itself failed. Brokers with
        different reconnect semantics (e.g. refreshing an OAuth token instead
        of a full re-handshake) may override this.
        """
        try:
            self.disconnect()
        except Exception as exc:
            log.debug("Broker disconnect during reconnect failed (%s); proceeding to connect() anyway", exc)
        self.connect()

    @abstractmethod
    def get_account(self) -> dict:
        """Return account summary: cash, equity, buying_power (at minimum)."""

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Return all open positions."""

    @abstractmethod
    def get_position(self, symbol: str) -> BrokerPosition | None:
        """Return the position for *symbol*, or None if flat."""

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderStatus:
        """Submit an order and return its initial status."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.  Return True on success."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Poll current status of a previously submitted order."""

    @abstractmethod
    def get_open_orders(self) -> list[OrderStatus]:
        """Return all orders that are not yet terminal."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Return last/mid price for *symbol*."""

    @abstractmethod
    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Batch price lookup."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """Return True if the primary market is currently open."""
