"""Tests for the broker abstraction layer.

Uses a MockBroker that implements the Broker ABC without any real connections.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from firm.brokers.base import (
    Broker,
    BrokerError,
    BrokerPosition,
    OrderRequest,
    OrderStatus,
)
from firm.time_utils import utcnow


# ---------------------------------------------------------------------------
# MockBroker – full Broker ABC implementation for testing
# ---------------------------------------------------------------------------

class MockBroker(Broker):
    """In-memory broker that simulates order fills instantly."""

    def __init__(self, initial_cash: float = 100_000) -> None:
        self._connected = False
        self._cash = initial_cash
        self._positions: dict[str, BrokerPosition] = {}
        self._orders: dict[str, OrderStatus] = {}
        self._prices: dict[str, float] = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOG": 140.0,
            "AMZN": 180.0,
            "TSLA": 250.0,
        }
        self._market_open = True

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_account(self) -> dict[str, Any]:
        equity = self._cash + sum(
            p.market_value for p in self._positions.values()
        )
        return {
            "cash": self._cash,
            "equity": equity,
            "buying_power": self._cash * 2,
        }

    def get_positions(self) -> list[BrokerPosition]:
        return list(self._positions.values())

    def get_position(self, symbol: str) -> BrokerPosition | None:
        return self._positions.get(symbol)

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        if not self._connected:
            raise BrokerError("Not connected")
        price = self._prices.get(order.symbol, 0.0)
        if price <= 0:
            raise BrokerError(f"No price for {order.symbol}")

        oid = uuid.uuid4().hex[:8]
        cost = price * order.quantity

        if order.side == "buy":
            self._cash -= cost
            existing = self._positions.get(order.symbol)
            if existing:
                total_qty = existing.quantity + order.quantity
                total_cost = existing.avg_cost * existing.quantity + cost
                self._positions[order.symbol] = BrokerPosition(
                    symbol=order.symbol,
                    quantity=total_qty,
                    avg_cost=total_cost / total_qty,
                    market_value=total_qty * price,
                    unrealized_pnl=0.0,
                )
            else:
                self._positions[order.symbol] = BrokerPosition(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_cost=price,
                    market_value=cost,
                    unrealized_pnl=0.0,
                )
        else:
            self._cash += cost
            existing = self._positions.get(order.symbol)
            if existing:
                remaining = existing.quantity - order.quantity
                if remaining <= 0:
                    del self._positions[order.symbol]
                else:
                    self._positions[order.symbol] = BrokerPosition(
                        symbol=order.symbol,
                        quantity=remaining,
                        avg_cost=existing.avg_cost,
                        market_value=remaining * price,
                        unrealized_pnl=0.0,
                    )

        status = OrderStatus(
            order_id=oid,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            avg_fill_price=price,
            status="filled",
            timestamp=utcnow(),
        )
        self._orders[oid] = status
        return status

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = "cancelled"
            return True
        return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        if order_id not in self._orders:
            raise BrokerError(f"Order {order_id} not found")
        return self._orders[order_id]

    def get_open_orders(self) -> list[OrderStatus]:
        return [o for o in self._orders.values() if o.status == "pending"]

    def get_current_price(self, symbol: str) -> float:
        price = self._prices.get(symbol, 0.0)
        if price <= 0:
            raise BrokerError(f"No price for {symbol}")
        return price

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: self._prices[s] for s in symbols if s in self._prices}

    def is_market_open(self) -> bool:
        return self._market_open


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBrokerABC:
    """Verify the ABC contract is correctly enforced."""

    def test_mock_broker_is_a_broker(self):
        assert issubclass(MockBroker, Broker)

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Broker()


class TestMockBrokerConnect:
    def test_connect_disconnect(self):
        b = MockBroker()
        assert not b.is_connected()
        b.connect()
        assert b.is_connected()
        b.disconnect()
        assert not b.is_connected()

    def test_account_initial(self):
        b = MockBroker(initial_cash=50_000)
        b.connect()
        acct = b.get_account()
        assert acct["cash"] == 50_000
        assert acct["equity"] == 50_000


class TestOrderLifecycle:
    @pytest.fixture()
    def broker(self) -> MockBroker:
        b = MockBroker()
        b.connect()
        return b

    def test_buy_creates_position(self, broker: MockBroker):
        req = OrderRequest(symbol="AAPL", side="buy", quantity=10)
        status = broker.submit_order(req)

        assert status.status == "filled"
        assert status.filled_quantity == 10
        assert status.avg_fill_price == 150.0

        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 10
        assert pos.avg_cost == 150.0

    def test_sell_reduces_position(self, broker: MockBroker):
        broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=10))
        broker.submit_order(OrderRequest(symbol="AAPL", side="sell", quantity=4))

        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 6

    def test_sell_all_removes_position(self, broker: MockBroker):
        broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=10))
        broker.submit_order(OrderRequest(symbol="AAPL", side="sell", quantity=10))

        assert broker.get_position("AAPL") is None
        assert len(broker.get_positions()) == 0

    def test_cash_tracking(self, broker: MockBroker):
        initial = broker.get_account()["cash"]
        broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=10))
        after_buy = broker.get_account()["cash"]
        assert after_buy == initial - 10 * 150.0

        broker.submit_order(OrderRequest(symbol="AAPL", side="sell", quantity=10))
        after_sell = broker.get_account()["cash"]
        assert after_sell == initial

    def test_order_status_tracking(self, broker: MockBroker):
        status = broker.submit_order(OrderRequest(symbol="MSFT", side="buy", quantity=5))
        retrieved = broker.get_order_status(status.order_id)
        assert retrieved.order_id == status.order_id
        assert retrieved.status == "filled"

    def test_cancel_order(self, broker: MockBroker):
        status = broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=1))
        assert broker.cancel_order(status.order_id) is True
        assert broker.get_order_status(status.order_id).status == "cancelled"

    def test_cancel_nonexistent(self, broker: MockBroker):
        assert broker.cancel_order("nonexistent") is False

    def test_order_not_connected_raises(self):
        b = MockBroker()
        with pytest.raises(BrokerError, match="Not connected"):
            b.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=1))


class TestPrices:
    def test_get_current_price(self):
        b = MockBroker()
        assert b.get_current_price("AAPL") == 150.0

    def test_get_current_prices_batch(self):
        b = MockBroker()
        prices = b.get_current_prices(["AAPL", "MSFT", "UNKNOWN"])
        assert prices["AAPL"] == 150.0
        assert prices["MSFT"] == 300.0
        assert "UNKNOWN" not in prices

    def test_unknown_symbol_raises(self):
        b = MockBroker()
        with pytest.raises(BrokerError):
            b.get_current_price("UNKNOWN")


class TestMarketOpen:
    def test_market_open_default(self):
        b = MockBroker()
        assert b.is_market_open() is True

    def test_market_closed(self):
        b = MockBroker()
        b._market_open = False
        assert b.is_market_open() is False


class TestDataclasses:
    def test_order_request_defaults(self):
        req = OrderRequest(symbol="AAPL", side="buy", quantity=10)
        assert req.order_type == "market"
        assert req.limit_price is None
        assert req.time_in_force == "day"
        assert req.strategy == "composite"

    def test_order_request_limit(self):
        req = OrderRequest(
            symbol="AAPL", side="buy", quantity=10,
            order_type="limit", limit_price=145.0
        )
        assert req.order_type == "limit"
        assert req.limit_price == 145.0

    def test_order_status_defaults(self):
        s = OrderStatus(order_id="abc", symbol="AAPL", side="buy", quantity=10)
        assert s.filled_quantity == 0.0
        assert s.status == "pending"

    def test_broker_position_defaults(self):
        p = BrokerPosition(symbol="AAPL", quantity=100)
        assert p.avg_cost == 0.0
        assert p.unrealized_pnl == 0.0

    def test_order_request_mutable(self):
        req = OrderRequest(symbol="AAPL", side="buy", quantity=10)
        req.quantity = 20
        assert req.quantity == 20

    def test_broker_position_mutable(self):
        p = BrokerPosition(symbol="AAPL", quantity=100, avg_cost=150.0)
        p.market_value = 16000.0
        assert p.market_value == 16000.0
