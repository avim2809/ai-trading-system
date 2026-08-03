"""Tests for firm.live.order_reconciliation.

order_history.json is written once at submission time and never touched
again otherwise; reconcile_order_statuses() is what lets a later, real
broker-side fill/cancel correct that stale record.
"""

from __future__ import annotations

from firm.brokers.base import BrokerError, OrderStatus
from firm.live.order_reconciliation import reconcile_order_statuses
from firm.live.trade_history import TradeHistoryStore
from firm.time_utils import utcnow


class _FakeBroker:
    def __init__(self, statuses: dict[str, OrderStatus]):
        self._statuses = statuses

    def get_order_status(self, order_id: str) -> OrderStatus:
        if order_id not in self._statuses:
            raise BrokerError(f"Order {order_id} not found")
        return self._statuses[order_id]


def _store(tmp_path) -> TradeHistoryStore:
    return TradeHistoryStore(
        orders_path=tmp_path / "orders.json", cycles_path=tmp_path / "cycles.json",
    )


def test_no_pending_orders_returns_zero(tmp_path):
    store = _store(tmp_path)
    store.record_orders([{"order_id": "1", "status": "filled"}])
    assert reconcile_order_statuses(store, _FakeBroker({})) == 0


def test_updates_pending_order_that_broker_reports_filled(tmp_path):
    store = _store(tmp_path)
    store.record_orders([{
        "order_id": "1", "symbol": "AAPL", "status": "pending",
        "filled_quantity": 0.0, "avg_fill_price": 0.0,
    }])
    broker = _FakeBroker({
        "1": OrderStatus(
            order_id="1", symbol="AAPL", side="buy", quantity=10.0,
            filled_quantity=10.0, avg_fill_price=151.25, status="filled",
            timestamp=utcnow(),
        ),
    })

    assert reconcile_order_statuses(store, broker) == 1
    record = store.list_orders()[0]
    assert record["status"] == "filled"
    assert record["filled_quantity"] == 10.0
    assert record["avg_fill_price"] == 151.25


def test_leaves_record_alone_when_status_and_fill_qty_unchanged(tmp_path):
    store = _store(tmp_path)
    store.record_orders([{
        "order_id": "1", "symbol": "AAPL", "status": "pending",
        "filled_quantity": 0.0, "avg_fill_price": 0.0,
    }])
    broker = _FakeBroker({
        "1": OrderStatus(
            order_id="1", symbol="AAPL", side="buy", quantity=10.0,
            filled_quantity=0.0, avg_fill_price=0.0, status="pending",
            timestamp=utcnow(),
        ),
    })

    assert reconcile_order_statuses(store, broker) == 0


def test_order_unknown_to_broker_is_skipped_not_raised(tmp_path):
    store = _store(tmp_path)
    store.record_orders([{
        "order_id": "gone", "symbol": "AAPL", "status": "pending",
        "filled_quantity": 0.0, "avg_fill_price": 0.0,
    }])
    assert reconcile_order_statuses(store, _FakeBroker({})) == 0
    assert store.list_orders()[0]["status"] == "pending"


def test_unexpected_broker_error_on_one_order_does_not_abort_the_rest(tmp_path):
    class _FlakyBroker:
        def get_order_status(self, order_id: str) -> OrderStatus:
            if order_id == "1":
                raise RuntimeError("connection blip")
            return OrderStatus(
                order_id=order_id, symbol="MSFT", side="sell", quantity=5.0,
                filled_quantity=5.0, avg_fill_price=300.0, status="filled",
                timestamp=utcnow(),
            )

    store = _store(tmp_path)
    store.record_orders([
        {"order_id": "1", "symbol": "AAPL", "status": "pending", "filled_quantity": 0.0, "avg_fill_price": 0.0},
        {"order_id": "2", "symbol": "MSFT", "status": "pending", "filled_quantity": 0.0, "avg_fill_price": 0.0},
    ])

    assert reconcile_order_statuses(store, _FlakyBroker()) == 1
    by_id = {o["order_id"]: o for o in store.list_orders()}
    assert by_id["1"]["status"] == "pending"
    assert by_id["2"]["status"] == "filled"


def test_only_terminal_statuses_are_excluded_from_polling(tmp_path):
    store = _store(tmp_path)
    store.record_orders([
        {"order_id": "1", "symbol": "AAPL", "status": "filled"},
        {"order_id": "2", "symbol": "MSFT", "status": "cancelled"},
        {"order_id": "3", "symbol": "GOOG", "status": "rejected"},
    ])
    # No broker entries for any of them — would raise if polled.
    assert reconcile_order_statuses(store, _FakeBroker({})) == 0
