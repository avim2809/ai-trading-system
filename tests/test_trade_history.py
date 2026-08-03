"""Tests for persisted live order/cycle history."""

from __future__ import annotations

from firm.live.trade_history import TradeHistoryStore


def test_trade_history_persists_orders_and_cycles(tmp_path):
    orders_path = tmp_path / "orders.json"
    cycles_path = tmp_path / "cycles.json"
    store = TradeHistoryStore(orders_path=orders_path, cycles_path=cycles_path)

    store.record_cycle({"cycle_id": 1, "orders_submitted": 0})
    store.record_orders(
        [{"order_id": "o1", "symbol": "AAPL", "side": "buy", "quantity": 1, "status": "submitted"}],
        cycle_id=1,
        source="cycle",
    )

    reloaded = TradeHistoryStore(orders_path=orders_path, cycles_path=cycles_path)
    assert len(reloaded.list_cycles()) == 1
    assert reloaded.list_cycles()[0]["cycle_id"] == 1
    assert len(reloaded.list_orders()) == 1
    assert reloaded.list_orders()[0]["symbol"] == "AAPL"
    assert reloaded.list_orders()[0]["cycle_id"] == 1


def test_trade_history_clear_all(tmp_path):
    store = TradeHistoryStore(
        orders_path=tmp_path / "orders.json",
        cycles_path=tmp_path / "cycles.json",
    )
    store.record_cycle({"cycle_id": 1})
    store.record_orders([{"order_id": "o1", "symbol": "MSFT"}])

    cleared = store.clear_all()
    assert cleared == {"orders": 1, "cycles": 1}
    assert store.list_orders() == []
    assert store.list_cycles() == []


def test_update_order_status_corrects_a_stale_pending_record(tmp_path):
    store = TradeHistoryStore(
        orders_path=tmp_path / "orders.json",
        cycles_path=tmp_path / "cycles.json",
    )
    store.record_orders([{
        "order_id": "o1", "symbol": "AAPL", "side": "buy", "quantity": 10,
        "filled_quantity": 0.0, "avg_fill_price": 0.0, "status": "pending",
    }])

    updated = store.update_order_status(
        "o1", status="filled", filled_quantity=10.0, avg_fill_price=151.25,
    )

    assert updated is True
    record = store.list_orders()[0]
    assert record["status"] == "filled"
    assert record["filled_quantity"] == 10.0
    assert record["avg_fill_price"] == 151.25


def test_update_order_status_returns_false_for_unknown_order_id(tmp_path):
    store = TradeHistoryStore(
        orders_path=tmp_path / "orders.json",
        cycles_path=tmp_path / "cycles.json",
    )
    store.record_orders([{"order_id": "o1", "symbol": "AAPL", "status": "pending"}])

    assert store.update_order_status(
        "does-not-exist", status="filled", filled_quantity=1.0, avg_fill_price=1.0,
    ) is False


def test_update_order_status_persists_across_reload(tmp_path):
    orders_path = tmp_path / "orders.json"
    cycles_path = tmp_path / "cycles.json"
    store = TradeHistoryStore(orders_path=orders_path, cycles_path=cycles_path)
    store.record_orders([{"order_id": "o1", "symbol": "AAPL", "status": "pending"}])
    store.update_order_status("o1", status="cancelled", filled_quantity=0.0, avg_fill_price=0.0)

    reloaded = TradeHistoryStore(orders_path=orders_path, cycles_path=cycles_path)
    assert reloaded.list_orders()[0]["status"] == "cancelled"
