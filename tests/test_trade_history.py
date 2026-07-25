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
