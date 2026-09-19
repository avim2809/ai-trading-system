"""Tests for AlpacaBroker's position-flip order splitting.

Regression coverage for a real incident found running the live Alpaca paper
instance: the engine submits a single order per symbol sized as the full
long<->short delta (e.g. long 3 shares, target -4 -> `sell qty=7`). Alpaca's
API rejects a single sell/buy that would cross a position through zero —
error 40310000 "insufficient qty available for order" — even though the
account has shorting fully enabled (confirmed live via the account API:
shorting_enabled=True, margin multiplier=4x, real short positions already
held). Alpaca requires two sequential orders instead: flatten the existing
position, then open the new opposite side. IBKR has no such restriction
(one order flips fine), which is why only the Alpaca adapter needs this.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("alpaca")

from firm.brokers.alpaca import AlpacaBroker
from firm.brokers.base import BrokerError, OrderRequest
from alpaca.trading.requests import (
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)


class _FakeTradingClient:
    """Fake alpaca-py TradingClient covering only what AlpacaBroker calls.

    ``position_qty=None`` simulates no open position (get_open_position
    raises, matching AlpacaBroker.get_position's real 404-swallowing
    behavior). ``fill_status`` controls what every submitted order (and any
    later get_order_by_id poll) reports back — "filled" for the common case,
    "pending" (never advances) to exercise the flatten-never-fills path.
    """

    def __init__(self, position_qty: float | None, fill_status: str = "filled"):
        self._position_qty = position_qty
        self._fill_status = fill_status
        self.submit_calls: list[Any] = []
        self._orders: dict[str, SimpleNamespace] = {}
        self._n = 0

    def get_open_position(self, symbol: str):
        if self._position_qty is None:
            raise Exception("position does not exist")
        return SimpleNamespace(
            symbol=symbol,
            qty=self._position_qty,
            avg_entry_price=100.0,
            market_value=self._position_qty * 100.0,
            unrealized_pl=0.0,
        )

    def submit_order(self, req):
        self._n += 1
        oid = f"o{self._n}"
        self.submit_calls.append(req)
        filled = self._fill_status == "filled"
        order = SimpleNamespace(
            id=oid,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            filled_qty=req.qty if filled else 0.0,
            filled_avg_price=100.0 if filled else 0.0,
            status=self._fill_status,
            submitted_at=None,
        )
        self._orders[oid] = order
        return order

    def get_order_by_id(self, order_id: str):
        return self._orders[order_id]


def _broker(position_qty: float | None, fill_status: str = "filled") -> tuple[AlpacaBroker, _FakeTradingClient]:
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    client = _FakeTradingClient(position_qty, fill_status)
    broker._trading = client
    return broker, client


class TestNoSplitCases:
    """Orders that don't cross a position through zero must submit exactly
    as they did before this fix — a single order, full requested quantity."""

    def test_no_existing_position_submits_single_order(self):
        broker, client = _broker(position_qty=None)
        status = broker.submit_order(OrderRequest(symbol="AAPL", side="sell", quantity=5))
        assert len(client.submit_calls) == 1
        assert client.submit_calls[0].qty == 5
        assert status.quantity == 5

    def test_reduce_without_crossing_submits_single_order(self):
        # Long 3, sell 2 -> still long 1, never touches zero.
        broker, client = _broker(position_qty=3)
        broker.submit_order(OrderRequest(symbol="AAPL", side="sell", quantity=2))
        assert len(client.submit_calls) == 1
        assert client.submit_calls[0].qty == 2

    def test_exact_flatten_submits_single_order(self):
        # Long 3, sell 3 -> lands exactly at zero, not a flip.
        broker, client = _broker(position_qty=3)
        broker.submit_order(OrderRequest(symbol="AAPL", side="sell", quantity=3))
        assert len(client.submit_calls) == 1
        assert client.submit_calls[0].qty == 3

    def test_same_direction_add_submits_single_order(self):
        broker, client = _broker(position_qty=3)
        broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=2))
        assert len(client.submit_calls) == 1
        assert client.submit_calls[0].qty == 2

    def test_limit_order_flip_is_not_split(self):
        """A limit flip is ill-defined (the two legs would need different
        prices) — documented limitation: it passes through as one order,
        exactly like today, and Alpaca's own rejection (if any) surfaces
        as a plain BrokerError."""
        broker, client = _broker(position_qty=3)
        broker.submit_order(OrderRequest(
            symbol="AAPL", side="sell", quantity=7,
            order_type="limit", limit_price=100.0,
        ))
        assert len(client.submit_calls) == 1
        assert client.submit_calls[0].qty == 7


class TestFlipSplit:
    """The actual fix: a market order that would cross a position through
    zero is split into a flatten leg (closes the existing position exactly)
    followed by an open leg (the remainder, in the new direction)."""

    def test_long_to_short_splits_into_close_then_open(self):
        # Long 3, target -4 -> engine sends sell qty=7.
        broker, client = _broker(position_qty=3)
        status = broker.submit_order(
            OrderRequest(symbol="MSFT", side="sell", quantity=7, client_order_id="c1-MSFT-sell")
        )
        assert len(client.submit_calls) == 2
        assert client.submit_calls[0].qty == 3
        assert client.submit_calls[0].client_order_id == "c1-MSFT-sell-close"
        assert client.submit_calls[1].qty == 4
        assert client.submit_calls[1].client_order_id == "c1-MSFT-sell-open"
        # The returned status represents the resulting net position (the
        # open leg), not the flatten leg.
        assert status.quantity == 4

    def test_short_to_long_splits_symmetrically(self):
        # Short -3, target +4 -> engine sends buy qty=7.
        broker, client = _broker(position_qty=-3)
        status = broker.submit_order(
            OrderRequest(symbol="TSLA", side="buy", quantity=7, client_order_id="c2-TSLA-buy")
        )
        assert len(client.submit_calls) == 2
        assert client.submit_calls[0].qty == 3
        assert client.submit_calls[0].client_order_id == "c2-TSLA-buy-close"
        assert client.submit_calls[1].qty == 4
        assert client.submit_calls[1].client_order_id == "c2-TSLA-buy-open"
        assert status.quantity == 4

    def test_none_client_order_id_does_not_crash(self):
        broker, client = _broker(position_qty=3)
        status = broker.submit_order(OrderRequest(symbol="MSFT", side="sell", quantity=7))
        assert len(client.submit_calls) == 2
        assert client.submit_calls[0].client_order_id is None
        assert client.submit_calls[1].client_order_id is None
        assert status.quantity == 4

    def test_flatten_never_fills_defers_open_leg_to_next_cycle(self, caplog):
        """If the flatten leg doesn't fill within budget (market closed,
        partial fill, halt), the open leg must NOT be submitted — Alpaca
        would still see the pre-flatten qty and reject it with the same
        error. The next cycle recomputes the delta from the now-reduced
        position and submits a clean single order."""
        broker, client = _broker(position_qty=3, fill_status="pending")
        with patch("firm.brokers.alpaca._FLATTEN_MAX_WAIT_SECONDS", 0.05), \
             patch("firm.brokers.alpaca._FLATTEN_POLL_INTERVAL", 0.01), \
             caplog.at_level("WARNING"):
            status = broker.submit_order(
                OrderRequest(symbol="MSFT", side="sell", quantity=7, client_order_id="c3-MSFT-sell")
            )
        # Only the flatten leg was ever submitted.
        assert len(client.submit_calls) == 1
        assert client.submit_calls[0].qty == 3
        assert status.quantity == 3  # the flatten leg's own status
        assert status.status == "pending"
        assert any("did not fully fill" in r.message for r in caplog.records)

    def test_flatten_rejected_returns_rejected_status_without_open_leg(self):
        broker, client = _broker(position_qty=3, fill_status="rejected")
        status = broker.submit_order(
            OrderRequest(symbol="MSFT", side="sell", quantity=7, client_order_id="c4-MSFT-sell")
        )
        assert len(client.submit_calls) == 1
        assert status.status == "rejected"


# ---------------------------------------------------------------------------
# New order types: stop / stop_limit / trailing_stop, and extended_hours.
# ---------------------------------------------------------------------------

def _capturing_broker() -> tuple[AlpacaBroker, _FakeTradingClient]:
    """A broker with no existing position, so flip-split logic never
    engages -- every order here goes straight to _submit_single()."""
    return _broker(position_qty=None)


class TestNativeOrderTypes:
    """Coverage for the native stop/stop-limit/trailing-stop branches added
    to _submit_single -- previously only market/limit were ever built,
    even though alpaca-py and the broker's own connected account both
    support these order types natively."""

    def test_stop_order_builds_stop_order_request(self):
        broker, client = _capturing_broker()
        broker.submit_order(
            OrderRequest(symbol="AAPL", side="sell", quantity=10, order_type="stop", stop_price=95.0)
        )
        req = client.submit_calls[0]
        assert isinstance(req, StopOrderRequest)
        assert req.stop_price == 95.0
        assert req.qty == 10

    def test_stop_order_missing_stop_price_raises(self):
        broker, client = _capturing_broker()
        with pytest.raises(BrokerError, match="stop_price"):
            broker.submit_order(OrderRequest(symbol="AAPL", side="sell", quantity=10, order_type="stop"))
        assert client.submit_calls == []

    def test_stop_limit_order_builds_stop_limit_order_request(self):
        broker, client = _capturing_broker()
        broker.submit_order(
            OrderRequest(
                symbol="AAPL", side="sell", quantity=10, order_type="stop_limit",
                stop_price=95.0, limit_price=94.0,
            )
        )
        req = client.submit_calls[0]
        assert isinstance(req, StopLimitOrderRequest)
        assert req.stop_price == 95.0
        assert req.limit_price == 94.0

    def test_stop_limit_order_missing_prices_raises(self):
        broker, client = _capturing_broker()
        with pytest.raises(BrokerError, match="stop_price and limit_price"):
            broker.submit_order(
                OrderRequest(symbol="AAPL", side="sell", quantity=10, order_type="stop_limit", stop_price=95.0)
            )

    def test_trailing_stop_with_trail_percent(self):
        broker, client = _capturing_broker()
        broker.submit_order(
            OrderRequest(
                symbol="AAPL", side="sell", quantity=10, order_type="trailing_stop", trail_percent=5.0,
            )
        )
        req = client.submit_calls[0]
        assert isinstance(req, TrailingStopOrderRequest)
        assert req.trail_percent == 5.0
        assert req.trail_price is None

    def test_trailing_stop_with_trail_amount_uses_trail_price(self):
        broker, client = _capturing_broker()
        broker.submit_order(
            OrderRequest(
                symbol="AAPL", side="sell", quantity=10, order_type="trailing_stop", trail_amount=2.5,
            )
        )
        req = client.submit_calls[0]
        assert isinstance(req, TrailingStopOrderRequest)
        assert req.trail_price == 2.5
        assert req.trail_percent is None

    def test_trailing_stop_missing_both_raises(self):
        broker, client = _capturing_broker()
        with pytest.raises(BrokerError, match="trail_percent or trail_amount"):
            broker.submit_order(
                OrderRequest(symbol="AAPL", side="sell", quantity=10, order_type="trailing_stop")
            )

    def test_extended_hours_passed_through_for_limit_orders(self):
        broker, client = _capturing_broker()
        broker.submit_order(
            OrderRequest(
                symbol="AAPL", side="buy", quantity=10, order_type="limit",
                limit_price=100.0, extended_hours=True,
            )
        )
        req = client.submit_calls[0]
        assert req.extended_hours is True

    def test_extended_hours_ignored_with_warning_for_market_orders(self, caplog):
        broker, client = _capturing_broker()
        with caplog.at_level("WARNING"):
            broker.submit_order(
                OrderRequest(symbol="AAPL", side="buy", quantity=10, order_type="market", extended_hours=True)
            )
        req = client.submit_calls[0]
        assert not req.extended_hours
        assert any("extended_hours" in r.message for r in caplog.records)

    def test_extended_hours_ignored_with_warning_for_stop_orders(self, caplog):
        broker, client = _capturing_broker()
        with caplog.at_level("WARNING"):
            broker.submit_order(
                OrderRequest(
                    symbol="AAPL", side="sell", quantity=10, order_type="stop",
                    stop_price=95.0, extended_hours=True,
                )
            )
        req = client.submit_calls[0]
        assert not req.extended_hours
        assert any("extended_hours" in r.message for r in caplog.records)

    def test_extended_hours_false_is_unaffected_and_silent(self, caplog):
        broker, client = _capturing_broker()
        with caplog.at_level("WARNING"):
            broker.submit_order(
                OrderRequest(symbol="AAPL", side="buy", quantity=10, order_type="market")
            )
        assert not any("extended_hours" in r.message for r in caplog.records)
