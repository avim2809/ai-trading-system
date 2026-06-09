"""Interactive Brokers adapter using the ib_insync library."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from firm.brokers.base import (
    Broker,
    BrokerError,
    BrokerPosition,
    OrderRequest,
    OrderStatus,
)

log = logging.getLogger(__name__)

try:
    from ib_insync import IB, Stock, LimitOrder, MarketOrder, util

    _HAS_IB = True
except ImportError:
    _HAS_IB = False


def _require_ib() -> None:
    if not _HAS_IB:
        raise ImportError(
            "ib_insync is not installed. Install the live extra: "
            "pip install 'firm[live]' or pip install ib_insync"
        )


class IBKRBroker(Broker):
    """Interactive Brokers adapter (TWS / IB Gateway)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ) -> None:
        _require_ib()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib: IB | None = None

    def connect(self) -> None:
        _require_ib()
        self._ib = IB()
        try:
            self._ib.connect(self._host, self._port, clientId=self._client_id)
            log.info(
                "Connected to IBKR at %s:%d (client %d)",
                self._host,
                self._port,
                self._client_id,
            )
        except Exception as exc:
            self._ib = None
            raise BrokerError(f"Failed to connect to IBKR: {exc}") from exc

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None
            log.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    def _ensure_connected(self) -> IB:
        if self._ib is None or not self._ib.isConnected():
            raise BrokerError("Not connected to IBKR – call connect() first")
        return self._ib

    def get_account(self) -> dict[str, Any]:
        ib = self._ensure_connected()
        ib.reqAccountSummary()
        values = ib.accountSummary()
        result: dict[str, float] = {}
        key_map = {
            "TotalCashValue": "cash",
            "NetLiquidation": "equity",
            "BuyingPower": "buying_power",
            "GrossPositionValue": "portfolio_value",
        }
        for v in values:
            mapped = key_map.get(v.tag)
            if mapped:
                try:
                    result[mapped] = float(v.value)
                except (ValueError, TypeError):
                    pass
        for k in ("cash", "equity", "buying_power"):
            result.setdefault(k, 0.0)
        return result

    def get_positions(self) -> list[BrokerPosition]:
        ib = self._ensure_connected()
        positions = ib.positions()
        result: list[BrokerPosition] = []
        for p in positions:
            mkt_val = float(p.position) * float(p.avgCost)
            result.append(
                BrokerPosition(
                    symbol=p.contract.symbol,
                    quantity=float(p.position),
                    avg_cost=float(p.avgCost),
                    market_value=mkt_val,
                    unrealized_pnl=0.0,
                )
            )
        return result

    def get_position(self, symbol: str) -> BrokerPosition | None:
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        ib = self._ensure_connected()
        contract = Stock(order.symbol, "SMART", "USD")
        ib.qualifyContracts(contract)

        qty = order.quantity if order.side == "buy" else -order.quantity
        action = "BUY" if order.side == "buy" else "SELL"

        if order.order_type == "limit":
            if order.limit_price is None:
                raise BrokerError("limit_price required for limit orders")
            ib_order = LimitOrder(action, abs(qty), order.limit_price)
        else:
            ib_order = MarketOrder(action, abs(qty))

        trade = ib.placeOrder(contract, ib_order)
        ib.sleep(0.5)

        return self._map_trade(trade)

    def cancel_order(self, order_id: str) -> bool:
        ib = self._ensure_connected()
        for trade in ib.openTrades():
            if str(trade.order.orderId) == order_id:
                ib.cancelOrder(trade.order)
                return True
        return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        ib = self._ensure_connected()
        for trade in ib.trades():
            if str(trade.order.orderId) == order_id:
                return self._map_trade(trade)
        raise BrokerError(f"Order {order_id} not found")

    def get_open_orders(self) -> list[OrderStatus]:
        ib = self._ensure_connected()
        return [self._map_trade(t) for t in ib.openTrades()]

    def get_current_price(self, symbol: str) -> float:
        ib = self._ensure_connected()
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        [ticker] = ib.reqTickers(contract)
        mid = ticker.midpoint()
        if mid != mid:  # NaN check
            return ticker.last if ticker.last == ticker.last else 0.0
        return mid

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        ib = self._ensure_connected()
        contracts = [Stock(s, "SMART", "USD") for s in symbols]
        ib.qualifyContracts(*contracts)
        tickers = ib.reqTickers(*contracts)
        result: dict[str, float] = {}
        for contract, ticker in zip(contracts, tickers):
            mid = ticker.midpoint()
            if mid != mid:
                mid = ticker.last if ticker.last == ticker.last else 0.0
            result[contract.symbol] = mid
        return result

    def is_market_open(self) -> bool:
        ib = self._ensure_connected()
        # IBKR doesn't expose a simple is_open; check via contract details
        contract = Stock("SPY", "SMART", "USD")
        ib.qualifyContracts(contract)
        details = ib.reqContractDetails(contract)
        if details:
            hours = details[0].tradingHours
            now = datetime.utcnow().strftime("%Y%m%d:%H%M")
            for segment in hours.split(";"):
                if "-" in segment:
                    parts = segment.split("-")
                    if len(parts) == 2 and parts[0] <= now <= parts[1]:
                        return True
        return False

    @staticmethod
    def _map_trade(trade: Any) -> OrderStatus:
        status_map = {
            "Submitted": "pending",
            "PreSubmitted": "pending",
            "PendingSubmit": "pending",
            "PendingCancel": "pending",
            "Filled": "filled",
            "Cancelled": "cancelled",
            "Inactive": "rejected",
        }
        order = trade.order
        fill_qty = sum(f.execution.shares for f in trade.fills) if trade.fills else 0.0
        avg_price = 0.0
        if trade.fills:
            total_cost = sum(f.execution.shares * f.execution.price for f in trade.fills)
            avg_price = total_cost / fill_qty if fill_qty else 0.0

        raw_status = trade.orderStatus.status if trade.orderStatus else "Submitted"
        if fill_qty > 0 and fill_qty < float(order.totalQuantity):
            mapped = "partial"
        else:
            mapped = status_map.get(raw_status, "pending")

        return OrderStatus(
            order_id=str(order.orderId),
            symbol=trade.contract.symbol,
            side="buy" if order.action == "BUY" else "sell",
            quantity=float(order.totalQuantity),
            filled_quantity=fill_qty,
            avg_fill_price=avg_price,
            status=mapped,
            timestamp=datetime.utcnow(),
        )
