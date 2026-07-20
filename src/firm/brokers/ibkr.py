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
    from ib_async import IB, Stock, LimitOrder, MarketOrder

    _HAS_IB = True
except ImportError:
    _HAS_IB = False


def _require_ib() -> None:
    if not _HAS_IB:
        raise ImportError(
            "ib_async is not installed. Install the live extra: "
            "pip install 'firm[live]' or pip install ib_async"
        )


class IBKRBroker(Broker):
    """Interactive Brokers adapter (TWS / IB Gateway)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        market_data_type: int = 3,
    ) -> None:
        _require_ib()
        self._host = host
        self._port = port
        self._client_id = client_id
        # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen. Defaults to delayed
        # since most accounts (paper included) have no live-data subscription
        # — without setting this, reqTickers silently returns NaN for every
        # field instead of an error, which looks like "no price available".
        self._market_data_type = market_data_type
        self._ib: IB | None = None

    def connect(self) -> None:
        _require_ib()
        self._ib = IB()
        try:
            self._ib.connect(self._host, self._port, clientId=self._client_id)
            self._ib.reqMarketDataType(self._market_data_type)
            log.info(
                "Connected to IBKR at %s:%d (client %d, market_data_type=%d)",
                self._host,
                self._port,
                self._client_id,
                self._market_data_type,
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
                    log.warning(
                        "Could not parse IBKR account field %s=%r as float; omitting",
                        v.tag, v.value,
                    )
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

        # Forward the idempotency token as the IBKR order reference.
        if order.client_order_id:
            ib_order.orderRef = order.client_order_id

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

    @staticmethod
    def _resolve_price(ticker, symbol: str) -> float:
        """Best available price from a ticker, falling back gracefully.

        Outside active trading hours (or without a live data subscription),
        IBKR's bid/ask come back as the empty sentinel ``-1`` (midpoint()
        computes as NaN from that) and ``last`` defaults to ``0.0`` — a real
        float, not NaN, so a NaN-only check silently accepts it as a fake
        zero price. Falls through to the previous session's ``close``
        (usually still populated) before giving up, since a stale-but-real
        price is far better than a fabricated zero feeding into order sizing.
        """
        mid = ticker.midpoint()
        if mid == mid and mid > 0:
            return mid
        if ticker.last == ticker.last and ticker.last > 0:
            return ticker.last
        if ticker.close == ticker.close and ticker.close > 0:
            log.warning(
                "No live quote for %s (market likely closed); using previous close %.2f",
                symbol, ticker.close,
            )
            return ticker.close
        log.warning("No midpoint, last, or close price available for %s; returning 0.0", symbol)
        return 0.0

    def get_current_price(self, symbol: str) -> float:
        ib = self._ensure_connected()
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        [ticker] = ib.reqTickers(contract)
        return self._resolve_price(ticker, symbol)

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        ib = self._ensure_connected()
        contracts = [Stock(s, "SMART", "USD") for s in symbols]
        ib.qualifyContracts(*contracts)
        tickers = ib.reqTickers(*contracts)
        return {
            contract.symbol: self._resolve_price(ticker, contract.symbol)
            for contract, ticker in zip(contracts, tickers)
        }

    def is_market_open(self) -> bool:
        """True during the regular trading session (not pre/post-market).

        IBKR doesn't expose a simple is_open flag, so this checks via
        contract details: ``liquidHours`` is the regular session (e.g.
        9:30am-4pm ET) — deliberately not ``tradingHours``, which includes
        the extended pre/post-market window most strategies here aren't
        designed to trade in. Both fields are expressed in the contract's
        own exchange timezone (``timeZoneId``, e.g. "US/Eastern"), which
        must be converted to before comparing — comparing against a raw
        UTC clock reading (the previous implementation) silently produces
        wrong answers whenever UTC and the exchange timezone differ, which
        is always.
        """
        ib = self._ensure_connected()
        contract = Stock("SPY", "SMART", "USD")
        ib.qualifyContracts(contract)
        details = ib.reqContractDetails(contract)
        if not details:
            return False
        cd = details[0]
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo(cd.timeZoneId))
        except Exception:
            log.warning(
                "Could not resolve exchange timezone %r; falling back to UTC "
                "for market-hours check", cd.timeZoneId, exc_info=True,
            )
            now = datetime.utcnow()
        now_str = now.strftime("%Y%m%d:%H%M")
        for segment in cd.liquidHours.split(";"):
            if "-" not in segment:
                continue  # e.g. "20260725:CLOSED" (holiday) — no session that day
            parts = segment.split("-")
            if len(parts) == 2 and parts[0] <= now_str <= parts[1]:
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
            "ApiCancelled": "cancelled",
            "Inactive": "rejected",
            # Not a native IBKR orderStatus value — ib_async sets this when the
            # API rejects the order outright (e.g. "Read-Only mode", bad
            # contract). Must not fall into the unknown-status default below,
            # or a rejected order gets reported as merely "pending" forever.
            "ValidationError": "rejected",
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
            # Fail safe: an unrecognised status must never default to
            # "pending" — that would mask a real (if unanticipated) rejection
            # as an order still quietly in flight.
            mapped = status_map.get(raw_status, "rejected")

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
