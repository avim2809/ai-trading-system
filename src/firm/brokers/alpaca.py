"""Alpaca broker adapter using the alpaca-py SDK."""

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
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        LimitOrderRequest,
        MarketOrderRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False

_TIF_MAP = {
    "day": "day",
    "gtc": "gtc",
    "ioc": "ioc",
    "fok": "fok",
}


def _require_alpaca() -> None:
    if not _HAS_ALPACA:
        raise ImportError(
            "alpaca-py is not installed. Install the live extra: "
            "pip install 'firm[live]' or pip install alpaca-py"
        )


class AlpacaBroker(Broker):
    """Alpaca Markets broker adapter (paper and live)."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ) -> None:
        _require_alpaca()
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._trading: TradingClient | None = None
        self._data: StockHistoricalDataClient | None = None

    def connect(self) -> None:
        _require_alpaca()
        self._trading = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )
        self._data = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        acct = self._trading.get_account()
        log.info(
            "Connected to Alpaca (%s) – equity $%s",
            "paper" if self._paper else "live",
            acct.equity,
        )

    def disconnect(self) -> None:
        self._trading = None
        self._data = None
        log.info("Disconnected from Alpaca")

    def is_connected(self) -> bool:
        if self._trading is None:
            return False
        try:
            self._trading.get_account()
            return True
        except Exception:
            return False

    def _ensure_connected(self) -> TradingClient:
        if self._trading is None:
            raise BrokerError("Not connected to Alpaca – call connect() first")
        return self._trading

    def get_account(self) -> dict[str, Any]:
        client = self._ensure_connected()
        acct = client.get_account()
        return {
            "cash": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "currency": acct.currency,
        }

    def get_positions(self) -> list[BrokerPosition]:
        client = self._ensure_connected()
        positions = client.get_all_positions()
        return [
            BrokerPosition(
                symbol=p.symbol,
                quantity=float(p.qty),
                avg_cost=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
            )
            for p in positions
        ]

    def get_position(self, symbol: str) -> BrokerPosition | None:
        client = self._ensure_connected()
        try:
            p = client.get_open_position(symbol)
            return BrokerPosition(
                symbol=p.symbol,
                quantity=float(p.qty),
                avg_cost=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
            )
        except Exception:
            return None

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        client = self._ensure_connected()
        tif = getattr(TimeInForce, _TIF_MAP.get(order.time_in_force, "day").upper(), TimeInForce.DAY)

        try:
            if order.order_type == "limit":
                if order.limit_price is None:
                    raise BrokerError("limit_price required for limit orders")
                req = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                    client_order_id=order.client_order_id,
                )
            else:
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                    time_in_force=tif,
                    client_order_id=order.client_order_id,
                )

            result = client.submit_order(req)
            return self._map_order(result)

        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"Order submission failed: {exc}") from exc

    def cancel_order(self, order_id: str) -> bool:
        client = self._ensure_connected()
        try:
            client.cancel_order_by_id(order_id)
            return True
        except Exception:
            log.warning("Failed to cancel order %s", order_id, exc_info=True)
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        client = self._ensure_connected()
        order = client.get_order_by_id(order_id)
        return self._map_order(order)

    def get_open_orders(self) -> list[OrderStatus]:
        client = self._ensure_connected()
        try:
            orders = client.get_orders(filter=QueryOrderStatus.OPEN)
        except Exception:
            orders = client.get_orders()
        return [self._map_order(o) for o in orders]

    def get_current_price(self, symbol: str) -> float:
        if self._data is None:
            raise BrokerError("Data client not initialized – call connect() first")
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self._data.get_stock_latest_quote(request)
        quote = quotes.get(symbol)
        if quote is None:
            raise BrokerError(f"No quote for {symbol}")
        mid = (float(quote.ask_price) + float(quote.bid_price)) / 2
        return mid if mid > 0 else float(quote.ask_price or quote.bid_price)

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        if self._data is None:
            raise BrokerError("Data client not initialized – call connect() first")
        request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = self._data.get_stock_latest_quote(request)
        result: dict[str, float] = {}
        for sym in symbols:
            quote = quotes.get(sym)
            if quote is not None:
                mid = (float(quote.ask_price) + float(quote.bid_price)) / 2
                result[sym] = mid if mid > 0 else float(quote.ask_price or quote.bid_price)
        return result

    def is_market_open(self) -> bool:
        client = self._ensure_connected()
        clock = client.get_clock()
        return clock.is_open

    @staticmethod
    def _map_order(order: Any) -> OrderStatus:
        status_map = {
            "new": "pending",
            "accepted": "pending",
            "pending_new": "pending",
            "partially_filled": "partial",
            "filled": "filled",
            "canceled": "cancelled",
            "cancelled": "cancelled",
            "expired": "cancelled",
            "rejected": "rejected",
            "pending_cancel": "pending",
            "pending_replace": "pending",
        }
        raw_status = str(getattr(order, "status", "pending")).lower()
        mapped = status_map.get(raw_status, "pending")
        return OrderStatus(
            order_id=str(order.id),
            symbol=order.symbol,
            side=str(order.side).lower().replace("ordersidé.", ""),
            quantity=float(order.qty) if order.qty else 0.0,
            filled_quantity=float(order.filled_qty) if order.filled_qty else 0.0,
            avg_fill_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
            status=mapped,
            timestamp=order.submitted_at or datetime.utcnow(),
        )
