"""Alpaca broker adapter using the alpaca-py SDK."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from firm.brokers.base import (
    Broker,
    BrokerError,
    BrokerPosition,
    MarketHoursStatus,
    OrderRequest,
    OrderStatus,
)
from firm.time_utils import utcnow

log = logging.getLogger(__name__)

try:
    from alpaca.common.exceptions import APIError
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
        StopLimitOrderRequest,
        StopOrderRequest,
        TrailingStopOrderRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False

# Alpaca's generic "insufficient qty/wash trade" error code -- see both
# _plan_flip_split's docstring (same code, "insufficient qty available",
# handled by splitting) and _wash_trade_conflict_order_id below (same code,
# "potential wash trade detected", handled by cancel-and-retry).
_ERR_CODE_ORDER_CONFLICT = 40310000

# Alpaca's 404-backed "position does not exist" error -- the expected,
# common response from GET /v2/positions/{symbol} whenever the broker
# doesn't currently hold that symbol (see get_position / _is_no_position_error).
_ERR_CODE_NO_POSITION = 40410000

_TIF_MAP = {
    "day": "day",
    "gtc": "gtc",
    "ioc": "ioc",
    "fok": "fok",
}

# Bound on waiting for the flatten leg of a position-flip order to fill
# before submitting the opening leg (see AlpacaBroker.submit_order). Confirmed
# live: Alpaca rejects a single sell/buy that would cross a position through
# zero (e.g. sell 7 against a long 3 — "insufficient qty available", error
# 40310000) and requires two sequential orders instead. Both legs are the
# *same side*, so if the open leg were sent before the flatten leg actually
# fills, Alpaca would still see the pre-flatten qty and reject it the same
# way. A market order fills in well under a second during regular trading
# hours, so this is generous headroom, not a routine wait.
_FLATTEN_POLL_INTERVAL = 0.5
_FLATTEN_MAX_WAIT_SECONDS = 8.0


def _require_alpaca() -> None:
    if not _HAS_ALPACA:
        raise ImportError(
            "alpaca-py is not installed. Install the live extra: "
            "pip install 'firm[live]' or pip install alpaca-py"
        )


def _wash_trade_conflict_order_id(exc: Exception) -> str | None:
    """Return the blocking ``existing_order_id`` if *exc* is Alpaca's
    wash-trade rejection (error 40310000, an opposite-side order already
    resting for the same symbol), else ``None``.

    Distinguishing on the message text (not just the code) matters: code
    40310000 is also used for the unrelated "insufficient qty available"
    rejection that :meth:`AlpacaBroker._plan_flip_split` already handles by
    splitting the order -- that case has no ``existing_order_id`` to cancel
    and must not be treated as a wash trade.
    """
    try:
        if exc.code != _ERR_CODE_ORDER_CONFLICT:  # type: ignore[attr-defined]
            return None
        payload = json.loads(str(exc))
    except Exception:
        # exc.code itself re-parses the raw error text (see alpaca-py's
        # APIError.code) and isn't guaranteed to exist on every exception
        # this could see in principle -- degrade to "not a wash trade"
        # rather than let a malformed/unexpected payload block the
        # original error from propagating.
        return None
    if not isinstance(payload, dict):
        return None
    order_id = payload.get("existing_order_id")
    return order_id if isinstance(order_id, str) and order_id else None


def _is_no_position_error(exc: Exception) -> bool:
    """True if *exc* is Alpaca's expected "no open position for this
    symbol" 404 (error 40410000, ``GET /v2/positions/{symbol}``) -- the
    normal, common response whenever a market order targets a symbol the
    broker doesn't currently hold.

    Confirmed live (2026-09-23, Alpaca paper instance): every one of that
    day's ~16 occurrences (across BKNG/VLO/IWM/META/NEM/NKE/PCG/PLD/SPY/UNH)
    came from :meth:`AlpacaBroker._plan_flip_split`, which calls
    ``get_position`` on *every* market order specifically to check whether
    the order would flip a position through zero -- a symbol the broker
    doesn't hold yet trivially "isn't a flip", which is exactly what a 404
    here means. Cross-checked against ``order_history.json``: the very
    next order for each of those symbols filled normally that same cycle
    -- none were stuck failed. Any *other* exception (auth, rate limit,
    network, a malformed response, ...) is a genuine anomaly that looks
    identical at the call site and should stay loud, not be folded into
    this benign case.
    """
    try:
        return exc.code == _ERR_CODE_NO_POSITION  # type: ignore[attr-defined]
    except Exception:
        return False


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
            log.warning("Alpaca connectivity check failed", exc_info=True)
            return False

    def health_check(self) -> bool:
        """No-op-ish liveness probe for a stateless REST client.

        Alpaca has no persistent socket that can go half-open, so — unlike
        IBKR — there is nothing here for a proactive per-cycle check to
        repair; a pure local check is correct and avoids an extra
        ``get_account()`` REST round-trip every cycle (the base default
        would call :meth:`is_connected`, which hits the network). Genuine
        Alpaca outages still surface through the real REST calls in
        ``refresh()``/reconciliation and are handled by the existing
        reactive reconnect path.
        """
        return self._trading is not None and self._data is not None

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
        except Exception as exc:
            if _is_no_position_error(exc):
                # Expected/benign (see _is_no_position_error) -- log
                # briefly with no traceback so it doesn't read as a crash
                # when scanning logs for "ERROR"/"Traceback" (this used to
                # log.info(..., exc_info=True) unconditionally, which
                # dumped a full stack trace for the single most common
                # case: opening a brand-new position).
                log.info("get_position(%s): no open position at Alpaca", symbol)
            else:
                # Genuinely unexpected (auth, rate limit, network, a
                # malformed response, ...) -- this is the case the old
                # blanket handling couldn't distinguish from the benign
                # 404 above. Surface it loudly, with the full traceback.
                log.warning(
                    "get_position(%s) failed unexpectedly", symbol, exc_info=True,
                )
            return None

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        """Submit *order*, transparently splitting it into two legs when it
        would flip a position's sign (long -> short or short -> long).

        Confirmed live: Alpaca's paper account has shorting enabled
        (margin multiplier 4x, and it already holds real short positions) —
        this is not a permissions gap. The rejection is Alpaca's order-level
        rule that a single sell/buy cannot reduce a position past zero into
        the opposite side; e.g. long 3 shares + a target of short 4 means
        the engine submits ``sell qty=7``, and Alpaca rejects the 4 that
        would open the short with error 40310000 ("insufficient qty
        available"), even though shorting itself is fully permitted. See
        :meth:`_plan_flip_split`.
        """
        split = None
        if order.order_type == "market":
            split = self._plan_flip_split(order)

        if split is None:
            return self._submit_single(
                order, qty=order.quantity, client_order_id=order.client_order_id,
            )

        close_qty, open_qty = split
        close_coid = self._suffix_coid(order.client_order_id, "close")
        log.info(
            "Flip %s: flattening %.0f share(s) first (order_id suffix=%s) "
            "before opening %.0f in the new direction",
            order.symbol, close_qty, close_coid, open_qty,
        )
        close_status = self._submit_single(order, qty=close_qty, client_order_id=close_coid)
        log.info(
            "Flip %s: flatten leg %s submitted -> %s",
            order.symbol, close_status.order_id, close_status.status,
        )

        if not self._await_fill(close_status.order_id):
            log.warning(
                "Flip %s: flatten leg %s did not fully fill within %.0fs — "
                "deferring the open leg to next cycle rather than risk "
                "re-triggering the same rejection or over-shorting",
                order.symbol, close_status.order_id, _FLATTEN_MAX_WAIT_SECONDS,
            )
            return close_status

        open_coid = self._suffix_coid(order.client_order_id, "open")
        open_status = self._submit_single(order, qty=open_qty, client_order_id=open_coid)
        log.info(
            "Flip %s: open leg %s submitted -> %s",
            order.symbol, open_status.order_id, open_status.status,
        )
        return open_status

    def _plan_flip_split(self, order: OrderRequest) -> tuple[int, int] | None:
        """Return ``(close_qty, open_qty)`` if *order* would flip the
        symbol's position through zero, else ``None`` (submit as one order,
        unchanged from today's behavior).

        Reads the current signed position directly rather than relying on
        the engine's own book, since :meth:`get_position` already reflects
        real broker state (and already swallows read failures to ``None``,
        which degrades safely here to "not a flip" — the single-order path,
        exactly today's behavior, rather than risking a wrong split from a
        stale read).
        """
        pos = self.get_position(order.symbol)
        current = pos.quantity if pos is not None else 0.0
        if current == 0.0:
            return None

        if order.side == "sell" and current > 0 and order.quantity > current:
            close_qty = current
        elif order.side == "buy" and current < 0 and order.quantity > abs(current):
            close_qty = abs(current)
        else:
            return None

        close_int = int(round(close_qty))
        open_int = int(round(order.quantity - close_qty))
        if close_int <= 0 or open_int <= 0:
            return None
        return close_int, open_int

    @staticmethod
    def _suffix_coid(client_order_id: str | None, suffix: str) -> str | None:
        """Append *suffix* ("close"/"open") to a client_order_id for one leg
        of a split order. ``None`` stays ``None`` — Alpaca auto-generates an
        id in that case, for both legs independently. Re-running the same
        cycle regenerates identical ids, so Alpaca's own idempotency dedupes
        each leg rather than double-submitting."""
        if client_order_id is None:
            return None
        return f"{client_order_id}-{suffix}"

    def _await_fill(self, order_id: str) -> bool:
        """Bounded poll for *order_id* to reach a terminal ``filled`` state.

        Both legs of a flip are the *same side*, so the open leg must not be
        submitted until the flatten leg has actually settled — otherwise
        Alpaca still sees the pre-flatten quantity and rejects the open leg
        with the identical error. Mirrors IBKRBroker's own bounded
        post-submit poll (`_wait_for_order_resolution`) in spirit: fail safe
        (``False``) rather than hang, on timeout, rejection/cancellation, or
        a polling error.
        """
        deadline = time.monotonic() + _FLATTEN_MAX_WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                status = self.get_order_status(order_id)
            except Exception:
                log.warning(
                    "Could not poll flatten-leg order %s status", order_id, exc_info=True,
                )
                return False
            if status.status == "filled":
                return True
            if status.status in ("rejected", "cancelled"):
                return False
            time.sleep(_FLATTEN_POLL_INTERVAL)
        return False

    def _extended_hours_kwarg(self, order: OrderRequest) -> dict[str, bool]:
        """Return ``{"extended_hours": True}`` when *order* both requested it
        and its order type actually supports it, else ``{}``.

        Alpaca's API only honors ``extended_hours`` on a regular limit order
        (day TIF) — market, stop, stop-limit, and trailing-stop orders are
        rejected (or silently ignored, depending on endpoint version) if it's
        set. Rather than let that surface as a confusing broker-side
        rejection, degrade to a regular-hours order and log why.
        """
        if not order.extended_hours:
            return {}
        if order.order_type == "limit":
            return {"extended_hours": True}
        log.warning(
            "extended_hours requested for %s order on %s, but Alpaca only "
            "supports it on limit orders — submitting as a regular-hours "
            "order instead",
            order.order_type, order.symbol,
        )
        return {}

    def _submit_single(
        self, order: OrderRequest, *, qty: float, client_order_id: str | None,
    ) -> OrderStatus:
        client = self._ensure_connected()
        tif = getattr(TimeInForce, _TIF_MAP.get(order.time_in_force, "day").upper(), TimeInForce.DAY)
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        extended_hours_kwarg = self._extended_hours_kwarg(order)

        try:
            if order.order_type == "limit":
                if order.limit_price is None:
                    raise BrokerError("limit_price required for limit orders")
                req = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                    client_order_id=client_order_id,
                    **extended_hours_kwarg,
                )
            elif order.order_type == "stop":
                if order.stop_price is None:
                    raise BrokerError("stop_price required for stop orders")
                req = StopOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                    client_order_id=client_order_id,
                )
            elif order.order_type == "stop_limit":
                if order.stop_price is None or order.limit_price is None:
                    raise BrokerError("stop_price and limit_price required for stop_limit orders")
                req = StopLimitOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                    client_order_id=client_order_id,
                )
            elif order.order_type == "trailing_stop":
                if order.trail_percent is None and order.trail_amount is None:
                    raise BrokerError(
                        "trail_percent or trail_amount required for trailing_stop orders"
                    )
                trail_kwargs: dict[str, float] = (
                    {"trail_percent": order.trail_percent}
                    if order.trail_percent is not None
                    else {"trail_price": order.trail_amount}
                )
                req = TrailingStopOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    client_order_id=client_order_id,
                    **trail_kwargs,
                )
            else:
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    client_order_id=client_order_id,
                    **extended_hours_kwarg,
                )

            result = self._submit_with_wash_trade_retry(client, req)
            return self._map_order(result)

        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"Order submission failed: {exc}") from exc

    def _submit_with_wash_trade_retry(self, client: "TradingClient", req: Any) -> Any:
        """Submit *req*; if Alpaca rejects it as a wash trade against a
        stale resting order for the same symbol, cancel that order and
        retry once.

        Confirmed live (2026-09-23, Alpaca paper instance): a protective
        stop submitted via ``ExecutionAgent._maybe_submit_protective_order``
        for a freshly-opened BKNG position was never cancelled when that
        *same* cycle's own primary entry order for BKNG was itself rejected
        as a wash trade against it (that method has no broker order-ID
        tracking by design -- see its docstring -- so nothing ever cancels
        a protective stop once placed). The primary order never got
        submitted, but the protective stop stayed resting at the broker,
        and then blocked every later cycle's opposite-side BKNG order too:
        the same ``existing_order_id`` was cited in wash-trade rejections
        at 01:13, 18:30, and 19:30 that day, each one silently dropping
        that cycle's risk-approved rebalance for the symbol.
        Alpaca's error 40310000 with message "potential wash trade
        detected" / reject_reason "opposite side market/stop order exists"
        includes the blocking order's id directly in the payload, so rather
        than let a stale resting order (protective stop or otherwise) jam a
        symbol indefinitely, cancel it and resubmit the current,
        risk-approved order once. If the retry still fails for any reason,
        that's surfaced exactly as before (as a plain BrokerError from the
        caller's except block) -- no unbounded retry loop.
        """
        try:
            return client.submit_order(req)
        except APIError as exc:
            existing_order_id = _wash_trade_conflict_order_id(exc)
            if existing_order_id is None:
                raise
            log.warning(
                "Order for %s rejected as a wash trade against stale "
                "resting order %s -- cancelling it and retrying once",
                req.symbol, existing_order_id,
            )
            if not self.cancel_order(existing_order_id):
                raise
            return client.submit_order(req)

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
            orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        except Exception:
            log.warning(
                "Filtered open-orders request failed; falling back to "
                "unfiltered get_orders() — reconciliation may see stale/closed orders",
                exc_info=True,
            )
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
            else:
                log.warning("No quote returned for %s; omitting from prices", sym)
        return result

    def is_market_open(self) -> bool:
        client = self._ensure_connected()
        clock = client.get_clock()
        return clock.is_open

    def market_hours(self) -> MarketHoursStatus:
        """Alpaca's own clock endpoint already returns next_open/next_close
        directly (holiday-aware server-side) — no separate parsing needed,
        unlike IBKR's own liquidHours-schedule-based override."""
        client = self._ensure_connected()
        clock = client.get_clock()
        return MarketHoursStatus(
            is_open=clock.is_open,
            next_open=clock.next_open,
            next_close=clock.next_close,
        )

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
        # order.status/order.side/order.type are alpaca-py enums; str() on
        # one yields "OrderStatus.FILLED"/"OrderSide.SELL"/"OrderType.STOP",
        # so take the part after the last "." for all three -- same
        # canonical set as IBKRBroker._map_trade's order_type.
        raw_status = str(getattr(order, "status", "pending")).lower().rsplit(".", 1)[-1]
        mapped = status_map.get(raw_status, "pending")
        raw_type = str(getattr(order, "type", "market")).lower()
        return OrderStatus(
            order_id=str(order.id),
            symbol=order.symbol,
            side=str(order.side).lower().rsplit(".", 1)[-1],
            quantity=float(order.qty) if order.qty else 0.0,
            filled_quantity=float(order.filled_qty) if order.filled_qty else 0.0,
            avg_fill_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
            status=mapped,
            timestamp=order.submitted_at or utcnow(),
            order_type=raw_type.rsplit(".", 1)[-1],
        )
