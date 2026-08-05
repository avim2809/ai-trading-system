"""Interactive Brokers adapter using the ib_insync library."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
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

# How long submit_order will keep watching a newly-placed order before
# trusting a non-"Filled" terminal-looking status. See
# IBKRBroker._wait_for_order_resolution's docstring for the real incident
# this guards against (a benign IBKR informational relay transiently
# flipping status to "Cancelled" ~0.5-0.6s before the order proceeds to
# actually fill, confirmed live).
_ORDER_STATUS_POLL_INTERVAL = 0.25
_ORDER_STATUS_MAX_WAIT_SECONDS = 5.0

# ib_async's IB.RequestTimeout defaults to 0 (wait forever) for every
# request routed through IB._run() (qualifyContracts, reqContractDetails,
# reqTickers, ...). Confirmed live in production: a qualifyContracts call
# hung indefinitely — no response ever arrived (plausibly issued right as
# IB Gateway was mid-reconnect to its own backend) — and since every
# IBKRBroker method serialises on the same lock, that one hung call froze
# every other broker operation (positions, account, reconciliation, all
# future cycles) for the rest of the process's life; only a full service
# restart cleared it. Bounding requests here means a stuck call fails loud
# and fast instead of hanging forever.
_IB_REQUEST_TIMEOUT_SECONDS = 20.0

# Bound on acquiring _ib_lock itself, independent of the request timeout
# above — a second layer of defence so that even a hang this module didn't
# anticipate (anything not routed through IB._run(), or a future ib_async
# behavior change) can't cascade into freezing every other broker caller
# forever. Comfortably above _IB_REQUEST_TIMEOUT_SECONDS plus the
# submit_order's own _ORDER_STATUS_MAX_WAIT_SECONDS poll, so a legitimately
# slow (not stuck) call is never mistaken for one that's hung.
_IB_LOCK_ACQUIRE_TIMEOUT_SECONDS = 45.0

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
        self._market_hours_details: Any = None
        # ib_async is bound to the thread that called connect(); all IB I/O
        # must be serialised on that path so API approval handlers and the
        # live-cycle worker thread never call qualifyContracts/placeOrder
        # concurrently (that deadlock hangs the whole firm-api process).
        self._ib_lock = threading.Lock()

    @contextmanager
    def _locked(self):
        """Acquire ``_ib_lock`` with a bound, and translate a request that
        exceeds ``ib.RequestTimeout`` into a ``BrokerError``.

        Without a bound on both of these, a single hung IB call blocks
        forever and — since every ``IBKRBroker`` method serialises on this
        same lock — permanently freezes every other broker operation
        (positions, account, reconciliation, all future cycles) until the
        process is restarted. See ``_IB_REQUEST_TIMEOUT_SECONDS``'s comment
        for the real incident this closes.
        """
        if not self._ib_lock.acquire(timeout=_IB_LOCK_ACQUIRE_TIMEOUT_SECONDS):
            raise BrokerError(
                f"Timed out after {_IB_LOCK_ACQUIRE_TIMEOUT_SECONDS:.0f}s waiting "
                "for the IBKR connection lock — a previous call is likely stuck; "
                "broker access is temporarily unavailable."
            )
        try:
            yield
        except TimeoutError as exc:
            raise BrokerError(f"IBKR request timed out: {exc}") from exc
        finally:
            self._ib_lock.release()

    def connect(self) -> None:
        _require_ib()
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            self._ib = IB()
            self._ib.RequestTimeout = _IB_REQUEST_TIMEOUT_SECONDS
            try:
                self._ib.connect(self._host, self._port, clientId=self._client_id)
                self._ib.reqMarketDataType(self._market_data_type)
                # Subscribe once, here, on the connecting thread. ib_async binds
                # any call that awaits a Future (reqAccountSummary, reqTickers,
                # etc.) to whatever asyncio event loop is current in the calling
                # thread — but get_account() may later be called from a
                # different thread (e.g. FastAPI's anyio threadpool services
                # each request on whichever worker thread is free), which has no
                # event loop at all and crashes. accountSummary() below is a
                # plain cached read (like positions()) that's safe from any
                # thread once the subscription is live, so only subscribe here.
                self._ib.reqAccountSummary()
                contract = Stock("SPY", "SMART", "USD")
                self._ib.qualifyContracts(contract)
                details = self._ib.reqContractDetails(contract)
                self._market_hours_details = details[0] if details else None
                log.info(
                    "Connected to IBKR at %s:%d (client %d, market_data_type=%d)",
                    self._host,
                    self._port,
                    self._client_id,
                    self._market_data_type,
                )
                return
            except Exception as exc:
                last_exc = exc
                if self._ib is not None:
                    try:
                        self._ib.disconnect()
                    except Exception:
                        pass
                self._ib = None
                if attempt < 3:
                    delay = 2.0 * attempt
                    log.warning(
                        "IBKR connect attempt %d/3 failed (%s) — retrying in %.0fs",
                        attempt, exc, delay,
                    )
                    time.sleep(delay)
        raise BrokerError(f"Failed to connect to IBKR: {last_exc}") from last_exc

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

    @contextmanager
    def shared_connection(self):
        """Yield the connected ``IB`` instance under the same lock that
        serialises every other broker call.

        For other components (currently ``IBKRProvider``) that need to reuse
        this connection instead of opening a second one — running two
        independent ``ib_async`` connections on the live-cycle worker thread
        was confirmed live (2026-08-05) to make the broker's own
        ``qualifyContracts`` calls hang until timeout, minutes into a real
        cycle, even though neither connection nor IB Gateway was otherwise
        unhealthy. Reusing one connection requires this lock: without it, a
        data-fetch call on the cycle thread and a reconciliation call on
        APScheduler's own worker thread could hit the same ``IB`` object at
        the same time, which ``_ib_lock`` exists specifically to prevent.
        """
        with self._locked():
            yield self._ensure_connected()

    def get_account(self) -> dict[str, Any]:
        with self._locked():
            return self._get_account_unlocked()

    def _get_account_unlocked(self) -> dict[str, Any]:
        ib = self._ensure_connected()
        # Deliberately not calling reqAccountSummary() here — see the
        # comment in connect(). accountSummary() is a cached read, safe to
        # call from any thread once the subscription made in connect() is
        # live.
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
        with self._locked():
            return self._get_positions_unlocked()

    def _get_positions_unlocked(self) -> list[BrokerPosition]:
        ib = self._ensure_connected()
        items = ib.portfolio()
        result: list[BrokerPosition] = []
        for item in items:
            result.append(
                BrokerPosition(
                    symbol=item.contract.symbol,
                    quantity=float(item.position),
                    avg_cost=float(item.averageCost),
                    market_value=float(item.marketValue),
                    unrealized_pnl=float(item.unrealizedPNL),
                )
            )
        return result

    def get_position(self, symbol: str) -> BrokerPosition | None:
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        with self._locked():
            ib = self._ensure_connected()
            contract = Stock(order.symbol, "SMART", "USD")
            ib.qualifyContracts(contract)

            qty = int(round(abs(order.quantity)))
            if qty <= 0:
                raise BrokerError(f"invalid quantity for {order.symbol}: {order.quantity!r}")
            action = "BUY" if order.side == "buy" else "SELL"

            if order.order_type == "limit":
                if order.limit_price is None:
                    raise BrokerError("limit_price required for limit orders")
                ib_order = LimitOrder(action, qty, order.limit_price)
            else:
                ib_order = MarketOrder(action, qty)

            if order.client_order_id:
                ib_order.orderRef = order.client_order_id

            trade = ib.placeOrder(contract, ib_order)
            self._wait_for_order_resolution(ib, trade)

            return self._map_trade(trade)

    @staticmethod
    def _wait_for_order_resolution(ib: Any, trade: Any) -> None:
        """Wait for *trade* to reach a status that will actually stick,
        instead of grabbing a snapshot after one fixed sleep.

        Real incident (found comparing a live dashboard against the real
        IBKR paper account — the dashboard was showing stale positions):
        IBKR relays some benign informational messages (e.g. errorCode
        10349, "Order TIF was set to DAY based on order preset") through
        the same channel real cancellations use. ib_async surfaces this as
        a transient flip of ``orderStatus.status`` to "Cancelled" within
        ~10ms of submission — moments before the order proceeds completely
        normally through PreSubmitted -> Submitted -> Filled (confirmed
        live: the real fill can take ~0.5-0.6s+ to arrive, well past a
        fixed 0.5s sleep). Two real fills got permanently recorded in this
        project's own order history as "cancelled, 0 filled" as a result.

        Fix: "Filled" is trusted immediately (a genuine fill essentially
        never reverses). Any other apparently-terminal status
        (Cancelled/ApiCancelled/Inactive) is NOT trusted on sight — this
        method keeps polling for the full wait budget in case a real Fill
        (or a different, more final status) supersedes it. Only when the
        whole budget elapses without a Fill arriving is that status
        accepted as final. This intentionally makes a *genuinely*
        rejected/cancelled order take the full timeout to report — an
        acceptable cost (this system runs on a once/day cycle cadence, not
        latency-sensitive) for not silently corrupting the fill record.
        """
        deadline = time.monotonic() + _ORDER_STATUS_MAX_WAIT_SECONDS
        while time.monotonic() < deadline:
            ib.sleep(_ORDER_STATUS_POLL_INTERVAL)
            status = trade.orderStatus.status
            if status == "Filled":
                return
            # Anything else (PendingSubmit/PreSubmitted/Submitted, or a
            # Cancelled/ApiCancelled/Inactive that might still be the
            # benign transient blip) — keep watching until the deadline.
        # Timeout: no Fill ever superseded whatever status is showing now.
        # _map_trade's own status_map handles every remaining case,
        # including a genuine terminal Cancelled/rejected order.

    def cancel_order(self, order_id: str) -> bool:
        with self._locked():
            ib = self._ensure_connected()
            for trade in ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    ib.cancelOrder(trade.order)
                    return True
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        with self._locked():
            ib = self._ensure_connected()
            for trade in ib.trades():
                if str(trade.order.orderId) == order_id:
                    return self._map_trade(trade)
            raise BrokerError(f"Order {order_id} not found")

    def get_open_orders(self) -> list[OrderStatus]:
        with self._locked():
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
        # Unlike the previous-close fallback above, this is the exact
        # "fabricated zero feeding into order sizing" worst case this
        # method's docstring warns about — deserves a louder level than the
        # benign stale-but-real-price fallback three lines up.
        log.error("No midpoint, last, or close price available for %s; returning 0.0", symbol)
        return 0.0

    def get_current_price(self, symbol: str) -> float:
        with self._locked():
            ib = self._ensure_connected()
            contract = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(contract)
            [ticker] = ib.reqTickers(contract)
            return self._resolve_price(ticker, symbol)

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        with self._locked():
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

        Reads the contract details cached by connect() rather than fetching
        them live — see the comment there for why a live call here can hang
        forever when called from a different thread (e.g. a scheduled
        cycle running on APScheduler's own worker thread).
        """
        self._ensure_connected()
        cd = self._market_hours_details
        if cd is None:
            log.warning("No cached market-hours contract details; failing open")
            return True
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo(cd.timeZoneId))
        except Exception:
            log.warning(
                "Could not resolve exchange timezone %r; falling back to UTC "
                "for market-hours check", cd.timeZoneId, exc_info=True,
            )
            now = utcnow()
        now_str = now.strftime("%Y%m%d:%H%M")
        today_str = now.strftime("%Y%m%d")
        found_today = False
        for segment in cd.liquidHours.split(";"):
            if segment.startswith(today_str):
                found_today = True
            if "-" not in segment:
                continue  # e.g. "20260725:CLOSED" (holiday) — no session that day
            parts = segment.split("-")
            if len(parts) == 2 and parts[0] <= now_str <= parts[1]:
                return True
        if not found_today:
            # The cached schedule (fetched once at connect() time) doesn't
            # cover today at all — likely stale from a very long-running
            # connection outliving the window IBKR returned. Failing closed
            # here would silently stop the engine from ever trading again
            # with no error, the same class of bug this whole cache exists
            # to fix elsewhere. Fail open instead and let a human notice via
            # the resulting (harmless if actually closed) cycle activity.
            log.warning(
                "Cached market-hours schedule has no entry for %s; it may be "
                "stale (reconnect to refresh). Failing open.", today_str,
            )
            return True
        return False

    def market_hours(self) -> MarketHoursStatus:
        """Holiday-aware open/closed state + next session's open/close time.

        Reuses the same cached ``liquidHours`` schedule as ``is_market_open``
        (see that method's docstring for why it's cached rather than fetched
        live), but parses every segment into real datetimes instead of just
        bounding today's session — the schedule genuinely contains multiple
        future days' segments (including ``CLOSED`` holiday entries), so
        this is real next-transition data, not a guess.
        """
        self._ensure_connected()
        cd = self._market_hours_details
        if cd is None:
            log.warning("No cached market-hours contract details; failing open")
            return MarketHoursStatus(is_open=True)
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(cd.timeZoneId)
        except Exception:
            log.warning(
                "Could not resolve exchange timezone %r; falling back to UTC "
                "for market-hours check", cd.timeZoneId, exc_info=True,
            )
            tz = timezone.utc
        now = datetime.now(tz)

        is_open = False
        current_close: datetime | None = None
        upcoming: list[tuple[datetime, datetime]] = []
        for segment in cd.liquidHours.split(";"):
            if "-" not in segment:
                continue  # e.g. "20260725:CLOSED" (holiday) — no session that day
            parts = segment.split("-")
            if len(parts) != 2:
                continue
            try:
                seg_open = datetime.strptime(parts[0], "%Y%m%d:%H%M").replace(tzinfo=tz)
                seg_close = datetime.strptime(parts[1], "%Y%m%d:%H%M").replace(tzinfo=tz)
            except ValueError:
                continue
            if seg_open <= now <= seg_close:
                is_open = True
                current_close = seg_close
            elif seg_open > now:
                upcoming.append((seg_open, seg_close))

        if is_open:
            next_open = min((o for o, _ in upcoming), default=None)
            return MarketHoursStatus(is_open=True, next_open=next_open, next_close=current_close)
        if upcoming:
            next_open, next_close = min(upcoming, key=lambda pair: pair[0])
            return MarketHoursStatus(is_open=False, next_open=next_open, next_close=next_close)
        return MarketHoursStatus(is_open=False)

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
            timestamp=utcnow(),
        )
