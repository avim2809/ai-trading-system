"""Tests for IBKRBroker's price resolution.

Regression coverage for a real incident found verifying a live IBKR paper
run: outside trading hours (or without a live data subscription), IBKR's
delayed ticker has bid/ask as the empty sentinel -1 (midpoint() -> NaN) and
``last`` defaults to 0.0 — a real float, not NaN, so a NaN-only check
silently accepted it as a fake zero price and every order got skipped with
"No price for X".
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("ib_async")

from firm.brokers.ibkr import IBKRBroker


def _ticker(midpoint=float("nan"), last=float("nan"), close=float("nan")):
    t = SimpleNamespace(last=last, close=close)
    t.midpoint = lambda: midpoint
    return t


class TestResolvePrice:
    def test_uses_valid_midpoint(self):
        assert IBKRBroker._resolve_price(_ticker(midpoint=100.5), "AAPL") == 100.5

    def test_falls_back_to_last_when_midpoint_nan(self):
        assert IBKRBroker._resolve_price(_ticker(last=101.0), "AAPL") == 101.0

    def test_ignores_zero_last_sentinel_falls_back_to_close(self):
        # The exact bug: last=0.0 is a real float (not NaN) that must not be
        # mistaken for a genuine last-trade price.
        ticker = _ticker(last=0.0, close=333.74)
        assert IBKRBroker._resolve_price(ticker, "AAPL") == 333.74

    def test_all_empty_returns_zero_with_warning(self, caplog):
        ticker = _ticker(last=0.0, close=float("nan"))
        with caplog.at_level("WARNING"):
            price = IBKRBroker._resolve_price(ticker, "AAPL")
        assert price == 0.0
        assert any("No midpoint, last, or close" in r.message for r in caplog.records)

    def test_negative_or_zero_midpoint_is_rejected(self):
        ticker = _ticker(midpoint=0.0, last=0.0, close=50.0)
        assert IBKRBroker._resolve_price(ticker, "AAPL") == 50.0

    def test_get_current_prices_uses_resolve_price_per_symbol(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=1)
        fake_ib = SimpleNamespace(
            isConnected=lambda: True,
            qualifyContracts=lambda *cs: None,
            reqTickers=lambda *cs: [_ticker(last=0.0, close=200.0), _ticker(midpoint=50.0)],
        )
        broker._ib = fake_ib
        with patch("firm.brokers.ibkr.Stock", side_effect=lambda sym, *_a: SimpleNamespace(symbol=sym)):
            prices = broker.get_current_prices(["AAPL", "MSFT"])
        assert prices == {"AAPL": 200.0, "MSFT": 50.0}


class TestIsMarketOpen:
    """Regression coverage for a real incident: is_market_open() compared
    Eastern-time trading-hours strings against a raw UTC clock reading with
    no timezone conversion, and used the extended tradingHours window
    instead of the regular liquidHours session — together making it report
    "open" at 2:55am ET simply because the UTC clock digits (06:55) happened
    to fall inside the Eastern-time range string (0400-2000).

    Also covers a second, later incident: is_market_open() used to fetch
    contract details live on every call via ib.qualifyContracts()/
    reqContractDetails() — both blocking async round-trips tied to
    whichever asyncio event loop is current on the *calling* thread.
    Reproduced directly against a real IBKR connection: calling this from
    a thread other than the one that called connect() (exactly what
    APScheduler's BackgroundScheduler does for every scheduled cycle) hangs
    forever with no timeout and no error. A real cycle hung for 24+ hours
    this way. Fixed by fetching the contract details once in connect() (on
    the connecting thread) and having is_market_open() read that cache —
    these tests set the cache directly rather than mocking qualifyContracts/
    reqContractDetails, since is_market_open() must not call either anymore.
    """

    def _contract_details(self, liquid_hours: str, trading_hours: str = "20260720:0400-20260720:2000"):
        return SimpleNamespace(liquidHours=liquid_hours, tradingHours=trading_hours, timeZoneId="US/Eastern")

    def _broker_with(self, details):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=2)
        broker._ib = SimpleNamespace(isConnected=lambda: True)
        broker._market_hours_details = details
        return broker

    def test_false_before_market_open_despite_utc_string_match(self, monkeypatch):
        # 06:55 UTC on 2026-07-20 is 02:55am ET — closed. The old bug
        # compared "20260720:0655" (UTC) directly against the ET session
        # string "20260720:0400-20260720:2000" and got a false positive.
        import datetime as dt
        from zoneinfo import ZoneInfo

        class _FixedDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2026, 7, 20, 2, 55, tzinfo=ZoneInfo("US/Eastern"))

        monkeypatch.setattr("firm.brokers.ibkr.datetime", _FixedDatetime)
        broker = self._broker_with(self._contract_details("20260720:0930-20260720:1600"))
        assert broker.is_market_open() is False

    def test_true_during_regular_session(self, monkeypatch):
        import datetime as dt
        from zoneinfo import ZoneInfo

        class _FixedDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2026, 7, 20, 11, 0, tzinfo=ZoneInfo("US/Eastern"))

        monkeypatch.setattr("firm.brokers.ibkr.datetime", _FixedDatetime)
        broker = self._broker_with(self._contract_details("20260720:0930-20260720:1600"))
        assert broker.is_market_open() is True

    def test_false_during_extended_hours_only(self, monkeypatch):
        # 6am ET is inside tradingHours (extended) but not liquidHours
        # (regular session) — must use liquidHours, not tradingHours.
        import datetime as dt
        from zoneinfo import ZoneInfo

        class _FixedDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2026, 7, 20, 6, 0, tzinfo=ZoneInfo("US/Eastern"))

        monkeypatch.setattr("firm.brokers.ibkr.datetime", _FixedDatetime)
        broker = self._broker_with(self._contract_details("20260720:0930-20260720:1600"))
        assert broker.is_market_open() is False

    def test_holiday_closed_segment_is_skipped_not_crashed(self, monkeypatch):
        import datetime as dt
        from zoneinfo import ZoneInfo

        class _FixedDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2026, 7, 25, 11, 0, tzinfo=ZoneInfo("US/Eastern"))

        monkeypatch.setattr("firm.brokers.ibkr.datetime", _FixedDatetime)
        broker = self._broker_with(self._contract_details("20260725:CLOSED"))
        assert broker.is_market_open() is False

    def test_no_cached_details_fails_open(self):
        """Missing cache (e.g. connect() got no details back) must not
        silently and permanently report "closed" — that would stop the
        engine from ever trading again with no error, the same class of
        silent failure this whole cache exists to prevent elsewhere."""
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=3)
        broker._ib = SimpleNamespace(isConnected=lambda: True)
        broker._market_hours_details = None
        assert broker.is_market_open() is True

    def test_stale_schedule_with_no_entry_for_today_fails_open(self, monkeypatch):
        """The cache is fetched once at connect() time and never refreshed
        mid-connection; if a long-lived connection outlives the schedule
        window IBKR returned, there's no entry for today at all (distinct
        from an explicit CLOSED holiday entry) — must fail open, not
        silently report closed forever."""
        import datetime as dt
        from zoneinfo import ZoneInfo

        class _FixedDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2027, 1, 15, 11, 0, tzinfo=ZoneInfo("US/Eastern"))

        monkeypatch.setattr("firm.brokers.ibkr.datetime", _FixedDatetime)
        broker = self._broker_with(self._contract_details("20260720:0930-20260720:1600"))
        assert broker.is_market_open() is True

    def test_does_not_call_qualify_or_req_contract_details(self, monkeypatch):
        """The actual fix: no live network round-trip on this call path at
        all — that's what hung for 24+ hours when called from a different
        thread than connect()."""
        import datetime as dt
        from zoneinfo import ZoneInfo

        class _FixedDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2026, 7, 20, 11, 0, tzinfo=ZoneInfo("US/Eastern"))

        monkeypatch.setattr("firm.brokers.ibkr.datetime", _FixedDatetime)
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=9)
        calls = {"n": 0}

        def _boom(*_a, **_k):
            calls["n"] += 1
            raise AssertionError("must not be called")

        broker._ib = SimpleNamespace(isConnected=lambda: True, qualifyContracts=_boom, reqContractDetails=_boom)
        broker._market_hours_details = self._contract_details("20260720:0930-20260720:1600")

        assert broker.is_market_open() is True
        assert calls["n"] == 0

    def test_callable_from_a_different_thread_than_connect(self, monkeypatch):
        """The actual production scenario: APScheduler runs every scheduled
        cycle on its own worker thread, never the thread that called
        connect(). Before the fix, this hung forever (reproduced live
        against a real IBKR connection); now it's a plain attribute read
        with no thread affinity at all."""
        import datetime as dt
        import threading
        from zoneinfo import ZoneInfo

        class _FixedDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2026, 7, 20, 11, 0, tzinfo=ZoneInfo("US/Eastern"))

        monkeypatch.setattr("firm.brokers.ibkr.datetime", _FixedDatetime)
        broker = self._broker_with(self._contract_details("20260720:0930-20260720:1600"))
        result_holder = {}

        def _call_from_worker():
            result_holder["result"] = broker.is_market_open()

        t = threading.Thread(target=_call_from_worker)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()
        assert result_holder["result"] is True


class TestMarketHours:
    """market_hours() reuses the same cached liquidHours schedule as
    is_market_open() (see TestIsMarketOpen above) but parses every segment
    into real datetimes to compute the next open/close transition, not just
    bound today's session."""

    def _contract_details(self, liquid_hours: str):
        return SimpleNamespace(liquidHours=liquid_hours, timeZoneId="US/Eastern")

    def _broker_with(self, details):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=4)
        broker._ib = SimpleNamespace(isConnected=lambda: True)
        broker._market_hours_details = details
        return broker

    def _fixed_now(self, monkeypatch, y, mo, d, h, mi):
        import datetime as dt
        from zoneinfo import ZoneInfo

        class _FixedDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(y, mo, d, h, mi, tzinfo=ZoneInfo("US/Eastern"))

        monkeypatch.setattr("firm.brokers.ibkr.datetime", _FixedDatetime)

    def test_open_now_reports_todays_close(self, monkeypatch):
        self._fixed_now(monkeypatch, 2026, 7, 20, 11, 0)
        broker = self._broker_with(self._contract_details(
            "20260720:0930-20260720:1600;20260721:0930-20260721:1600"
        ))
        mh = broker.market_hours()
        assert mh.is_open is True
        assert mh.next_close.isoformat() == "2026-07-20T16:00:00-04:00"
        # Also surfaces the following day's open, since it's known.
        assert mh.next_open.isoformat() == "2026-07-21T09:30:00-04:00"

    def test_closed_before_open_reports_todays_session(self, monkeypatch):
        self._fixed_now(monkeypatch, 2026, 7, 20, 6, 0)
        broker = self._broker_with(self._contract_details("20260720:0930-20260720:1600"))
        mh = broker.market_hours()
        assert mh.is_open is False
        assert mh.next_open.isoformat() == "2026-07-20T09:30:00-04:00"
        assert mh.next_close.isoformat() == "2026-07-20T16:00:00-04:00"

    def test_closed_after_close_reports_next_days_session(self, monkeypatch):
        self._fixed_now(monkeypatch, 2026, 7, 20, 18, 0)
        broker = self._broker_with(self._contract_details(
            "20260720:0930-20260720:1600;20260721:0930-20260721:1600"
        ))
        mh = broker.market_hours()
        assert mh.is_open is False
        assert mh.next_open.isoformat() == "2026-07-21T09:30:00-04:00"
        assert mh.next_close.isoformat() == "2026-07-21T16:00:00-04:00"

    def test_holiday_segment_skipped_next_transition_is_day_after(self, monkeypatch):
        self._fixed_now(monkeypatch, 2026, 7, 24, 11, 0)
        broker = self._broker_with(self._contract_details(
            "20260724:0930-20260724:1600;20260725:CLOSED;20260726:0930-20260726:1600"
        ))
        mh = broker.market_hours()
        assert mh.is_open is True
        assert mh.next_close.isoformat() == "2026-07-24T16:00:00-04:00"
        # 7/25 is a holiday (CLOSED, no "-") — must be skipped, not crash,
        # and next_open must correctly skip past it to 7/26.
        assert mh.next_open.isoformat() == "2026-07-26T09:30:00-04:00"

    def test_no_cached_details_fails_open_with_unknown_transitions(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=5)
        broker._ib = SimpleNamespace(isConnected=lambda: True)
        broker._market_hours_details = None
        mh = broker.market_hours()
        assert mh.is_open is True
        assert mh.next_open is None
        assert mh.next_close is None

    def test_no_upcoming_segments_at_all_reports_closed_unknown(self, monkeypatch):
        """Every segment already fully in the past (stale schedule) — must
        not crash, degrades to closed/unknown rather than fabricating a date."""
        self._fixed_now(monkeypatch, 2026, 7, 22, 11, 0)
        broker = self._broker_with(self._contract_details("20260720:0930-20260720:1600"))
        mh = broker.market_hours()
        assert mh.is_open is False
        assert mh.next_open is None
        assert mh.next_close is None


class TestConnectCachesMarketHoursDetails:
    """connect() must fetch the market-hours contract details once, on the
    connecting thread, so is_market_open() never needs to — see
    TestIsMarketOpen for why a live call from a different thread hangs.
    """

    def test_connect_populates_the_cache(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=10)
        details = SimpleNamespace(liquidHours="20260720:0930-20260720:1600", timeZoneId="US/Eastern")
        fake_ib = SimpleNamespace(
            connect=lambda *a, **k: None,
            reqMarketDataType=lambda *a: None,
            reqAccountSummary=lambda: None,
            qualifyContracts=lambda *cs: None,
            reqContractDetails=lambda *cs: [details],
        )
        with patch("firm.brokers.ibkr.IB", return_value=fake_ib), patch("firm.brokers.ibkr.Stock"):
            broker.connect()
        assert broker._market_hours_details is details

    def test_connect_handles_empty_contract_details(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=11)
        fake_ib = SimpleNamespace(
            connect=lambda *a, **k: None,
            reqMarketDataType=lambda *a: None,
            reqAccountSummary=lambda: None,
            qualifyContracts=lambda *cs: None,
            reqContractDetails=lambda *cs: [],
        )
        with patch("firm.brokers.ibkr.IB", return_value=fake_ib), patch("firm.brokers.ibkr.Stock"):
            broker.connect()
        assert broker._market_hours_details is None


class TestGetAccountThreadSafety:
    """Regression coverage for a real incident: get_account() called
    ib.reqAccountSummary() (which awaits an asyncio Future) on every call.
    FastAPI's sync route handlers run in an anyio threadpool that may
    service any given request on a different thread than the one that
    called connect() — a thread with no asyncio event loop at all, which
    crashed reqAccountSummary() with "no current event loop in thread".
    Reproduced live: /api/live/account failed 100% of the time once the
    engine had been running a while, while /api/live/positions (which only
    reads the cached ib.positions() list, no Future) worked fine.

    Fix: subscribe once in connect() (guaranteed to run on a consistent
    thread), then get_account() only reads the cached accountSummary() —
    exactly the same pattern get_positions() already used successfully.
    """

    def test_get_account_does_not_call_reqAccountSummary(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=4)
        calls = {"req": 0}

        def _req_account_summary():
            calls["req"] += 1

        broker._ib = SimpleNamespace(
            isConnected=lambda: True,
            reqAccountSummary=_req_account_summary,
            accountSummary=lambda: [
                SimpleNamespace(tag="NetLiquidation", value="1001461.74"),
                SimpleNamespace(tag="TotalCashValue", value="1000086.08"),
            ],
        )
        result = broker.get_account()
        assert calls["req"] == 0  # only connect() should ever call this
        assert result["equity"] == 1001461.74
        assert result["cash"] == 1000086.08

    def test_connect_subscribes_to_account_summary(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=5)
        calls = {"req": 0}
        fake_ib = SimpleNamespace(
            connect=lambda *a, **k: None,
            reqMarketDataType=lambda *a: None,
            reqAccountSummary=lambda: calls.__setitem__("req", calls["req"] + 1),
            qualifyContracts=lambda *cs: None,
            reqContractDetails=lambda *cs: [],
        )
        with patch("firm.brokers.ibkr.IB", return_value=fake_ib), patch("firm.brokers.ibkr.Stock"):
            broker.connect()
        assert calls["req"] == 1

    def test_get_account_callable_from_a_different_thread(self):
        # The actual regression: simulate connect() happening on "this"
        # thread and get_account() being called from a genuinely different
        # one, the way FastAPI's threadpool would for a later request.
        import threading

        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=6)
        broker._ib = SimpleNamespace(
            isConnected=lambda: True,
            accountSummary=lambda: [SimpleNamespace(tag="NetLiquidation", value="500.0")],
        )
        result_holder = {}

        def _call_from_worker():
            result_holder["result"] = broker.get_account()

        t = threading.Thread(target=_call_from_worker)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()
        assert result_holder["result"]["equity"] == 500.0


class TestGetPositionsUsesRealMarketValue:
    """Regression coverage for a real incident: get_positions() read
    ib.positions() (cost basis only — no live price) and computed
    market_value as quantity * avgCost — the ORIGINAL cost, not current
    market value — with unrealized_pnl hardcoded to 0.0 always. A position
    would show the exact same "market value" and a permanent $0.00 P&L
    regardless of any real price movement, which looks exactly like a
    stale/non-updating position in the GUI. Verified live against a real
    IBKR paper position: the old code would have reported market_value
    equal to the $332.50 cost basis and unrealized_pnl 0.0 forever, while
    the real account showed the price had moved to $327.11 (-$5.39
    unrealized). Fixed by switching to ib.portfolio(), which carries
    IBKR's own live-priced marketValue/unrealizedPNL (same kind of
    locally-cached, push-updated list as ib.positions(), so no new
    blocking call or subscription is introduced).
    """

    def test_reads_live_market_value_and_pnl_not_cost_basis(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=7)
        item = SimpleNamespace(
            contract=SimpleNamespace(symbol="AAPL"),
            position=1.0,
            averageCost=332.5,
            marketValue=327.11,
            unrealizedPNL=-5.39,
        )
        broker._ib = SimpleNamespace(isConnected=lambda: True, portfolio=lambda: [item])

        [pos] = broker.get_positions()

        assert pos.symbol == "AAPL"
        assert pos.avg_cost == 332.5
        # The bug: these two used to always be avgCost*qty and 0.0.
        assert pos.market_value == 327.11
        assert pos.unrealized_pnl == -5.39

    def test_does_not_call_positions(self):
        """ib.positions() lacks live pricing entirely — must not be used."""
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=8)
        calls = {"positions": 0}
        broker._ib = SimpleNamespace(
            isConnected=lambda: True,
            portfolio=lambda: [],
            positions=lambda: calls.__setitem__("positions", calls["positions"] + 1),
        )

        broker.get_positions()

        assert calls["positions"] == 0


class _FakeOrderStatus:
    def __init__(self, status: str):
        self.status = status


class _FakeTrade:
    def __init__(self, contract, order, statuses: list[str]):
        self.contract = contract
        self.order = order
        self.orderStatus = _FakeOrderStatus(statuses[0])
        self._statuses = statuses
        self._i = 0
        self.fills: list = []

    def advance(self) -> None:
        if self._i + 1 < len(self._statuses):
            self._i += 1
            self.orderStatus.status = self._statuses[self._i]


def _fake_ib(statuses: list[str]):
    """A fake ``ib`` whose placeOrder returns a _FakeTrade that advances
    through *statuses* one step per ib.sleep() call — simulating status
    updates arriving asynchronously over time, at a fake-but-monotonic
    clock so tests don't take real wall-clock seconds. Uses the real
    MarketOrder/Stock objects submit_order constructs (only orderId is
    injected, mirroring what a real placeOrder call assigns)."""
    state = {"trade": None, "clock": 0.0}

    def place_order(contract, ib_order):
        ib_order.orderId = 1
        trade = _FakeTrade(contract, ib_order, statuses)
        state["trade"] = trade
        return trade

    def fake_sleep(interval):
        state["clock"] += interval
        if state["trade"] is not None:
            state["trade"].advance()

    ib = SimpleNamespace(
        isConnected=lambda: True,
        qualifyContracts=lambda *contracts: None,
        placeOrder=place_order,
        sleep=fake_sleep,
    )
    return ib, state


class TestSubmitOrderWaitsForRealResolution:
    """Regression coverage for a real incident: IBKR relays a benign
    informational message (errorCode 10349, "Order TIF was set to DAY
    based on order preset") through the same channel real cancellations
    use, transiently flipping status to "Cancelled" moments before the
    order proceeds normally to Filled. The original fixed-0.5s-sleep-
    then-snapshot approach captured and permanently recorded that blip as
    the order's final status — confirmed live: two real fills got
    recorded as "cancelled, 0 filled" in this project's own order history,
    only caught by comparing the live dashboard against the real IBKR
    account."""

    def test_benign_cancel_blip_then_real_fill_is_recorded_as_filled(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=9)
        ib, state = _fake_ib(["PendingSubmit", "Cancelled", "Cancelled", "PreSubmitted", "Submitted", "Filled"])
        broker._ib = ib

        from firm.brokers.base import OrderRequest
        status = broker.submit_order(OrderRequest(symbol="V", side="sell", quantity=44))

        assert status.status == "filled"

    def test_genuine_fill_returns_without_waiting_out_the_full_budget(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=10)
        ib, state = _fake_ib(["PendingSubmit", "Submitted", "Filled"])
        broker._ib = ib

        from firm.brokers.base import OrderRequest
        status = broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=5))

        assert status.status == "filled"
        # Should stop polling as soon as Filled is seen, not exhaust the
        # whole 5s wait budget.
        assert state["clock"] < 5.0

    def test_genuine_cancellation_with_no_fill_waits_full_budget_then_reports_cancelled(self):
        """Patches time.monotonic to follow the fake sleep-driven clock, so
        this test verifies the full-timeout-then-accept behavior without
        actually costing 5 real wall-clock seconds."""
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=11)
        ib, state = _fake_ib(["PendingSubmit", "Cancelled"])  # never advances further
        broker._ib = ib

        from firm.brokers.base import OrderRequest
        with patch("firm.brokers.ibkr.time.monotonic", side_effect=lambda: state["clock"]):
            status = broker.submit_order(OrderRequest(symbol="XOM", side="sell", quantity=1))

        assert status.status == "cancelled"
        # Waited out the full timeout rather than trusting the first
        # Cancelled-looking status immediately.
        from firm.brokers.ibkr import _ORDER_STATUS_MAX_WAIT_SECONDS
        assert state["clock"] >= _ORDER_STATUS_MAX_WAIT_SECONDS
