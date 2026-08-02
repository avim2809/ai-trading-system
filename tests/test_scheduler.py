"""Dedicated unit tests for firm.live.scheduler.

Covers the pure helpers (trading_day_key, _pending_approvals_on_disk), the
catch-up-cycle branch logic (maybe_catch_up_session_cycle), and the
TradingScheduler wrapper around APScheduler's BackgroundScheduler. Uses
plain Mock/MagicMock engines rather than a real LiveTradingEngine — the
scheduler only touches a handful of duck-typed attributes/methods, so a
full engine fixture would just add noise and slow the suite down.

test_live_engine.py::TestMarketSessionSync covers one end-to-end catch-up
path against a real LiveTradingEngine; this file focuses on exhaustively
covering the branches within firm.live.scheduler itself.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone as dt_tz
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from firm.live.scheduler import (
    DEFAULT_MARKET_TIMEZONE,
    TradingScheduler,
    _pending_approvals_on_disk,
    maybe_catch_up_session_cycle,
    trading_day_key,
)


# ---------------------------------------------------------------------------
# trading_day_key
# ---------------------------------------------------------------------------

class TestTradingDayKey:
    def test_naive_datetime_treated_as_utc(self):
        naive = datetime(2026, 7, 26, 3, 0)
        assert trading_day_key(naive, "US/Eastern") == "2026-07-25"

    def test_aware_datetime_converted_to_target_timezone(self):
        aware = datetime(2026, 7, 26, 3, 0, tzinfo=dt_tz.utc)
        assert trading_day_key(aware, "US/Eastern") == "2026-07-25"

    def test_default_timezone_is_us_eastern(self):
        aware = datetime(2026, 7, 26, 3, 0, tzinfo=dt_tz.utc)
        assert trading_day_key(aware) == trading_day_key(aware, DEFAULT_MARKET_TIMEZONE)

    def test_different_timezone_produces_different_key(self):
        aware = datetime(2026, 7, 26, 3, 0, tzinfo=dt_tz.utc)
        assert trading_day_key(aware, "UTC") == "2026-07-26"


# ---------------------------------------------------------------------------
# _pending_approvals_on_disk
# ---------------------------------------------------------------------------

class TestPendingApprovalsOnDisk:
    def test_missing_file_returns_false(self, tmp_path):
        assert _pending_approvals_on_disk(tmp_path / "does_not_exist.json") is False

    def test_no_pending_entries_returns_false(self, tmp_path):
        path = tmp_path / "approvals.json"
        path.write_text(json.dumps([{"status": "approved"}, {"status": "rejected"}]))
        assert _pending_approvals_on_disk(path) is False

    def test_pending_entry_returns_true(self, tmp_path):
        path = tmp_path / "approvals.json"
        path.write_text(json.dumps([{"status": "approved"}, {"status": "pending"}]))
        assert _pending_approvals_on_disk(path) is True

    def test_malformed_json_degrades_to_false(self, tmp_path):
        path = tmp_path / "approvals.json"
        path.write_text("{not valid json")
        assert _pending_approvals_on_disk(path) is False

    def test_non_list_rows_are_ignored(self, tmp_path):
        path = tmp_path / "approvals.json"
        path.write_text(json.dumps(["not-a-dict", 42, {"status": "pending"}]))
        assert _pending_approvals_on_disk(path) is True


# ---------------------------------------------------------------------------
# maybe_catch_up_session_cycle
# ---------------------------------------------------------------------------

def _mock_engine(*, is_running=True, shutting_down=False, had_cycle_today=False, market_open=True):
    engine = MagicMock()
    engine.is_running = is_running
    engine._shutting_down = shutting_down
    engine.had_cycle_today.return_value = had_cycle_today
    engine._broker.is_market_open.return_value = market_open
    return engine


def _some_weekday() -> datetime:
    """A tz-aware US/Eastern datetime guaranteed to be Mon-Fri, computed
    at runtime so tests don't depend on which day CI happens to run on."""
    candidate = datetime.now(ZoneInfo(DEFAULT_MARKET_TIMEZONE))
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _some_saturday() -> datetime:
    candidate = datetime.now(ZoneInfo(DEFAULT_MARKET_TIMEZONE))
    while candidate.weekday() != 5:
        candidate += timedelta(days=1)
    return candidate


def _patched_weekday_now():
    """Context manager replacing the ``datetime`` name in firm.live.scheduler
    with a stand-in whose .now() returns a fixed Mon-Fri instant, so the
    weekend short-circuit in maybe_catch_up_session_cycle never triggers
    regardless of the real calendar date the suite happens to run on.

    datetime.datetime is an immutable C type — its .now classmethod can't
    be patched in place, so the whole module-level name is swapped instead.
    """
    mock_dt = MagicMock(wraps=datetime)
    mock_dt.now.return_value = _some_weekday()
    return patch("firm.live.scheduler.datetime", mock_dt)


class TestMaybeCatchUpSessionCycle:
    def test_noop_for_non_session_schedule(self):
        engine = _mock_engine()
        maybe_catch_up_session_cycle(engine, "hourly")
        engine.had_cycle_today.assert_not_called()

    def test_noop_for_unrecognized_schedule(self):
        engine = _mock_engine()
        maybe_catch_up_session_cycle(engine, "every_15_minutes")
        engine.had_cycle_today.assert_not_called()

    def test_skipped_when_warmup_not_ready_in_time(self):
        engine = _mock_engine()
        gate = MagicMock()
        gate.is_ready = False
        gate.wait_ready.return_value = False
        maybe_catch_up_session_cycle(engine, "market_open", warmup_gate=gate)
        gate.wait_ready.assert_called_once()
        engine.had_cycle_today.assert_not_called()

    def test_proceeds_when_warmup_gate_reports_ready(self):
        engine = _mock_engine(had_cycle_today=True)
        gate = MagicMock()
        gate.is_ready = True
        maybe_catch_up_session_cycle(engine, "market_open", warmup_gate=gate)
        gate.wait_ready.assert_not_called()
        engine.had_cycle_today.assert_called_once()

    def test_proceeds_when_warmup_wait_ready_succeeds(self):
        engine = _mock_engine(had_cycle_today=True)
        gate = MagicMock()
        gate.is_ready = False
        gate.wait_ready.return_value = True
        maybe_catch_up_session_cycle(engine, "market_open", warmup_gate=gate)
        engine.had_cycle_today.assert_called_once()

    def test_skipped_when_pending_approvals_on_disk(self, tmp_path):
        approvals = tmp_path / "approvals.json"
        approvals.write_text(json.dumps([{"status": "pending"}]))
        engine = _mock_engine()
        maybe_catch_up_session_cycle(engine, "market_open", approvals_path=approvals)
        engine.had_cycle_today.assert_not_called()

    def test_skipped_when_engine_not_running(self, tmp_path):
        engine = _mock_engine(is_running=False)
        maybe_catch_up_session_cycle(
            engine, "market_open", approvals_path=tmp_path / "missing.json"
        )
        engine.had_cycle_today.assert_not_called()

    def test_skipped_when_engine_shutting_down(self, tmp_path):
        engine = _mock_engine(shutting_down=True)
        maybe_catch_up_session_cycle(
            engine, "market_open", approvals_path=tmp_path / "missing.json"
        )
        engine.had_cycle_today.assert_not_called()

    def test_skipped_when_cycle_already_ran_today(self, tmp_path):
        engine = _mock_engine(had_cycle_today=True)
        maybe_catch_up_session_cycle(
            engine, "market_open", approvals_path=tmp_path / "missing.json"
        )
        engine._broker.is_market_open.assert_not_called()

    def test_skipped_on_weekend(self, tmp_path):
        engine = _mock_engine(had_cycle_today=False)
        mock_dt = MagicMock(wraps=datetime)
        mock_dt.now.return_value = _some_saturday()
        with patch("firm.live.scheduler.datetime", mock_dt):
            maybe_catch_up_session_cycle(
                engine, "market_open", approvals_path=tmp_path / "missing.json"
            )
        engine._broker.is_market_open.assert_not_called()

    def test_skipped_when_market_hours_check_raises(self, tmp_path):
        engine = _mock_engine(had_cycle_today=False)
        engine._broker.is_market_open.side_effect = RuntimeError("boom")
        with _patched_weekday_now():
            maybe_catch_up_session_cycle(
                engine, "market_open", approvals_path=tmp_path / "missing.json"
            )
        # Exception is swallowed and logged, not propagated.

    def test_skipped_when_market_closed(self, tmp_path):
        engine = _mock_engine(had_cycle_today=False, market_open=False)
        with _patched_weekday_now():
            maybe_catch_up_session_cycle(
                engine, "market_open", approvals_path=tmp_path / "missing.json"
            )
        engine.run_cycle.assert_not_called()

    def test_starts_catch_up_cycle_when_market_open(self, tmp_path):
        ran = threading.Event()
        engine = _mock_engine(had_cycle_today=False, market_open=True)
        engine.run_cycle.side_effect = lambda: ran.set()

        with _patched_weekday_now():
            maybe_catch_up_session_cycle(
                engine, "market_close", approvals_path=tmp_path / "missing.json"
            )
        assert ran.wait(timeout=5.0), "catch-up thread never invoked run_cycle()"
        engine.run_cycle.assert_called_once()

    def test_catch_up_thread_logs_and_swallows_run_cycle_errors(self, tmp_path):
        done = threading.Event()
        engine = _mock_engine(had_cycle_today=False, market_open=True)

        def _boom():
            done.set()
            raise RuntimeError("pipeline exploded")

        engine.run_cycle.side_effect = _boom
        with _patched_weekday_now():
            maybe_catch_up_session_cycle(
                engine, "market_open", approvals_path=tmp_path / "missing.json"
            )
        assert done.wait(timeout=5.0)

    def test_catch_up_thread_noop_when_shutdown_flag_flips_before_run(self, tmp_path):
        """The inner _run() re-checks _shutting_down right before calling
        run_cycle() to close a race where shutdown begins after the
        top-level guard passed but before the daemon thread executes."""
        engine = _mock_engine(had_cycle_today=False, market_open=True)

        # Bypass the outer guard (which would also skip on _shutting_down)
        # by flipping it back to False only for the outer check, then True
        # for the inner thread — simulate via a mutable flag object instead.
        class _FlippingBool:
            def __init__(self):
                self._n = 0

            def __bool__(self):
                self._n += 1
                return self._n > 1

        engine._shutting_down = _FlippingBool()
        with _patched_weekday_now():
            maybe_catch_up_session_cycle(
                engine, "market_open", approvals_path=tmp_path / "missing.json"
            )
        time.sleep(0.2)
        engine.run_cycle.assert_not_called()


# ---------------------------------------------------------------------------
# TradingScheduler
# ---------------------------------------------------------------------------

class TestTradingSchedulerConstruction:
    def test_raises_without_apscheduler_installed(self):
        with patch("firm.live.scheduler._HAS_APSCHEDULER", False):
            with pytest.raises(ImportError, match="apscheduler"):
                TradingScheduler(engine=MagicMock())

    def test_defaults(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine)
        assert sched._schedule_spec == "market_open"
        assert sched._timezone == DEFAULT_MARKET_TIMEZONE
        assert sched._universe == []
        assert sched.is_running() is False
        assert sched.next_run() is None


class TestTradingSchedulerLifecycle:
    def test_start_and_stop_without_universe(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="hourly")
        try:
            sched.start()
            assert sched.is_running() is True
            job = sched._scheduler.get_job(sched._job_id)
            assert job is not None
            assert sched._scheduler.get_job(sched._fundamentals_job_id) is None
        finally:
            sched.stop()
        assert sched.is_running() is False

    def test_start_adds_fundamentals_job_when_universe_given(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open", universe=["AAPL", "MSFT"]
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._fundamentals_job_id)
            assert job is not None
        finally:
            sched.stop()

    def test_dynamic_universe_job_not_added_by_default(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open", universe=["AAPL", "MSFT"]
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._dynamic_universe_job_id) is None
        finally:
            sched.stop()

    def test_dynamic_universe_job_added_when_enabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open", universe=["AAPL", "MSFT"],
            dynamic_universe_enabled=True,
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._dynamic_universe_job_id)
            assert job is not None
        finally:
            sched.stop()

    def test_dynamic_universe_job_not_added_without_universe_even_if_enabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open", dynamic_universe_enabled=True,
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._dynamic_universe_job_id) is None
        finally:
            sched.stop()

    def test_dynamic_universe_sync_hour_defaults_to_before_fundamentals(self):
        sched = TradingScheduler(engine=MagicMock(), fundamentals_refresh_hour=8)
        assert sched._dynamic_universe_sync_hour == 7

    def test_dynamic_universe_sync_hour_explicit_override(self):
        sched = TradingScheduler(
            engine=MagicMock(), fundamentals_refresh_hour=8, dynamic_universe_sync_hour=3,
        )
        assert sched._dynamic_universe_sync_hour == 3

    def test_stop_is_idempotent_when_never_started(self):
        sched = TradingScheduler(engine=MagicMock())
        sched.stop()  # must not raise
        assert sched.is_running() is False

    def test_stop_is_idempotent_after_stopping_twice(self):
        sched = TradingScheduler(engine=MagicMock())
        sched.start()
        sched.stop()
        sched.stop()  # must not raise
        assert sched.is_running() is False

    def test_run_now_invokes_engine_run_cycle_directly(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine)
        sched.run_now()
        engine.run_cycle.assert_called_once()

    def test_run_now_swallows_engine_errors(self):
        engine = MagicMock()
        engine.run_cycle.side_effect = RuntimeError("boom")
        sched = TradingScheduler(engine=engine)
        sched.run_now()  # must not raise

    def test_next_run_none_when_job_missing(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine)
        try:
            sched.start()
            sched._scheduler.remove_job(sched._job_id)
            assert sched.next_run() is None
        finally:
            sched.stop()

    def test_next_run_returns_datetime_after_start(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="hourly")
        try:
            sched.start()
            nxt = sched.next_run()
            assert nxt is not None
            assert isinstance(nxt, datetime)
        finally:
            sched.stop()


class TestBuildTrigger:
    @pytest.mark.parametrize("spec", ["market_open", "market_close", "hourly"])
    def test_preset_schedules(self, spec):
        sched = TradingScheduler(engine=MagicMock(), schedule=spec)
        trigger = sched._build_trigger(spec)
        assert trigger is not None

    def test_cron_shorthand_hh_mm(self):
        sched = TradingScheduler(engine=MagicMock())
        trigger = sched._build_trigger("cron:14:30")
        assert trigger is not None
        assert str(trigger.fields[trigger.FIELD_NAMES.index("hour")]) == "14"

    def test_cron_shorthand_hour_only_defaults_minute_zero(self):
        sched = TradingScheduler(engine=MagicMock())
        trigger = sched._build_trigger("cron:9")
        assert str(trigger.fields[trigger.FIELD_NAMES.index("minute")]) == "0"

    def test_every_n_minutes_interval(self):
        sched = TradingScheduler(engine=MagicMock())
        trigger = sched._build_trigger("every_5_minutes")
        assert trigger.interval == timedelta(minutes=5)

    def test_raw_crontab_expression(self):
        sched = TradingScheduler(engine=MagicMock())
        trigger = sched._build_trigger("*/10 * * * mon-fri")
        assert trigger is not None

    def test_invalid_spec_raises(self):
        sched = TradingScheduler(engine=MagicMock())
        with pytest.raises(Exception):
            sched._build_trigger("not-a-valid-spec!!")


class TestRunCycleSafe:
    def test_delegates_to_engine_run_cycle(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine)
        sched._run_cycle_safe()
        engine.run_cycle.assert_called_once()

    def test_logs_and_swallows_exception(self, caplog):
        engine = MagicMock()
        engine.run_cycle.side_effect = ValueError("nope")
        sched = TradingScheduler(engine=engine)
        with caplog.at_level("ERROR"):
            sched._run_cycle_safe()
        assert any("Scheduled cycle failed" in r.message for r in caplog.records)
