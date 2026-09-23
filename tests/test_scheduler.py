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
    EXTENDED_HOURS_CYCLE_TYPES,
    HOURLY_MARKET_HOURS,
    TradingScheduler,
    _check_cpu,
    _check_disk,
    _check_memory,
    _pending_approvals_on_disk,
    check_resource_health,
    cycle_had_no_trading_outcome,
    extended_hours_session_config,
    maybe_catch_up_session_cycle,
    maybe_retry_lost_cycle,
    run_order_reconciliation,
    run_position_reconciliation,
    run_resource_health_check,
    trading_day_key,
    within_extended_hours_window,
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

def _mock_engine(
    *,
    is_running=True,
    shutting_down=False,
    had_cycle_today=False,
    market_open=True,
    cycles_today=(),
):
    engine = MagicMock()
    engine.is_running = is_running
    engine._shutting_down = shutting_down
    engine.had_cycle_today.return_value = had_cycle_today
    engine.cycles_today.return_value = list(cycles_today)
    engine._broker.is_market_open.return_value = market_open
    return engine


def _blocked_cycle(cycle_id=1, orders_generated=5):
    """A cycle summary matching the "every order held by the news guard"
    signature: generated orders, but none submitted/queued/failed."""
    return {
        "cycle_id": cycle_id, "orders_generated": orders_generated,
        "orders_submitted": 0, "orders_queued": 0, "orders_failed": 0,
        "skipped": False, "error": None,
    }


def _successful_cycle(cycle_id=1, orders_submitted=3):
    return {
        "cycle_id": cycle_id, "orders_generated": orders_submitted,
        "orders_submitted": orders_submitted, "orders_queued": 0,
        "orders_failed": 0, "skipped": False, "error": None,
    }


def _errored_cycle(cycle_id=1, error="broker disconnect timed out"):
    """A cycle that ran but died before producing any trading outcome —
    e.g. a broker disconnect or a hard timeout — as opposed to a skipped
    cycle (market closed, shutdown, concurrent run already in progress)."""
    return {
        "cycle_id": cycle_id, "orders_generated": 0,
        "orders_submitted": 0, "orders_queued": 0, "orders_failed": 0,
        "skipped": False, "error": error,
    }


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
# cycle_had_no_trading_outcome / maybe_retry_lost_cycle
# ---------------------------------------------------------------------------

class TestCycleHadNoTradingOutcome:
    def test_true_for_the_fully_news_guard_blocked_signature(self):
        assert cycle_had_no_trading_outcome(_blocked_cycle()) is True

    def test_true_when_cycle_errored_even_with_orders_generated(self):
        summary = _blocked_cycle()
        summary["error"] = "broker disconnect timed out"
        assert cycle_had_no_trading_outcome(summary) is True

    def test_true_when_cycle_errored_with_nothing_generated(self):
        summary = _errored_cycle()
        assert cycle_had_no_trading_outcome(summary) is True

    def test_false_when_skipped_even_though_error_is_also_set(self):
        """A "skipped: market closed" cycle sets both skipped=True and
        error — skipped must take priority, since market-closed/shutdown/
        concurrent-cycle are not lost trading days worth retrying."""
        summary = _errored_cycle()
        summary["skipped"] = True
        assert cycle_had_no_trading_outcome(summary) is False

    def test_false_when_nothing_was_generated_and_no_error(self):
        summary = _blocked_cycle(orders_generated=0)
        assert cycle_had_no_trading_outcome(summary) is False

    def test_false_when_some_orders_submitted(self):
        summary = _successful_cycle()
        assert cycle_had_no_trading_outcome(summary) is False

    def test_false_when_some_orders_queued_for_approval(self):
        summary = _blocked_cycle()
        summary["orders_queued"] = 2
        assert cycle_had_no_trading_outcome(summary) is False

    def test_false_when_some_orders_failed(self):
        summary = _blocked_cycle()
        summary["orders_failed"] = 1
        assert cycle_had_no_trading_outcome(summary) is False


class TestMaybeRetryLostCycle:
    def test_skipped_when_engine_not_running(self):
        engine = _mock_engine(is_running=False, cycles_today=[_blocked_cycle()])
        maybe_retry_lost_cycle(engine)
        engine.cycles_today.assert_not_called()

    def test_skipped_when_engine_shutting_down(self):
        engine = _mock_engine(shutting_down=True, cycles_today=[_blocked_cycle()])
        maybe_retry_lost_cycle(engine)
        engine.cycles_today.assert_not_called()

    def test_skipped_on_weekend(self):
        engine = _mock_engine(cycles_today=[_blocked_cycle()])
        mock_dt = MagicMock(wraps=datetime)
        mock_dt.now.return_value = _some_saturday()
        with patch("firm.live.scheduler.datetime", mock_dt):
            maybe_retry_lost_cycle(engine)
        engine._broker.is_market_open.assert_not_called()

    def test_skipped_when_market_hours_check_raises(self):
        engine = _mock_engine(cycles_today=[_blocked_cycle()])
        engine._broker.is_market_open.side_effect = RuntimeError("boom")
        with _patched_weekday_now():
            maybe_retry_lost_cycle(engine)
        engine.cycles_today.assert_not_called()

    def test_skipped_when_market_closed(self):
        engine = _mock_engine(market_open=False, cycles_today=[_blocked_cycle()])
        with _patched_weekday_now():
            maybe_retry_lost_cycle(engine)
        engine.run_cycle.assert_not_called()

    def test_skipped_when_no_cycles_ran_today(self):
        engine = _mock_engine(cycles_today=[])
        with _patched_weekday_now():
            maybe_retry_lost_cycle(engine)
        engine.run_cycle.assert_not_called()

    def test_skipped_when_most_recent_cycle_today_was_not_fully_blocked(self):
        engine = _mock_engine(cycles_today=[_blocked_cycle(cycle_id=1), _successful_cycle(cycle_id=2)])
        with _patched_weekday_now():
            maybe_retry_lost_cycle(engine)
        engine.run_cycle.assert_not_called()

    def test_skipped_when_most_recent_cycle_today_was_a_market_closed_skip(self):
        """A market-closed skip is already covered by this function's own
        market-hours gate for *today's* live check, but a stale persisted
        skip from earlier in the day (e.g. a brief window right at the
        open/close boundary) must not itself trigger a retry."""
        summary = _errored_cycle()
        summary["skipped"] = True
        engine = _mock_engine(cycles_today=[summary])
        with _patched_weekday_now():
            maybe_retry_lost_cycle(engine)
        engine.run_cycle.assert_not_called()

    def test_retries_when_most_recent_cycle_today_was_fully_blocked(self):
        ran = threading.Event()
        engine = _mock_engine(cycles_today=[_blocked_cycle()])
        engine.run_cycle.side_effect = lambda **kwargs: ran.set()
        with _patched_weekday_now():
            maybe_retry_lost_cycle(engine)
        assert ran.wait(timeout=5.0), "retry thread never invoked run_cycle()"
        engine.run_cycle.assert_called_once_with(cycle_type=None)

    def test_retries_when_most_recent_cycle_today_errored_out(self):
        ran = threading.Event()
        engine = _mock_engine(cycles_today=[_errored_cycle(error="IB Gateway disconnect timed out")])
        engine.run_cycle.side_effect = lambda **kwargs: ran.set()
        with _patched_weekday_now():
            maybe_retry_lost_cycle(engine)
        assert ran.wait(timeout=5.0), "retry thread never invoked run_cycle()"
        engine.run_cycle.assert_called_once_with(cycle_type=None)

    def test_retry_forwards_explicit_cycle_type(self):
        """The "hourly_market_hours" schedule hardcodes cycle_type="intraday"
        for this job (see TradingScheduler.start()) — verify it's forwarded
        straight through to engine.run_cycle rather than dropped."""
        ran = threading.Event()
        engine = _mock_engine(cycles_today=[_blocked_cycle()])
        engine.run_cycle.side_effect = lambda **kwargs: ran.set()
        with _patched_weekday_now():
            maybe_retry_lost_cycle(engine, cycle_type="intraday")
        assert ran.wait(timeout=5.0), "retry thread never invoked run_cycle()"
        engine.run_cycle.assert_called_once_with(cycle_type="intraday")

    def test_retry_thread_logs_and_swallows_run_cycle_errors(self):
        done = threading.Event()
        engine = _mock_engine(cycles_today=[_blocked_cycle()])

        def _boom(**kwargs):
            done.set()
            raise RuntimeError("pipeline exploded")

        engine.run_cycle.side_effect = _boom
        with _patched_weekday_now():
            maybe_retry_lost_cycle(engine)
        assert done.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# run_order_reconciliation
# ---------------------------------------------------------------------------

class TestRunOrderReconciliation:
    def test_skipped_when_engine_not_running(self):
        engine = _mock_engine(is_running=False)
        run_order_reconciliation(engine)
        engine.reconcile_order_history.assert_not_called()

    def test_skipped_when_engine_shutting_down(self):
        engine = _mock_engine(shutting_down=True)
        run_order_reconciliation(engine)
        engine.reconcile_order_history.assert_not_called()

    def test_delegates_to_engine_reconcile_order_history(self):
        engine = _mock_engine()
        run_order_reconciliation(engine)
        engine.reconcile_order_history.assert_called_once()

    def test_swallows_engine_errors(self):
        engine = _mock_engine()
        engine.reconcile_order_history.side_effect = RuntimeError("boom")
        run_order_reconciliation(engine)  # must not raise


# ---------------------------------------------------------------------------
# run_position_reconciliation
# ---------------------------------------------------------------------------

class TestRunPositionReconciliation:
    def test_skipped_when_engine_not_running(self):
        engine = _mock_engine(is_running=False)
        run_position_reconciliation(engine)
        engine.check_reconciliation.assert_not_called()

    def test_skipped_when_engine_shutting_down(self):
        engine = _mock_engine(shutting_down=True)
        run_position_reconciliation(engine)
        engine.check_reconciliation.assert_not_called()

    def test_delegates_to_engine_check_reconciliation(self):
        engine = _mock_engine()
        run_position_reconciliation(engine)
        engine.check_reconciliation.assert_called_once()

    def test_swallows_engine_errors(self):
        engine = _mock_engine()
        engine.check_reconciliation.side_effect = RuntimeError("boom")
        run_position_reconciliation(engine)  # must not raise

    def test_first_mismatch_alerts(self):
        """Regression: check_reconciliation itself no longer alerts (moved
        here so a plain GET /api/live/reconciliation doesn't push a
        notification) -- the job must alert on the first observed mismatch."""
        engine = _mock_engine()
        engine.check_reconciliation.return_value = {
            "status": "mismatch",
            "discrepancies": [{"type": "cash_mismatch", "internal": 100.0, "broker": 90.0}],
        }
        state: dict[str, str] = {}
        run_position_reconciliation(engine, state)
        engine._emit_alert.assert_called_once()
        args = engine._emit_alert.call_args
        assert args[0][0] == "portfolio_reconciliation_mismatch"
        assert args[0][1] == "warning"
        assert state["position_reconciliation"] == "mismatch"

    def test_repeated_mismatch_does_not_realert(self):
        """The bug this fixes: a sustained mismatch (e.g. a multi-hour
        broker-data outage blocking the sync that would resolve it) was
        re-alerting the full discrepancy list on every 30-minute tick."""
        engine = _mock_engine()
        engine.check_reconciliation.return_value = {
            "status": "mismatch",
            "discrepancies": [{"type": "cash_mismatch", "internal": 100.0, "broker": 90.0}],
        }
        state: dict[str, str] = {}
        run_position_reconciliation(engine, state)
        run_position_reconciliation(engine, state)
        run_position_reconciliation(engine, state)
        engine._emit_alert.assert_called_once()

    def test_recovery_alerts_once_then_stays_quiet(self):
        engine = _mock_engine()
        engine.check_reconciliation.return_value = {"status": "mismatch", "discrepancies": []}
        state: dict[str, str] = {}
        run_position_reconciliation(engine, state)
        engine._emit_alert.reset_mock()

        engine.check_reconciliation.return_value = {"status": "ok", "discrepancies": []}
        run_position_reconciliation(engine, state)
        engine._emit_alert.assert_called_once_with(
            "portfolio_reconciliation_recovered", "info",
            "Broker/internal reconciliation is back in sync.",
        )

        engine._emit_alert.reset_mock()
        run_position_reconciliation(engine, state)  # still "ok" -- no repeat
        engine._emit_alert.assert_not_called()

    def test_ok_from_the_start_never_alerts(self):
        engine = _mock_engine()
        engine.check_reconciliation.return_value = {"status": "ok", "discrepancies": []}
        run_position_reconciliation(engine, {})
        engine._emit_alert.assert_not_called()

    def test_unknown_status_does_not_alert_or_change_state(self):
        """A broker-query failure ("unknown") must never be treated as
        either a mismatch or a recovery -- it says nothing about whether
        internal state actually matches the broker."""
        engine = _mock_engine()
        engine.check_reconciliation.return_value = {"status": "mismatch", "discrepancies": []}
        state: dict[str, str] = {}
        run_position_reconciliation(engine, state)
        engine._emit_alert.reset_mock()

        engine.check_reconciliation.return_value = {"status": "unknown", "discrepancies": []}
        run_position_reconciliation(engine, state)
        engine._emit_alert.assert_not_called()
        assert state["position_reconciliation"] == "unknown"


# ---------------------------------------------------------------------------
# Resource health check (disk / memory / CPU)
# ---------------------------------------------------------------------------

class TestCheckDisk:
    def test_ok_when_plenty_of_space(self, tmp_path):
        usage = MagicMock(total=100 * 1024 ** 3, used=10 * 1024 ** 3, free=90 * 1024 ** 3)
        with patch("firm.live.scheduler.shutil.disk_usage", return_value=usage):
            result = _check_disk(str(tmp_path))
        assert result["severity"] == "ok"
        assert result["free_gb"] == pytest.approx(90.0, abs=0.1)

    def test_warning_below_free_threshold(self):
        usage = MagicMock(total=20 * 1024 ** 3, used=16 * 1024 ** 3, free=4 * 1024 ** 3)
        with patch("firm.live.scheduler.shutil.disk_usage", return_value=usage):
            result = _check_disk(".")
        assert result["severity"] == "warning"

    def test_critical_below_free_threshold(self):
        usage = MagicMock(total=20 * 1024 ** 3, used=19 * 1024 ** 3, free=1 * 1024 ** 3)
        with patch("firm.live.scheduler.shutil.disk_usage", return_value=usage):
            result = _check_disk(".")
        assert result["severity"] == "critical"

    def test_critical_from_used_pct_even_with_free_gb_above_threshold(self):
        # A much bigger disk can cross the used-% threshold while still
        # having several GB nominally free -- either signal alone must
        # be enough to trip the check.
        usage = MagicMock(total=1000 * 1024 ** 3, used=970 * 1024 ** 3, free=30 * 1024 ** 3)
        with patch("firm.live.scheduler.shutil.disk_usage", return_value=usage):
            result = _check_disk(".")
        assert result["severity"] == "critical"

    def test_returns_none_on_oserror(self):
        with patch("firm.live.scheduler.shutil.disk_usage", side_effect=OSError("nope")):
            assert _check_disk("/does/not/exist") is None

    def test_thresholds_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("HEALTH_CHECK_DISK_WARN_FREE_GB", "1000")
        usage = MagicMock(total=100 * 1024 ** 3, used=10 * 1024 ** 3, free=90 * 1024 ** 3)
        with patch("firm.live.scheduler.shutil.disk_usage", return_value=usage):
            result = _check_disk(".")
        assert result["severity"] == "warning"  # 90GB free now below the overridden 1000GB bar

    def test_invalid_env_override_falls_back_to_default(self, monkeypatch, caplog):
        monkeypatch.setenv("HEALTH_CHECK_DISK_WARN_FREE_GB", "not-a-number")
        usage = MagicMock(total=100 * 1024 ** 3, used=10 * 1024 ** 3, free=90 * 1024 ** 3)
        with patch("firm.live.scheduler.shutil.disk_usage", return_value=usage):
            result = _check_disk(".")
        assert result["severity"] == "ok"  # default (5GB) used instead of the bad override


class TestCheckMemory:
    def test_ok_when_plenty_available(self):
        meminfo = "MemTotal:       10000000 kB\nMemAvailable:    5000000 kB\n"
        with patch("builtins.open", return_value=_fake_file(meminfo)):
            result = _check_memory()
        assert result["severity"] == "ok"
        assert result["available_pct"] == pytest.approx(50.0, abs=0.1)

    def test_critical_when_available_low(self):
        meminfo = "MemTotal:       10000000 kB\nMemAvailable:     500000 kB\n"
        with patch("builtins.open", return_value=_fake_file(meminfo)):
            result = _check_memory()
        assert result["severity"] == "critical"

    def test_returns_none_on_oserror(self):
        with patch("builtins.open", side_effect=OSError("no /proc here")):
            assert _check_memory() is None

    def test_returns_none_when_memtotal_missing(self):
        with patch("builtins.open", return_value=_fake_file("SomeOtherField: 1 kB\n")):
            assert _check_memory() is None


class TestCheckCpu:
    def test_ok_under_light_load(self):
        with patch("firm.live.scheduler.os.getloadavg", return_value=(0.1, 0.2, 0.1)), \
             patch("firm.live.scheduler.os.cpu_count", return_value=2):
            result = _check_cpu()
        assert result["severity"] == "ok"
        assert result["load_per_core"] == pytest.approx(0.1, abs=0.01)

    def test_critical_under_heavy_sustained_load(self):
        with patch("firm.live.scheduler.os.getloadavg", return_value=(7.0, 7.0, 7.0)), \
             patch("firm.live.scheduler.os.cpu_count", return_value=2):
            result = _check_cpu()
        assert result["severity"] == "critical"

    def test_returns_none_on_oserror(self):
        with patch("firm.live.scheduler.os.getloadavg", side_effect=OSError("unsupported")):
            assert _check_cpu() is None


def _fake_file(contents: str):
    """A context-manager mock standing in for ``open(...)`` returning *contents*."""
    handle = MagicMock()
    handle.__enter__.return_value = contents.splitlines(keepends=True)
    handle.__exit__.return_value = False
    return handle


class TestCheckResourceHealth:
    def test_aggregates_all_three_metrics_when_all_measurable(self):
        ok_disk = MagicMock(total=100 * 1024 ** 3, used=1 * 1024 ** 3, free=99 * 1024 ** 3)
        meminfo = "MemTotal:       10000000 kB\nMemAvailable:    5000000 kB\n"
        with patch("firm.live.scheduler.shutil.disk_usage", return_value=ok_disk), \
             patch("builtins.open", return_value=_fake_file(meminfo)), \
             patch("firm.live.scheduler.os.getloadavg", return_value=(0.1, 0.1, 0.1)), \
             patch("firm.live.scheduler.os.cpu_count", return_value=2):
            result = check_resource_health()
        assert set(result) == {"disk", "memory", "cpu"}
        assert all(m["severity"] == "ok" for m in result.values())

    def test_one_unmeasurable_metric_does_not_block_the_others(self):
        ok_disk = MagicMock(total=100 * 1024 ** 3, used=1 * 1024 ** 3, free=99 * 1024 ** 3)
        with patch("firm.live.scheduler.shutil.disk_usage", return_value=ok_disk), \
             patch("builtins.open", side_effect=OSError("no /proc here")), \
             patch("firm.live.scheduler.os.getloadavg", return_value=(0.1, 0.1, 0.1)), \
             patch("firm.live.scheduler.os.cpu_count", return_value=2):
            result = check_resource_health()
        assert set(result) == {"disk", "cpu"}  # memory dropped, not raised


class TestRunResourceHealthCheck:
    def test_alerts_once_on_transition_into_a_breach(self):
        engine = MagicMock()
        state: dict[str, str] = {}
        with patch(
            "firm.live.scheduler.check_resource_health",
            return_value={"disk": {"severity": "warning", "message": "low disk"}},
        ):
            run_resource_health_check(engine, state)
        engine._emit_alert.assert_called_once_with(
            "host_disk_low", "warning", "low disk",
        )
        assert state == {"disk": "warning"}

    def test_does_not_realert_while_severity_is_unchanged(self):
        engine = MagicMock()
        state = {"disk": "warning"}
        with patch(
            "firm.live.scheduler.check_resource_health",
            return_value={"disk": {"severity": "warning", "message": "still low"}},
        ):
            run_resource_health_check(engine, state)
        engine._emit_alert.assert_not_called()

    def test_alerts_on_escalation(self):
        engine = MagicMock()
        state = {"disk": "warning"}
        with patch(
            "firm.live.scheduler.check_resource_health",
            return_value={"disk": {"severity": "critical", "message": "critical now"}},
        ):
            run_resource_health_check(engine, state)
        engine._emit_alert.assert_called_once_with(
            "host_disk_low", "critical", "critical now",
        )
        assert state == {"disk": "critical"}

    def test_alerts_recovery_as_info_with_recovered_kind(self):
        engine = MagicMock()
        state = {"disk": "critical"}
        with patch(
            "firm.live.scheduler.check_resource_health",
            return_value={"disk": {"severity": "ok", "message": "back to normal"}},
        ):
            run_resource_health_check(engine, state)
        engine._emit_alert.assert_called_once_with(
            "host_disk_low_recovered", "info", "back to normal",
        )
        assert state == {"disk": "ok"}

    def test_does_not_gate_on_engine_running_or_shutting_down(self):
        # Unlike run_order_reconciliation/maybe_retry_lost_cycle, a resource
        # crunch matters whether or not the trading engine is active.
        engine = _mock_engine(is_running=False, shutting_down=True)
        with patch(
            "firm.live.scheduler.check_resource_health",
            return_value={"disk": {"severity": "critical", "message": "low"}},
        ):
            run_resource_health_check(engine, {})
        engine._emit_alert.assert_called_once()

    def test_swallows_measurement_errors(self):
        engine = MagicMock()
        with patch(
            "firm.live.scheduler.check_resource_health", side_effect=RuntimeError("boom"),
        ):
            run_resource_health_check(engine, {})  # must not raise
        engine._emit_alert.assert_not_called()

    def test_defaults_to_a_fresh_state_dict_when_none_given(self):
        engine = MagicMock()
        with patch(
            "firm.live.scheduler.check_resource_health",
            return_value={"disk": {"severity": "warning", "message": "low"}},
        ):
            run_resource_health_check(engine)  # no state arg -- must not raise
        engine._emit_alert.assert_called_once()


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

    def test_order_reconciliation_job_added_regardless_of_schedule(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="hourly")
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._order_reconciliation_job_id) is not None
        finally:
            sched.stop()

    def test_resource_health_job_added_regardless_of_schedule(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="hourly")
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._resource_health_job_id) is not None
        finally:
            sched.stop()

    def test_position_reconciliation_job_added_regardless_of_schedule(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="hourly")
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._position_reconciliation_job_id) is not None
        finally:
            sched.stop()

    def test_lost_cycle_retry_job_added_for_session_schedule(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="market_open")
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._lost_cycle_retry_job_id) is not None
        finally:
            sched.stop()

    def test_lost_cycle_retry_job_not_added_for_non_session_schedule(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="hourly")
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._lost_cycle_retry_job_id) is None
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

    def test_news_ingestion_job_not_added_by_default(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open", universe=["AAPL", "MSFT"],
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._news_ingestion_job_id) is None
        finally:
            sched.stop()

    def test_news_ingestion_job_not_added_when_disabled_explicitly(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open", universe=["AAPL", "MSFT"],
            news_ingestion={"enabled": False},
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._news_ingestion_job_id) is None
        finally:
            sched.stop()

    def test_news_ingestion_job_not_added_without_universe_even_if_enabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            news_ingestion={"enabled": True},
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._news_ingestion_job_id) is None
        finally:
            sched.stop()

    def test_news_ingestion_job_added_when_enabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open", universe=["AAPL", "MSFT"],
            news_ingestion={"enabled": True, "hour": 6, "days": 2},
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._news_ingestion_job_id)
            assert job is not None
            assert str(job.trigger.fields[job.trigger.FIELD_NAMES.index("hour")]) == "6"
        finally:
            sched.stop()

    def test_news_ingestion_job_fires_with_configured_universe_and_days(self):
        # Patched *before* start() — the job callback does a local
        # ``from firm.live.news_ingestion_job import run_scheduled_news_
        # ingestion`` at job-registration time inside start(), so the patch
        # must be in place before that import executes for the lambda's
        # closed-over reference to resolve to the mock.
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open", universe=["AAPL", "MSFT"],
            news_ingestion={"enabled": True, "days": 5},
        )
        try:
            with patch(
                "firm.live.news_ingestion_job.run_scheduled_news_ingestion"
            ) as mock_run:
                sched.start()
                job = sched._scheduler.get_job(sched._news_ingestion_job_id)
                job.func()
            mock_run.assert_called_once_with(["AAPL", "MSFT"], days=5)
        finally:
            sched.stop()

    def test_capital_reallocation_job_not_added_by_default(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="market_open")
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._capital_reallocation_job_id) is None
        finally:
            sched.stop()

    def test_capital_reallocation_job_not_added_when_disabled_explicitly(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            capital_reallocation={"enabled": False},
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._capital_reallocation_job_id) is None
        finally:
            sched.stop()

    def test_capital_reallocation_job_added_when_enabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            capital_reallocation={"enabled": True, "day_of_week": "mon", "hour": 5},
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._capital_reallocation_job_id)
            assert job is not None
            assert str(job.trigger.fields[job.trigger.FIELD_NAMES.index("hour")]) == "5"
        finally:
            sched.stop()

    def test_capital_reallocation_job_fires_scheduled_check(self):
        # Patched before start() -- the job callback does a local import at
        # registration time inside start(), same reason as the equivalent
        # news_ingestion test above.
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            capital_reallocation={"enabled": True},
        )
        try:
            with patch(
                "firm.live.capital_reallocation_job.run_scheduled_capital_reallocation_check"
            ) as mock_run:
                sched.start()
                job = sched._scheduler.get_job(sched._capital_reallocation_job_id)
                job.func()
            mock_run.assert_called_once_with(engine)
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

    def test_next_lost_cycle_retry_returns_datetime_for_session_schedule(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="market_open")
        try:
            sched.start()
            nxt = sched.next_lost_cycle_retry()
            assert nxt is not None
            assert isinstance(nxt, datetime)
        finally:
            sched.stop()

    def test_next_lost_cycle_retry_none_for_non_session_schedule(self):
        # An interval/hourly schedule never registers the retry job at all
        # (see test_lost_cycle_retry_job_not_added_for_non_session_schedule)
        # since it already retries naturally on its own next regular tick.
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="hourly")
        try:
            sched.start()
            assert sched.next_lost_cycle_retry() is None
        finally:
            sched.stop()

    def test_next_lost_cycle_retry_none_before_start(self):
        sched = TradingScheduler(engine=MagicMock(), schedule="market_open")
        assert sched.next_lost_cycle_retry() is None


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
        engine.run_cycle.assert_called_once_with(cycle_type=None)

    def test_forwards_explicit_cycle_type(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine)
        sched._run_cycle_safe(cycle_type="open")
        engine.run_cycle.assert_called_once_with(cycle_type="open")

    def test_logs_and_swallows_exception(self, caplog):
        engine = MagicMock()
        engine.run_cycle.side_effect = ValueError("nope")
        sched = TradingScheduler(engine=engine)
        with caplog.at_level("ERROR"):
            sched._run_cycle_safe()
        assert any("Scheduled cycle failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# "hourly_market_hours" composite schedule (added 2026-09-18)
# ---------------------------------------------------------------------------

class TestHourlyMarketHoursSchedule:
    """The "hourly_market_hours" schedule (config/live.yaml's ``schedule:``)
    registers three separate jobs instead of the usual single "live_cycle"
    job -- an open leg (9:30, cycle_type="open"), an intraday leg
    (hour="10-14", minute=30, cycle_type="intraday"), and a close-anchor leg
    (15:50, cycle_type="close") -- see
    TradingScheduler._start_hourly_market_hours_jobs.
    """

    def test_registers_three_jobs_not_the_single_job(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule=HOURLY_MARKET_HOURS)
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._job_id) is not None
            assert sched._scheduler.get_job(sched._intraday_job_id) is not None
            assert sched._scheduler.get_job(sched._close_job_id) is not None
        finally:
            sched.stop()

    def test_open_leg_cron_matches_market_open_preset(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule=HOURLY_MARKET_HOURS)
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._job_id)
            trigger = job.trigger
            assert str(trigger.fields[trigger.FIELD_NAMES.index("hour")]) == "9"
            assert str(trigger.fields[trigger.FIELD_NAMES.index("minute")]) == "30"
        finally:
            sched.stop()

    def test_intraday_leg_restricted_to_market_hours(self):
        """Unlike the plain "hourly" preset (unrestricted hours), this leg's
        cron hour field is itself restricted to 10-14 so it never fires
        outside RTH at all (no wasted provider-fetch)."""
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule=HOURLY_MARKET_HOURS)
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._intraday_job_id)
            trigger = job.trigger
            assert str(trigger.fields[trigger.FIELD_NAMES.index("hour")]) == "10-14"
            assert str(trigger.fields[trigger.FIELD_NAMES.index("minute")]) == "30"
        finally:
            sched.stop()

    def test_close_leg_fires_a_few_minutes_before_market_close(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule=HOURLY_MARKET_HOURS)
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._close_job_id)
            trigger = job.trigger
            assert str(trigger.fields[trigger.FIELD_NAMES.index("hour")]) == "15"
            assert str(trigger.fields[trigger.FIELD_NAMES.index("minute")]) == "50"
        finally:
            sched.stop()

    def test_each_leg_passes_its_own_explicit_cycle_type(self):
        """The whole point of three separate jobs instead of one generic
        job inferring the cycle type from the wall clock: each leg's own
        callback already knows which leg it is."""
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule=HOURLY_MARKET_HOURS)
        try:
            sched.start()
            for job_id, expected_cycle_type in (
                (sched._job_id, "open"),
                (sched._intraday_job_id, "intraday"),
                (sched._close_job_id, "close"),
            ):
                engine.reset_mock()
                sched._scheduler.get_job(job_id).func()
                engine.run_cycle.assert_called_once_with(cycle_type=expected_cycle_type)
        finally:
            sched.stop()

    def test_lost_cycle_retry_job_registered_for_hourly_market_hours(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule=HOURLY_MARKET_HOURS)
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._lost_cycle_retry_job_id) is not None
        finally:
            sched.stop()

    def test_lost_cycle_retry_uses_intraday_cycle_type(self):
        """The retry job hardcodes cycle_type="intraday" (the conservative,
        cost-safe default) regardless of which leg actually failed -- see
        TradingScheduler.start()'s comment above this job's registration."""
        ran = threading.Event()
        engine = _mock_engine(cycles_today=[_blocked_cycle()])
        engine.run_cycle.side_effect = lambda **kwargs: ran.set()
        sched = TradingScheduler(engine=engine, schedule=HOURLY_MARKET_HOURS)
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._lost_cycle_retry_job_id)
            with _patched_weekday_now():
                job.func()
            assert ran.wait(timeout=5.0), "retry thread never invoked run_cycle()"
            engine.run_cycle.assert_called_once_with(cycle_type="intraday")
        finally:
            sched.stop()

    def test_next_run_returns_earliest_leg_when_job_id_omitted(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule=HOURLY_MARKET_HOURS)
        try:
            sched.start()
            overall = sched.next_run()
            per_leg = [
                sched.next_run(jid)
                for jid in (sched._job_id, sched._intraday_job_id, sched._close_job_id)
            ]
            assert overall is not None
            assert overall == min(per_leg)
        finally:
            sched.stop()

    def test_next_run_still_honours_an_explicit_job_id(self):
        """Passing an explicit job id bypasses the "earliest across all
        three legs" behavior and returns that one job's own next fire."""
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule=HOURLY_MARKET_HOURS)
        try:
            sched.start()
            close_job = sched._scheduler.get_job(sched._close_job_id)
            assert sched.next_run(sched._close_job_id) == close_job.next_run_time
        finally:
            sched.stop()

    def test_next_run_unaffected_for_non_composite_schedules(self):
        """Every other schedule keeps resolving next_run() to its single
        job exactly as before -- the composite branch only activates for
        HOURLY_MARKET_HOURS."""
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="market_open")
        try:
            sched.start()
            assert sched.next_run() == sched.next_run(sched._job_id)
        finally:
            sched.stop()


# ---------------------------------------------------------------------------
# Extended-hours trading (opt-in, off by default; added 2026-09-19)
# ---------------------------------------------------------------------------

class TestExtendedHoursSessionConfig:
    def test_none_when_cfg_is_none(self):
        assert extended_hours_session_config(None, "premarket") is None

    def test_none_when_top_level_disabled(self):
        cfg = {"enabled": False, "premarket": {"enabled": True}}
        assert extended_hours_session_config(cfg, "premarket") is None

    def test_none_when_top_level_enabled_but_session_missing(self):
        cfg = {"enabled": True}
        assert extended_hours_session_config(cfg, "premarket") is None

    def test_none_when_top_level_enabled_but_session_disabled(self):
        cfg = {"enabled": True, "premarket": {"enabled": False}}
        assert extended_hours_session_config(cfg, "premarket") is None

    def test_returns_session_dict_when_both_enabled(self):
        cfg = {"enabled": True, "premarket": {"enabled": True, "start": "05:00"}}
        assert extended_hours_session_config(cfg, "premarket") == {
            "enabled": True, "start": "05:00",
        }

    def test_sessions_are_independent(self):
        cfg = {
            "enabled": True,
            "premarket": {"enabled": True},
            "afterhours": {"enabled": False},
        }
        assert extended_hours_session_config(cfg, "premarket") is not None
        assert extended_hours_session_config(cfg, "afterhours") is None


class TestWithinExtendedHoursWindow:
    """EDT (UTC-4) is in effect for every date used below (2026-07-27, a
    Monday), so naive-UTC inputs are converted the same way
    trading_day_key's own tests verify: treated as UTC, then shifted -4h
    into US/Eastern.
    """

    _ENABLED_PREMARKET = {"enabled": True, "premarket": {"enabled": True}}
    _ENABLED_AFTERHOURS = {"enabled": True, "afterhours": {"enabled": True}}

    def test_false_when_feature_disabled(self):
        now = datetime(2026, 7, 27, 12, 0)  # 08:00 ET -- inside the window
        assert within_extended_hours_window(now, "premarket", {}) is False
        assert within_extended_hours_window(now, "premarket", None) is False

    def test_false_when_session_disabled(self):
        now = datetime(2026, 7, 27, 12, 0)  # 08:00 ET
        cfg = {"enabled": True, "premarket": {"enabled": False}}
        assert within_extended_hours_window(now, "premarket", cfg) is False

    def test_true_inside_default_premarket_window(self):
        now = datetime(2026, 7, 27, 12, 0)  # 08:00 ET, within 04:00-09:30
        assert within_extended_hours_window(
            now, "premarket", self._ENABLED_PREMARKET
        ) is True

    def test_false_outside_default_premarket_window_same_day(self):
        now = datetime(2026, 7, 27, 14, 0)  # 10:00 ET, past the 09:30 end
        assert within_extended_hours_window(
            now, "premarket", self._ENABLED_PREMARKET
        ) is False

    def test_false_on_weekend_even_inside_time_window(self):
        now = datetime(2026, 7, 25, 12, 0)  # Saturday, 08:00 ET
        assert within_extended_hours_window(
            now, "premarket", self._ENABLED_PREMARKET
        ) is False

    def test_true_inside_default_afterhours_window(self):
        now = datetime(2026, 7, 27, 21, 0)  # 17:00 ET, within 16:00-20:00
        assert within_extended_hours_window(
            now, "afterhours", self._ENABLED_AFTERHOURS
        ) is True

    def test_false_outside_default_afterhours_window(self):
        now = datetime(2026, 7, 28, 1, 0)  # 21:00 ET the prior day, past 20:00 end
        assert within_extended_hours_window(
            now, "afterhours", self._ENABLED_AFTERHOURS
        ) is False

    def test_custom_start_end_override_defaults(self):
        cfg = {
            "enabled": True,
            "premarket": {"enabled": True, "start": "05:00", "end": "06:00"},
        }
        inside = datetime(2026, 7, 27, 9, 30)  # 05:30 ET
        outside = datetime(2026, 7, 27, 12, 0)  # 08:00 ET, outside the narrowed window
        assert within_extended_hours_window(inside, "premarket", cfg) is True
        assert within_extended_hours_window(outside, "premarket", cfg) is False

    def test_aware_datetime_also_supported(self):
        aware = datetime(2026, 7, 27, 12, 0, tzinfo=dt_tz.utc)
        assert within_extended_hours_window(
            aware, "premarket", self._ENABLED_PREMARKET
        ) is True

    def test_end_boundary_is_exclusive(self):
        # 09:30 ET exactly -- the configured end -- must not count as "inside".
        now = datetime(2026, 7, 27, 13, 30)
        assert within_extended_hours_window(
            now, "premarket", self._ENABLED_PREMARKET
        ) is False

    def test_start_boundary_is_inclusive(self):
        # 04:00 ET exactly -- the configured start.
        now = datetime(2026, 7, 27, 8, 0)
        assert within_extended_hours_window(
            now, "premarket", self._ENABLED_PREMARKET
        ) is True


class TestExtendedHoursCycleTypes:
    def test_contains_exactly_premarket_and_afterhours(self):
        assert EXTENDED_HOURS_CYCLE_TYPES == frozenset({"premarket", "afterhours"})


class TestExtendedHoursSchedule:
    """TradingScheduler's opt-in premarket/afterhours job registration (see
    TradingScheduler._start_extended_hours_jobs). Additive to whichever
    ``schedule`` preset/composite is already active, and off unless
    ``extended_hours_trading["enabled"]`` is explicitly true.
    """

    def test_no_jobs_registered_by_default(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="market_open")
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._premarket_job_id) is None
            assert sched._scheduler.get_job(sched._afterhours_job_id) is None
        finally:
            sched.stop()

    def test_no_jobs_registered_when_top_level_disabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            extended_hours_trading={
                "enabled": False,
                "premarket": {"enabled": True},
                "afterhours": {"enabled": True},
            },
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._premarket_job_id) is None
            assert sched._scheduler.get_job(sched._afterhours_job_id) is None
        finally:
            sched.stop()

    def test_premarket_job_registered_when_enabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            extended_hours_trading={
                "enabled": True, "premarket": {"enabled": True},
            },
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._premarket_job_id) is not None
            assert sched._scheduler.get_job(sched._afterhours_job_id) is None
        finally:
            sched.stop()

    def test_afterhours_job_registered_when_enabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            extended_hours_trading={
                "enabled": True, "afterhours": {"enabled": True},
            },
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._afterhours_job_id) is not None
            assert sched._scheduler.get_job(sched._premarket_job_id) is None
        finally:
            sched.stop()

    def test_both_sessions_registered_when_both_enabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            extended_hours_trading={
                "enabled": True,
                "premarket": {"enabled": True},
                "afterhours": {"enabled": True},
            },
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._premarket_job_id) is not None
            assert sched._scheduler.get_job(sched._afterhours_job_id) is not None
        finally:
            sched.stop()

    def test_premarket_job_honours_custom_schedule_cron(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            extended_hours_trading={
                "enabled": True,
                "premarket": {"enabled": True, "schedule": "cron:05:15"},
            },
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._premarket_job_id)
            assert str(job.trigger.fields[job.trigger.FIELD_NAMES.index("hour")]) == "5"
            assert str(job.trigger.fields[job.trigger.FIELD_NAMES.index("minute")]) == "15"
        finally:
            sched.stop()

    def test_premarket_job_fires_with_premarket_cycle_type(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            extended_hours_trading={
                "enabled": True, "premarket": {"enabled": True},
            },
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._premarket_job_id)
            job.func()
        finally:
            sched.stop()
        engine.run_cycle.assert_called_once_with(cycle_type="premarket")

    def test_afterhours_job_fires_with_afterhours_cycle_type(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            extended_hours_trading={
                "enabled": True, "afterhours": {"enabled": True},
            },
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._afterhours_job_id)
            job.func()
        finally:
            sched.stop()
        engine.run_cycle.assert_called_once_with(cycle_type="afterhours")

    def test_additive_to_hourly_market_hours_composite_schedule(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule=HOURLY_MARKET_HOURS,
            extended_hours_trading={
                "enabled": True,
                "premarket": {"enabled": True},
                "afterhours": {"enabled": True},
            },
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._job_id) is not None
            assert sched._scheduler.get_job(sched._intraday_job_id) is not None
            assert sched._scheduler.get_job(sched._close_job_id) is not None
            assert sched._scheduler.get_job(sched._premarket_job_id) is not None
            assert sched._scheduler.get_job(sched._afterhours_job_id) is not None
        finally:
            sched.stop()


class TestPlanningJobSchedule:
    """TradingScheduler's opt-in pre-open "planning" job registration (see
    TradingScheduler._start_planning_job and firm.live.planning_cycle).
    Single job, off unless planning_cycle["enabled"] is explicitly true --
    mirrors TestExtendedHoursSchedule's pattern above."""

    def test_no_job_registered_by_default(self):
        engine = MagicMock()
        sched = TradingScheduler(engine=engine, schedule="market_open")
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._planning_job_id) is None
        finally:
            sched.stop()

    def test_no_job_registered_when_disabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            planning_cycle={"enabled": False},
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._planning_job_id) is None
        finally:
            sched.stop()

    def test_job_registered_when_enabled(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            planning_cycle={"enabled": True},
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._planning_job_id) is not None
        finally:
            sched.stop()

    def test_honours_custom_schedule_cron(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            planning_cycle={"enabled": True, "schedule": "cron:02:30"},
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._planning_job_id)
            assert str(job.trigger.fields[job.trigger.FIELD_NAMES.index("hour")]) == "2"
            assert str(job.trigger.fields[job.trigger.FIELD_NAMES.index("minute")]) == "30"
        finally:
            sched.stop()

    def test_defaults_to_nine_fifteen_am(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            planning_cycle={"enabled": True},
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._planning_job_id)
            assert str(job.trigger.fields[job.trigger.FIELD_NAMES.index("hour")]) == "9"
            assert str(job.trigger.fields[job.trigger.FIELD_NAMES.index("minute")]) == "15"
        finally:
            sched.stop()

    def test_job_fires_with_planning_cycle_type(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule="market_open",
            planning_cycle={"enabled": True},
        )
        try:
            sched.start()
            job = sched._scheduler.get_job(sched._planning_job_id)
            job.func()
        finally:
            sched.stop()
        engine.run_cycle.assert_called_once_with(cycle_type="planning")

    def test_additive_to_hourly_market_hours_composite_schedule(self):
        engine = MagicMock()
        sched = TradingScheduler(
            engine=engine, schedule=HOURLY_MARKET_HOURS,
            planning_cycle={"enabled": True},
        )
        try:
            sched.start()
            assert sched._scheduler.get_job(sched._job_id) is not None
            assert sched._scheduler.get_job(sched._planning_job_id) is not None
        finally:
            sched.stop()
