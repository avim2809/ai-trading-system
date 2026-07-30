"""Tests for firm.api.routers.live.auto_start_live_with_retries.

This is the safety net added after a 2026-07-29 boot-race outage: IB
Gateway's headless login can rarely still be mid-flight past
scripts/wait_for_ibgateway.sh's wait, so a single auto-start attempt at
boot silently left the live engine stopped for hours with no alert. These
tests drive the coroutine directly with asyncio.run() (no pytest-asyncio
dependency needed) and monkeypatch bootstrap_live_from_yaml/
build_alert_callback so every case runs with delays=(0, ...) instead of
real sleeps.
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import MagicMock

import pytest

from firm.api.routers import live as live_router


def _fake_app(engine=None) -> types.SimpleNamespace:
    return types.SimpleNamespace(state=types.SimpleNamespace(live_engine=engine))


def _fake_engine(is_running: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(is_running=is_running)


class TestAutoStartLiveWithRetries:
    def test_noop_when_flag_not_set(self, monkeypatch):
        monkeypatch.delenv("FIRM_AUTO_START_LIVE", raising=False)
        bootstrap = MagicMock()
        monkeypatch.setattr(live_router, "bootstrap_live_from_yaml", bootstrap)

        app = _fake_app()
        result = asyncio.run(live_router.auto_start_live_with_retries(app, delays=(0,)))

        assert result is False
        bootstrap.assert_not_called()

    def test_succeeds_on_first_try(self, monkeypatch):
        monkeypatch.setenv("FIRM_AUTO_START_LIVE", "1")
        app = _fake_app()

        def _bootstrap(a):
            a.state.live_engine = _fake_engine(is_running=True)

        bootstrap = MagicMock(side_effect=_bootstrap)
        monkeypatch.setattr(live_router, "bootstrap_live_from_yaml", bootstrap)

        result = asyncio.run(
            live_router.auto_start_live_with_retries(app, delays=(0, 0, 0, 0))
        )

        assert result is True
        assert bootstrap.call_count == 1

    def test_succeeds_on_second_try(self, monkeypatch):
        monkeypatch.setenv("FIRM_AUTO_START_LIVE", "true")
        app = _fake_app()
        calls = {"n": 0}

        def _bootstrap(a):
            calls["n"] += 1
            # First attempt: connect fails, engine object never gets set.
            # Second attempt: succeeds.
            if calls["n"] >= 2:
                a.state.live_engine = _fake_engine(is_running=True)

        bootstrap = MagicMock(side_effect=_bootstrap)
        monkeypatch.setattr(live_router, "bootstrap_live_from_yaml", bootstrap)

        result = asyncio.run(
            live_router.auto_start_live_with_retries(app, delays=(0, 0, 0, 0))
        )

        assert result is True
        assert bootstrap.call_count == 2

    def test_all_retries_exhausted_fires_critical_alert(self, monkeypatch):
        monkeypatch.setenv("FIRM_AUTO_START_LIVE", "1")
        app = _fake_app()  # live_engine stays None every attempt

        bootstrap = MagicMock()  # never sets app.state.live_engine
        monkeypatch.setattr(live_router, "bootstrap_live_from_yaml", bootstrap)

        alert_callback = MagicMock()
        monkeypatch.setattr(
            "firm.live.notifications.build_alert_callback",
            lambda: alert_callback,
        )

        result = asyncio.run(
            live_router.auto_start_live_with_retries(app, delays=(0, 0, 0))
        )

        assert result is False
        assert bootstrap.call_count == 3
        alert_callback.assert_called_once()
        fired = alert_callback.call_args[0][0]
        assert fired["kind"] == "auto_start_failed"
        assert fired["severity"] == "critical"

    def test_no_alert_when_webhook_unconfigured(self, monkeypatch):
        """build_alert_callback() returning None (unset ALERT_WEBHOOK_URL,
        the default) must not raise — same fail-silent contract as every
        other optional integration in this codebase."""
        monkeypatch.setenv("FIRM_AUTO_START_LIVE", "1")
        app = _fake_app()

        monkeypatch.setattr(live_router, "bootstrap_live_from_yaml", MagicMock())
        monkeypatch.setattr(
            "firm.live.notifications.build_alert_callback", lambda: None
        )

        result = asyncio.run(
            live_router.auto_start_live_with_retries(app, delays=(0,))
        )

        assert result is False

    def test_engine_present_but_not_running_keeps_retrying(self, monkeypatch):
        """An engine object that exists but never reports is_running=True
        (e.g. connect() failed inside it) must not be mistaken for success."""
        monkeypatch.setenv("FIRM_AUTO_START_LIVE", "1")
        app = _fake_app(engine=_fake_engine(is_running=False))

        bootstrap = MagicMock()  # leaves the not-running engine in place
        monkeypatch.setattr(live_router, "bootstrap_live_from_yaml", bootstrap)
        monkeypatch.setattr(
            "firm.live.notifications.build_alert_callback", lambda: None
        )

        result = asyncio.run(
            live_router.auto_start_live_with_retries(app, delays=(0, 0))
        )

        assert result is False
        assert bootstrap.call_count == 2

    @pytest.mark.parametrize("flag_value", ["0", "false", "no", ""])
    def test_various_falsy_flag_values_are_noop(self, monkeypatch, flag_value):
        monkeypatch.setenv("FIRM_AUTO_START_LIVE", flag_value)
        bootstrap = MagicMock()
        monkeypatch.setattr(live_router, "bootstrap_live_from_yaml", bootstrap)

        result = asyncio.run(
            live_router.auto_start_live_with_retries(_fake_app(), delays=(0,))
        )

        assert result is False
        bootstrap.assert_not_called()
