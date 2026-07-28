"""Tests for the server resources / service control router.

Mocks psutil and subprocess/threading so tests never touch the real host
CPU/RAM/disk state and never actually invoke systemctl.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from firm.api.app import create_app


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


class TestResources:
    def test_resources_shape(self, client, monkeypatch):
        import firm.api.routers.system as system_mod

        monkeypatch.setattr(system_mod.psutil, "cpu_percent", lambda interval=None: 12.5)
        monkeypatch.setattr(system_mod.psutil, "cpu_count", lambda: 4)
        monkeypatch.setattr(
            system_mod.psutil,
            "virtual_memory",
            lambda: SimpleNamespace(used=2_000_000_000, total=4_000_000_000, percent=50.0),
        )
        monkeypatch.setattr(
            system_mod.psutil,
            "disk_usage",
            lambda path: SimpleNamespace(used=10_000_000_000, total=100_000_000_000, percent=10.0),
        )

        r = client.get("/api/system/resources")
        assert r.status_code == 200
        data = r.json()

        assert data["cpu"] == {"percent": 12.5, "count": 4}
        assert data["memory"] == {
            "used": 2_000_000_000,
            "total": 4_000_000_000,
            "percent": 50.0,
        }
        assert data["disk"]["used"] == 10_000_000_000
        assert data["disk"]["total"] == 100_000_000_000
        assert data["disk"]["percent"] == 10.0
        assert data["disk"]["path"] == "/"

    def test_resources_does_not_block_on_cpu_sampling(self, client, monkeypatch):
        """cpu_percent must be called with interval=None (non-blocking),
        never a blocking positive interval, inside a request handler."""
        import firm.api.routers.system as system_mod

        calls = []

        def fake_cpu_percent(interval=None):
            calls.append(interval)
            return 1.0

        monkeypatch.setattr(system_mod.psutil, "cpu_percent", fake_cpu_percent)
        monkeypatch.setattr(system_mod.psutil, "cpu_count", lambda: 1)
        monkeypatch.setattr(
            system_mod.psutil,
            "virtual_memory",
            lambda: SimpleNamespace(used=1, total=2, percent=50.0),
        )
        monkeypatch.setattr(
            system_mod.psutil,
            "disk_usage",
            lambda path: SimpleNamespace(used=1, total=2, percent=50.0),
        )

        client.get("/api/system/resources")
        assert calls == [None]


class TestServiceControl:
    @pytest.fixture(autouse=True)
    def _capture_timers(self, monkeypatch):
        """Replace threading.Timer so the delayed action fires synchronously
        instead of on a real 1s background thread, and record what command
        it would have run."""
        import firm.api.routers.system as system_mod

        self.scheduled = []

        class ImmediateTimer:
            def __init__(self, interval, function, args=None, kwargs=None):
                self.interval = interval
                self.function = function

            def start(self):
                self.scheduled_interval = self.interval
                # Don't actually call self.function() here — tests assert
                # on the recorded subprocess.run call args instead, so we
                # invoke it explicitly per-test after asserting the HTTP
                # response returned first.

            @property
            def daemon(self):
                return True

            @daemon.setter
            def daemon(self, value):
                pass

        outer = self

        def fake_timer(interval, function, args=None, kwargs=None):
            t = ImmediateTimer(interval, function, args=args, kwargs=kwargs)
            outer.scheduled.append(t)
            return t

        monkeypatch.setattr(system_mod.threading, "Timer", fake_timer)

        self.run_calls = []
        monkeypatch.setattr(
            system_mod.subprocess,
            "run",
            lambda cmd, **kw: outer.run_calls.append(cmd),
        )
        monkeypatch.setattr("os.geteuid", lambda: 0, raising=False)

    def test_restart_returns_immediately_and_enqueues_delayed_call(self, client):
        r = client.post("/api/system/restart")
        assert r.status_code == 200
        assert r.json() == {"status": "restarting"}

        # Response came back without the subprocess ever running.
        assert self.run_calls == []
        assert len(self.scheduled) == 1
        assert self.scheduled[0].interval > 0

        # Now fire the deferred callback and confirm it calls the right command.
        self.scheduled[0].function()
        assert self.run_calls == [["systemctl", "restart", "ai-trading.service"]]

    def test_kill_returns_immediately_and_enqueues_delayed_call(self, client):
        r = client.post("/api/system/kill")
        assert r.status_code == 200
        assert r.json() == {"status": "killing"}

        assert self.run_calls == []
        assert len(self.scheduled) == 1

        self.scheduled[0].function()
        assert self.run_calls == [
            ["systemctl", "kill", "--signal=SIGKILL", "ai-trading.service"]
        ]

    def test_uses_sudo_when_not_root(self, client, monkeypatch):
        monkeypatch.setattr("os.geteuid", lambda: 1000, raising=False)

        r = client.post("/api/system/restart")
        assert r.status_code == 200

        self.scheduled[0].function()
        assert self.run_calls == [
            ["sudo", "-n", "systemctl", "restart", "ai-trading.service"]
        ]
