"""Tests for firm.live.notifications — no real webhook is ever called."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from firm.live.notifications import build_alert_callback


class TestBuildAlertCallback:
    def test_returns_none_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        assert build_alert_callback() is None

    def test_returns_callable_when_configured(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/x")
        cb = build_alert_callback()
        assert callable(cb)

    def test_posts_webhook_for_warning_by_default(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/x")
        monkeypatch.delenv("ALERT_MIN_SEVERITY", raising=False)
        cb = build_alert_callback()
        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp) as mock_post:
            cb({"kind": "drawdown_breach", "severity": "critical", "message": "8% dd", "cycle_id": 5})
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://hooks.example.com/x"
        assert "drawdown_breach" in kwargs["json"]["text"]

    def test_filters_below_min_severity(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/x")
        monkeypatch.setenv("ALERT_MIN_SEVERITY", "critical")
        cb = build_alert_callback()
        with patch("requests.post") as mock_post:
            cb({"kind": "reconciliation_degraded", "severity": "warning", "message": "x"})
        mock_post.assert_not_called()

    def test_webhook_failure_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/x")
        cb = build_alert_callback()
        with patch("requests.post", side_effect=ConnectionError("boom")):
            cb({"kind": "drawdown_breach", "severity": "critical", "message": "x"})  # must not raise


class TestEngineIntegration:
    def test_emit_alert_invokes_callback(self):
        """Regression: LiveTradingEngine._emit_alert must forward to whatever
        callback was wired at construction time (e.g. build_alert_callback())."""
        from firm.live.engine import LiveTradingEngine
        from tests.test_brokers import MockBroker

        received = []
        engine = LiveTradingEngine(
            config={"initial_capital": 100_000, "symbols": ["AAPL"]},
            broker=MockBroker(),
            data_feed=MagicMock(),
            approval_queue=MagicMock(),
            alert_callback=received.append,
        )
        engine._emit_alert("test_alert", "critical", "something happened")
        assert len(received) == 1
        assert received[0]["kind"] == "test_alert"
        assert received[0]["severity"] == "critical"
