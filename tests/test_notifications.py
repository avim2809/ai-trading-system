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
        # A human-readable title, not the raw snake_case kind — that's the
        # whole point of this formatting: "drawdown_breach" reads like a log
        # line, "Kill Switch Tripped — Drawdown Breach" is what an operator
        # skimming Discord on their phone can act on instantly.
        assert "Kill Switch Tripped" in kwargs["json"]["text"]
        assert "CRITICAL" in kwargs["json"]["text"]
        # Discord's webhook API specifically requires "content" (not "text")
        # or it rejects the payload with a 400 — both fields carry the same
        # message so one POST works for Slack/Teams and Discord alike.
        assert kwargs["json"]["content"] == kwargs["json"]["text"]
        # The raw machine-readable kind is still present (in the raw `alert`
        # passthrough), so nothing that parses it is broken by the friendlier
        # display title.
        assert kwargs["json"]["alert"]["kind"] == "drawdown_breach"

    def test_discord_content_is_truncated_to_2000_chars(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")
        cb = build_alert_callback()
        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp) as mock_post:
            cb({
                "kind": "drawdown_breach",
                "severity": "critical",
                "message": "x" * 3000,
                "cycle_id": 5,
            })
        content = mock_post.call_args.kwargs["json"]["content"]
        assert len(content) <= 2000

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


class TestDiscordEmbed:
    """The Discord-specific rich card: severity-coded color, a human title
    instead of the raw snake_case kind, and the alert's extra context
    (drawdown, symbol, consecutive_failures, ...) rendered as readable
    fields rather than a dict dump. This is the actual "does this look
    urgent and can I act on it from my phone" signal — text/content alone
    (still sent for Slack/Teams) doesn't carry color."""

    def _embed(self, monkeypatch, alert: dict) -> dict:
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")
        cb = build_alert_callback()
        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp) as mock_post:
            cb(alert)
        [embed] = mock_post.call_args.kwargs["json"]["embeds"]
        return embed

    def test_critical_is_red_warning_is_gold(self, monkeypatch):
        critical = self._embed(monkeypatch, {
            "kind": "drawdown_breach", "severity": "critical", "message": "8% dd",
        })
        warning = self._embed(monkeypatch, {
            "kind": "broker_reconnected", "severity": "warning", "message": "back up",
        })
        assert critical["color"] == 0xED4245
        assert warning["color"] == 0xFAA61A
        assert critical["color"] != warning["color"]

    def test_known_kind_gets_a_human_readable_title(self, monkeypatch):
        embed = self._embed(monkeypatch, {
            "kind": "cycle_all_orders_failed", "severity": "critical", "message": "0 submitted",
        })
        assert "All Orders Failed" in embed["title"]
        # The raw machine kind must not leak into the human title.
        assert "cycle_all_orders_failed" not in embed["title"]

    def test_unknown_kind_falls_back_to_title_case_not_a_crash(self, monkeypatch):
        embed = self._embed(monkeypatch, {
            "kind": "some_future_alert_kind", "severity": "warning", "message": "x",
        })
        assert embed["title"].endswith("Some Future Alert Kind")

    def test_message_becomes_description(self, monkeypatch):
        embed = self._embed(monkeypatch, {
            "kind": "drawdown_breach", "severity": "critical",
            "message": "Drawdown 8.3% breached kill switch 8.0%; halting new orders.",
        })
        assert embed["description"] == "Drawdown 8.3% breached kill switch 8.0%; halting new orders."

    def test_context_kwargs_become_readable_fields(self, monkeypatch):
        embed = self._embed(monkeypatch, {
            "kind": "drawdown_breach", "severity": "critical", "message": "x",
            "cycle_id": 12, "drawdown": 0.083, "nav": 918234.5, "peak_equity": 1001000.0,
        })
        by_name = {f["name"]: f["value"] for f in embed["fields"]}
        assert by_name["Cycle"] == "12"
        # 0.083 -> "8.3%", not the bare float repr — this is the "readable"
        # requirement in practice, not just field renaming.
        assert by_name["Drawdown"] == "8.3%"
        assert by_name["NAV"] == "918,234.50"

    def test_boolean_context_renders_as_yes_no_not_true_false(self, monkeypatch):
        embed = self._embed(monkeypatch, {
            "kind": "broker_unavailable", "severity": "warning", "message": "x",
            "reconnected": True, "consecutive_failures": 1,
        })
        by_name = {f["name"]: f["value"] for f in embed["fields"]}
        assert by_name["Reconnected"] == "Yes"

    def test_non_context_keys_are_not_duplicated_as_fields(self, monkeypatch):
        """timestamp/kind/severity/message already have a dedicated place in
        the embed (timestamp field, title, color, description) — they must
        not also show up as generic fields."""
        embed = self._embed(monkeypatch, {
            "kind": "drawdown_breach", "severity": "critical", "message": "x",
            "timestamp": "2026-08-16T00:00:00+00:00", "cycle_id": 1,
        })
        field_names = {f["name"] for f in embed["fields"]}
        assert field_names == {"Cycle"}
        assert embed["timestamp"] == "2026-08-16T00:00:00+00:00"

    def test_footer_identifies_the_system(self, monkeypatch):
        embed = self._embed(monkeypatch, {
            "kind": "drawdown_breach", "severity": "critical", "message": "x",
        })
        assert embed["footer"]["text"] == "AI Trading System"


class TestEngineIntegration:
    def test_emit_alert_invokes_callback(self, tmp_path):
        """Regression: LiveTradingEngine._emit_alert must forward to whatever
        callback was wired at construction time (e.g. build_alert_callback())."""
        from firm.live.engine import LiveTradingEngine
        from tests.test_brokers import MockBroker

        received = []
        engine = LiveTradingEngine(
            config={
                "initial_capital": 100_000,
                "symbols": ["AAPL"],
                "memory_log_path": str(tmp_path / "decisions.jsonl"),
            },
            broker=MockBroker(),
            data_feed=MagicMock(),
            approval_queue=MagicMock(),
            alert_callback=received.append,
        )
        engine._emit_alert("test_alert", "critical", "something happened")
        assert len(received) == 1
        assert received[0]["kind"] == "test_alert"
        assert received[0]["severity"] == "critical"
