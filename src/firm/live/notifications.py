"""Real-time alert delivery for the live trading engine.

``LiveTradingEngine._emit_alert`` already logs every operational alert and
appends it to an in-memory list surfaced via ``GET /api/live/alerts``, but
until now nothing pushed those alerts to a human outside of someone tailing
logs or polling the dashboard — a kill-switch trip or an
``FIRM_ALLOW_TRADING`` block could sit unnoticed indefinitely.

This module builds an ``alert_callback`` (the hook ``LiveTradingEngine``
already accepts) that posts alerts to a generic webhook. Slack and Microsoft
Teams incoming webhooks read a top-level ``text`` field; Discord's webhook
API is stricter — it *requires* at least one of ``content``/``embeds``/
``components``/``poll``/a file, and returns a 400 if none are present, so a
``text``-only body is silently rejected there. Sending ``text``/``content``
(a short plain-text line — what Slack/Teams render, and what Discord shows
as the notification-preview text) alongside a rich, severity-colored
``embeds`` card (what Discord actually renders in the channel) covers all
three with one POST and no hard dependency on any vendor SDK. Unknown extra
top-level keys (``embeds``, the raw ``alert`` dict) are simply ignored by
webhook consumers that don't understand them.

Configuration (all optional; the callback is a no-op — returns ``None``
from :func:`build_alert_callback` — when unset, so silence is the default
and this never blocks a fresh checkout from starting the engine):

    ALERT_WEBHOOK_URL      Slack/Teams/Discord/generic webhook URL.
    ALERT_MIN_SEVERITY     "warning" (default) or "critical" — alerts below
                           this level are not sent (still logged/queued).
    ALERT_WEBHOOK_TIMEOUT  HTTP timeout in seconds (default 5).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

log = logging.getLogger(__name__)

_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}

_DISCORD_CONTENT_LIMIT = 2000  # Discord rejects content over this length.
_DISCORD_TITLE_LIMIT = 256
_DISCORD_DESCRIPTION_LIMIT = 4096
_DISCORD_MAX_FIELDS = 25

# Discord embed side-bar colors, one per severity — the primary "is this
# urgent" signal (a plain [WARNING]/[CRITICAL] text prefix is easy to miss
# scrolling a busy channel; a red vs. gold vs. blue bar isn't).
_SEVERITY_COLOR = {
    "critical": 0xED4245,  # Discord's own "danger" red
    "warning": 0xFAA61A,   # Discord's own "caution" gold
    "info": 0x5865F2,      # Discord blurple
}
_SEVERITY_EMOJI = {"critical": "🔴", "warning": "🟠", "info": "🔵"}

# Human-friendly titles for every alert `kind` LiveTradingEngine actually
# emits (src/firm/live/engine.py) — a raw snake_case kind like
# "cycle_all_orders_failed" reads as a log line, not something an operator
# skimming Discord on their phone can act on instantly. Anything not listed
# here (a future alert kind) falls back to a generic title-cased rendering
# in _alert_title, so this never needs to be kept in perfect lockstep with
# engine.py to avoid looking broken.
_ALERT_TITLES = {
    "daily_limit_breach": "Daily Trade/Turnover Limit Breached",
    "drawdown_breach": "Kill Switch Tripped — Drawdown Breach",
    "kill_switch_reset": "Kill Switch Reset",
    "news_guard_blackout": "News-Guard Blackout — Orders Held",
    "news_guard_calendar_unavailable": "News-Guard Calendar Unavailable",
    "news_guard_stale_calendar": "News-Guard Using Stale Calendar",
    "cycle_hard_timeout": "Cycle Hard Timeout",
    "cycle_watchdog_timeout": "Cycle Watchdog Timeout",
    "cycle_all_orders_failed": "All Orders Failed This Cycle",
    "broker_unavailable": "Broker Unavailable",
    "broker_reconnected": "Broker Reconnected",
    "broker_disconnected_sustained": "Broker Disconnected (Sustained)",
    "broker_submission_circuit_open": "Order Submission Circuit Open",
    "reconciliation_degraded": "Reconciliation Degraded",
    "order_risk_cap_blocked": "Order Blocked — Risk Cap",
    "live_trading_locked": "Live Trading Locked",
}

# Friendly labels for context kwargs engine.py attaches to specific alerts
# (see the `**context` calls in LiveTradingEngine._emit_alert). Anything not
# listed here falls back to a title-cased rendering of the key itself.
_CONTEXT_LABELS = {
    "cycle_id": "Cycle",
    "consecutive_failures": "Consecutive Failures",
    "reconnected": "Reconnected",
    "drawdown": "Drawdown",
    "nav": "NAV",
    "peak_equity": "Peak Equity",
    "was_halted": "Was Halted",
    "new_peak_equity": "New Peak Equity",
    "symbols": "Symbols",
    "blocking_event": "Blocking Event",
    "symbol": "Symbol",
    "audit_id": "Audit ID",
}

# Alert-dict keys that already have a dedicated place in the embed (title,
# description, color) or aren't meaningful to a human as a bare field —
# everything else in the dict becomes a context field automatically, so a
# new `**context` kwarg added to some future _emit_alert call is surfaced
# without this module needing a matching update.
_NON_CONTEXT_KEYS = {"timestamp", "kind", "severity", "message"}


def _alert_title(kind: str) -> str:
    return _ALERT_TITLES.get(kind, kind.replace("_", " ").title())


def _format_field_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.1%}" if key == "drawdown" else f"{value:,.2f}"
    if isinstance(value, (list, tuple, set)):
        rendered = ", ".join(str(v) for v in value)
        return rendered or "—"
    if value is None:
        return "—"
    return str(value)


def _build_fields(alert: dict[str, Any]) -> list[dict[str, Any]]:
    fields = []
    for key, value in alert.items():
        if key in _NON_CONTEXT_KEYS:
            continue
        label = _CONTEXT_LABELS.get(key, key.replace("_", " ").title())
        fields.append({
            "name": label[:_DISCORD_TITLE_LIMIT],
            "value": _format_field_value(key, value)[:1024],
            "inline": True,
        })
    return fields[:_DISCORD_MAX_FIELDS]


def _build_embed(alert: dict[str, Any]) -> dict[str, Any]:
    severity = str(alert.get("severity", "warning")).lower()
    color = _SEVERITY_COLOR.get(severity, _SEVERITY_COLOR["warning"])
    title = f"{_SEVERITY_EMOJI.get(severity, '⚪')} {_alert_title(str(alert.get('kind', 'alert')))}"
    embed: dict[str, Any] = {
        "title": title[:_DISCORD_TITLE_LIMIT],
        "description": str(alert.get("message", ""))[:_DISCORD_DESCRIPTION_LIMIT],
        "color": color,
        "fields": _build_fields(alert),
        "footer": {"text": "AI Trading System"},
    }
    timestamp = alert.get("timestamp")
    if timestamp:
        # Already an ISO-8601 string (utcnow().isoformat() in engine.py) —
        # exactly the format Discord's embed timestamp field expects.
        embed["timestamp"] = timestamp
    return embed


def _post_webhook(url: str, alert: dict[str, Any], timeout: float) -> None:
    import requests

    severity = str(alert.get("severity", "warning")).lower()
    emoji = _SEVERITY_EMOJI.get(severity, "⚪")
    title = _alert_title(str(alert.get("kind", "alert")))
    # Short plain-text line: what Slack/Teams actually render, and what
    # Discord shows as the notification-preview/toast text even though the
    # richer, severity-colored card below (built by _build_embed) is what
    # actually renders in the channel.
    text = f"{emoji} [{severity.upper()}] {title}: {alert.get('message', '')}"
    text = text[: _DISCORD_CONTENT_LIMIT - 1] if len(text) > _DISCORD_CONTENT_LIMIT else text
    payload = {
        "text": text,
        "content": text,
        "embeds": [_build_embed(alert)],
        "alert": alert,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()


def build_alert_callback() -> Callable[[dict[str, Any]], None] | None:
    """Build a webhook-based alert callback from environment configuration.

    Returns ``None`` (no callback wired) when ``ALERT_WEBHOOK_URL`` is not
    set, matching the fail-silent default of every other optional
    integration in this codebase (news-guard offline mode, LLM providers,
    etc.) — an unconfigured deployment must keep working exactly as before.
    """
    url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        log.info(
            "ALERT_WEBHOOK_URL not set — live alerts will only be logged and "
            "surfaced via GET /api/live/alerts, not pushed to a webhook",
        )
        return None

    min_severity = os.getenv("ALERT_MIN_SEVERITY", "warning").strip().lower()
    min_rank = _SEVERITY_RANK.get(min_severity, 1)
    timeout = float(os.getenv("ALERT_WEBHOOK_TIMEOUT", "5"))

    def _callback(alert: dict[str, Any]) -> None:
        rank = _SEVERITY_RANK.get(str(alert.get("severity", "warning")).lower(), 1)
        if rank < min_rank:
            return
        try:
            _post_webhook(url, alert, timeout)
            log.debug("Alert delivered to webhook: %s", alert.get("kind"))
        except Exception:
            # Notification delivery must never break the trading loop —
            # _emit_alert already wraps this call in its own try/except, but
            # logging here too keeps the failure visible with full context.
            log.warning(
                "Failed to deliver alert %s to webhook", alert.get("kind"),
                exc_info=True,
            )

    log.info(
        "Live alert webhook configured (min_severity=%s) — kill-switch trips "
        "and execution-safety blocks will be pushed in real time",
        min_severity,
    )
    return _callback
