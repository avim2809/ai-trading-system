"""Real-time alert delivery for the live trading engine.

``LiveTradingEngine._emit_alert`` already logs every operational alert and
appends it to an in-memory list surfaced via ``GET /api/live/alerts``, but
until now nothing pushed those alerts to a human outside of someone tailing
logs or polling the dashboard — a kill-switch trip or an
``FIRM_ALLOW_TRADING`` block could sit unnoticed indefinitely.

This module builds an ``alert_callback`` (the hook ``LiveTradingEngine``
already accepts) that posts alerts to a generic webhook — Slack incoming
webhooks and Microsoft Teams/Discord-style webhooks all accept a JSON body
with a ``text`` field, so one HTTP POST covers the common cases without a
hard dependency on any single vendor SDK.

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


def _post_webhook(url: str, alert: dict[str, Any], timeout: float) -> None:
    import requests

    text = (
        f"[{alert.get('severity', '?').upper()}] {alert.get('kind', 'alert')}: "
        f"{alert.get('message', '')} (cycle={alert.get('cycle_id')})"
    )
    payload = {"text": text, "alert": alert}
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
