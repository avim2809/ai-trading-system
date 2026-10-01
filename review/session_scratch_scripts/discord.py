"""Post a session progress update to the configured Discord webhook (ALERT_WEBHOOK_URL in .env)."""
import os, sys
from datetime import datetime, timezone
from dotenv import dotenv_values
from firm.live.notifications import _post_webhook
url = os.environ.get("ALERT_WEBHOOK_URL") or dotenv_values(".env").get("ALERT_WEBHOOK_URL")
severity, message = sys.argv[1], sys.argv[2]
extra = dict(a.split("=", 1) for a in sys.argv[3:])
_post_webhook(url, {"kind": "research_session_update", "severity": severity, "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat(), **extra}, timeout=10)
print("sent")
