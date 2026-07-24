"""Trading scheduler – runs the live engine on configurable schedules.

Uses APScheduler's BackgroundScheduler for market-hours-aware scheduling
with support for cron expressions, interval triggers, and manual triggers.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    _HAS_APSCHEDULER = True
except ImportError:
    _HAS_APSCHEDULER = False

if TYPE_CHECKING:
    from firm.live.engine import LiveTradingEngine

_PRESET_SCHEDULES: dict[str, dict[str, Any]] = {
    # US equity RTH — times are US/Eastern (see TradingScheduler timezone).
    "market_open": {"hour": 9, "minute": 30, "day_of_week": "mon-fri"},
    "market_close": {"hour": 16, "minute": 0, "day_of_week": "mon-fri"},
    "hourly": {"minute": 5, "day_of_week": "mon-fri"},
}

_SESSION_SCHEDULES = frozenset({"market_open", "market_close"})

DEFAULT_MARKET_TIMEZONE = "US/Eastern"


def _pending_approvals_on_disk(path: str | Path) -> bool:
    """True when the persisted approval queue has pending entries."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return any(isinstance(row, dict) and row.get("status") == "pending" for row in data)
    except Exception:
        log.warning("Could not read approval queue at %s", p, exc_info=True)
        return False


def maybe_catch_up_session_cycle(
    engine: LiveTradingEngine,
    schedule_spec: str,
    *,
    timezone: str = DEFAULT_MARKET_TIMEZONE,
    approvals_path: str | Path = "data/approvals.json",
) -> None:
    """Run one cycle now if the service started mid-session without today's run.

    Only applies to session-anchored presets (``market_open`` / ``market_close``).
    Skipped when pending manual approvals exist — running a catch-up cycle
    concurrently would compete for the IBKR connection and re-queue orders.
    """
    if schedule_spec not in _SESSION_SCHEDULES:
        return
    if _pending_approvals_on_disk(approvals_path):
        log.info(
            "Catch-up cycle skipped — %s has pending approvals awaiting operator",
            approvals_path,
        )
        return
    if not engine.is_running:
        return
    if engine.had_cycle_today(timezone=timezone):
        return
    tz = ZoneInfo(timezone)
    now_local = datetime.now(tz)
    if now_local.weekday() >= 5:
        return
    try:
        if not engine._broker.is_market_open():
            return
    except Exception:
        log.warning(
            "Catch-up cycle skipped — could not determine market hours",
            exc_info=True,
        )
        return
    log.info(
        "Catch-up cycle: %s session open in %s but no cycle recorded today — starting now",
        schedule_spec, timezone,
    )

    def _run() -> None:
        try:
            engine.run_cycle()
        except Exception:
            log.error("Catch-up cycle failed", exc_info=True)

    threading.Thread(target=_run, name="live-cycle-catch-up", daemon=True).start()


class TradingScheduler:
    """Runs :meth:`LiveTradingEngine.run_cycle` on a configurable schedule."""

    def __init__(
        self,
        engine: LiveTradingEngine,
        schedule: str = "market_open",
        timezone: str = DEFAULT_MARKET_TIMEZONE,
        *,
        universe: list[str] | None = None,
        fundamentals_refresh_hour: int = 8,
    ) -> None:
        if not _HAS_APSCHEDULER:
            raise ImportError(
                "apscheduler is not installed. Install the live extra: "
                "pip install 'firm[live]' or pip install 'apscheduler>=3.10'"
            )

        self._engine = engine
        self._schedule_spec = schedule
        self._timezone = timezone
        self._universe = list(universe or [])
        self._fundamentals_refresh_hour = int(fundamentals_refresh_hour)
        self._scheduler: BackgroundScheduler | None = None
        self._job_id = "live_cycle"
        self._fundamentals_job_id = "fundamentals_refresh"

    def start(self) -> None:
        """Start the background scheduler."""
        self._scheduler = BackgroundScheduler(timezone=self._timezone)
        trigger = self._build_trigger(self._schedule_spec)
        self._scheduler.add_job(
            self._run_cycle_safe,
            trigger=trigger,
            id=self._job_id,
            replace_existing=True,
            # Never let a scheduled cycle overlap itself or pile up missed
            # runs into a burst; the engine also guards against overlap with
            # manual/API triggers via its own re-entrancy lock.
            max_instances=1,
            coalesce=True,
        )
        if self._universe:
            from firm.live.fundamentals_refresh import run_scheduled_fundamentals_refresh

            self._scheduler.add_job(
                lambda: run_scheduled_fundamentals_refresh(self._universe),
                trigger=CronTrigger(
                    hour=self._fundamentals_refresh_hour,
                    minute=0,
                    day_of_week="mon-fri",
                    timezone=self._timezone,
                ),
                id=self._fundamentals_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
        if self._universe:
            log.info(
                "Scheduler started: schedule=%s, tz=%s, fundamentals_refresh=%02d:00",
                self._schedule_spec, self._timezone, self._fundamentals_refresh_hour,
            )
        else:
            log.info("Scheduler started: schedule=%s, tz=%s", self._schedule_spec, self._timezone)
        self._scheduler.start()

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._scheduler is not None:
            if self._scheduler.running:
                self._scheduler.shutdown(wait=False)
            self._scheduler = None
            log.info("Scheduler stopped")

    def run_now(self) -> None:
        """Manually trigger an immediate cycle."""
        self._run_cycle_safe()

    def next_run(self) -> datetime | None:
        """Return the next scheduled execution time, or None."""
        if self._scheduler is None:
            return None
        job = self._scheduler.get_job(self._job_id)
        if job is None:
            return None
        return getattr(job, "next_run_time", None)

    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def _run_cycle_safe(self) -> None:
        """Wrapper that catches exceptions to avoid killing the scheduler."""
        try:
            self._engine.run_cycle()
        except Exception:
            log.error("Scheduled cycle failed", exc_info=True)

    def _build_trigger(self, spec: str) -> Any:
        if spec in _PRESET_SCHEDULES:
            kwargs = _PRESET_SCHEDULES[spec]
            return CronTrigger(**kwargs, timezone=self._timezone)

        if spec.startswith("cron:"):
            # "cron:HH:MM" shorthand
            parts = spec[5:].split(":")
            hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            return CronTrigger(
                hour=hour, minute=minute, day_of_week="mon-fri", timezone=self._timezone
            )

        if spec.startswith("every_") and spec.endswith("_minutes"):
            minutes = int(spec.replace("every_", "").replace("_minutes", ""))
            return IntervalTrigger(minutes=minutes)

        # Fall through: try to parse as a raw cron expression
        return CronTrigger.from_crontab(spec, timezone=self._timezone)
