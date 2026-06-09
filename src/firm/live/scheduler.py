"""Trading scheduler – runs the live engine on configurable schedules.

Uses APScheduler's BackgroundScheduler for market-hours-aware scheduling
with support for cron expressions, interval triggers, and manual triggers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

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
    "market_open": {"hour": 9, "minute": 35, "day_of_week": "mon-fri"},
    "market_close": {"hour": 15, "minute": 50, "day_of_week": "mon-fri"},
    "hourly": {"minute": 5, "day_of_week": "mon-fri"},
}


class TradingScheduler:
    """Runs :meth:`LiveTradingEngine.run_cycle` on a configurable schedule."""

    def __init__(
        self,
        engine: LiveTradingEngine,
        schedule: str = "market_open",
        timezone: str = "US/Eastern",
    ) -> None:
        if not _HAS_APSCHEDULER:
            raise ImportError(
                "apscheduler is not installed. Install the live extra: "
                "pip install 'firm[live]' or pip install 'apscheduler>=3.10'"
            )

        self._engine = engine
        self._schedule_spec = schedule
        self._timezone = timezone
        self._scheduler: BackgroundScheduler | None = None
        self._job_id = "live_cycle"

    def start(self) -> None:
        """Start the background scheduler."""
        self._scheduler = BackgroundScheduler(timezone=self._timezone)
        trigger = self._build_trigger(self._schedule_spec)
        self._scheduler.add_job(
            self._run_cycle_safe,
            trigger=trigger,
            id=self._job_id,
            replace_existing=True,
        )
        self._scheduler.start()
        log.info("Scheduler started: schedule=%s, tz=%s", self._schedule_spec, self._timezone)

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._scheduler is not None:
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
        return job.next_run_time

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
