"""Optional, independently-scheduled daily chart-pattern scan (Phase 3
follow-up, docs/pattern_recognition_plan.md §2a).

Deliberately separate from :class:`firm.live.scheduler.TradingScheduler` —
this module never imports or touches it, and never runs unless
``FIRM_ENABLE_PATTERN_SCAN`` is explicitly set (same on/off convention as
``FIRM_AUTO_START_LIVE``, see ``firm.api.routers.live.bootstrap_live_from_yaml``).
That makes this feature inert with respect to the live trading engines by
construction, not just by convention — its own ``BackgroundScheduler``
instance is created, started, and stopped independently of anything the live
engine owns.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pandas as pd

log = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    _HAS_APSCHEDULER = True
except ImportError:
    _HAS_APSCHEDULER = False

DEFAULT_TIMEZONE = "US/Eastern"
# Mirrors firm.patterns.ml.labeling.DEFAULT_TIMEOUT_BARS -- a pending row
# older than this many calendar days without a target/stop hit is called a
# timeout rather than left pending forever. Calendar days (not bars) since
# outcome-checking works off real fetched dates, not a fixed in-memory array.
_TIMEOUT_CALENDAR_DAYS = 30


def pattern_scan_enabled() -> bool:
    """Same on/off convention as ``FIRM_AUTO_START_LIVE`` — empty/unset,
    or anything other than 1/true/yes, means OFF.
    """
    flag = os.getenv("FIRM_ENABLE_PATTERN_SCAN", "").lower()
    return flag in ("1", "true", "yes")


class PatternScanJob:
    """Owns one independent ``BackgroundScheduler`` running a single daily
    job: scan ``symbols`` for confirmed chart patterns, persist results, and
    re-check any previously-pending historical matches' outcomes.
    """

    def __init__(
        self,
        *,
        symbols: list[str],
        history_store=None,
        data_source: str = "cache",
        timezone: str = DEFAULT_TIMEZONE,
    ) -> None:
        if not _HAS_APSCHEDULER:
            raise RuntimeError("apscheduler is required for PatternScanJob")
        self._symbols = list(symbols)
        self._data_source = data_source
        if history_store is None:
            from firm.live.pattern_scan_history import PatternScanHistoryStore

            data_dir = os.environ.get("FIRM_DATA_DIR", "data")
            history_store = PatternScanHistoryStore(db_path=f"{data_dir}/pattern_scan_history.db")
        self._history = history_store
        self._scheduler = BackgroundScheduler(timezone=timezone)

    def start(self) -> None:
        self._scheduler.add_job(
            self._run_safe,
            CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone=self._scheduler.timezone),
            id="daily_pattern_scan",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        log.info(
            "PatternScanJob started: daily 16:30 ET scan of %d symbols (data_source=%s)",
            len(self._symbols), self._data_source,
        )

    def stop(self) -> None:
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:
            log.warning("PatternScanJob shutdown failed", exc_info=True)

    def _run_safe(self) -> None:
        """Wrapper that catches exceptions to avoid killing the scheduler
        — mirrors ``firm.live.scheduler.TradingScheduler._run_cycle_safe``.
        """
        try:
            self.run_once()
        except Exception:
            log.error("Daily pattern scan job failed", exc_info=True)

    def run_once(self, *, asof: str | None = None) -> dict:
        """Run one scan-and-persist cycle plus a pending-outcomes sweep.
        Exposed as a plain method (not just the cron callback) so it can be
        triggered manually/from tests without going through APScheduler.
        """
        from firm.api.routers.patterns import run_scan

        asof = asof or datetime.now().date().isoformat()
        try:
            scan = run_scan(
                symbols=self._symbols,
                asof=asof,
                data_source=self._data_source,
                lookback_days=252,
                zigzag_pct=0.03,
                min_score=60.0,
                confirm_lookback_bars=3,
                stop_atr_floor=1.5,
                enabled_patterns=None,
            )
        except FileNotFoundError as exc:
            log.warning("Daily pattern scan: no %s price data available: %s", self._data_source, exc)
            return {"scanned": 0, "matches": 0, "outcomes_checked": 0}
        except ValueError:
            log.error("Daily pattern scan: invalid scan parameters", exc_info=True)
            return {"scanned": 0, "matches": 0, "outcomes_checked": 0}

        self._history.insert_matches(
            scan["results"], confirm_dates=scan["confirm_dates"], source="scheduled",
        )
        checked = self._check_pending_outcomes()
        log.info(
            "Daily pattern scan: %d/%d symbols scanned, %d matches persisted, "
            "%d pending outcome(s) checked",
            scan["scanned"], len(self._symbols), len(scan["results"]), checked,
        )
        return {"scanned": scan["scanned"], "matches": len(scan["results"]), "outcomes_checked": checked}

    def _check_pending_outcomes(self) -> int:
        """Re-check every still-pending historical match against fresh price
        data and record its outcome once resolved (target/stop hit, or a
        calendar-day timeout) -- reuses
        ``firm.patterns.ml.labeling.first_barrier_hit`` rather than a second
        hand-written barrier-check loop.
        """
        from firm.patterns.ml.labeling import first_barrier_hit

        pending = self._history.pending_outcomes()
        if not pending:
            return 0

        try:
            from firm.config import get_settings
            from firm.runtime import load_prices

            prices_df = load_prices(get_settings())
        except FileNotFoundError:
            log.debug("Pending-outcome check: no cached price data available yet")
            return 0

        checked = 0
        for row in pending:
            confirm_date = row.get("confirm_date")
            if not confirm_date:
                continue
            sym_prices = prices_df[prices_df["symbol"].astype(str).str.upper() == row["symbol"].upper()]
            sym_prices = sym_prices.sort_values("date")
            confirm_ts = pd.Timestamp(confirm_date)
            after = sym_prices[sym_prices["date"] > confirm_ts]
            if after.empty:
                continue

            high = after["high"].to_numpy(dtype=float)
            low = after["low"].to_numpy(dtype=float)
            result = first_barrier_hit(
                high, low, direction=row["direction"], stop=row["stop"], target=row["target"],
            )
            if result == 1:
                self._history.update_outcome(row["id"], "target_hit")
                checked += 1
            elif result == -1:
                self._history.update_outcome(row["id"], "stop_hit")
                checked += 1
            elif (datetime.now() - confirm_ts.to_pydatetime()) > timedelta(days=_TIMEOUT_CALENDAR_DAYS):
                self._history.update_outcome(row["id"], "timeout")
                checked += 1
        return checked
