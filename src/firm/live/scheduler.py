"""Trading scheduler – runs the live engine on configurable schedules.

Uses APScheduler's BackgroundScheduler for market-hours-aware scheduling
with support for cron expressions, interval triggers, and manual triggers.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime
from datetime import time as dt_time
from datetime import timezone as dt_tz
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
    # ------------------------------------------------------------------
    # 2026-09-18: legs of the "hourly_market_hours" composite schedule
    # (see HOURLY_MARKET_HOURS / TradingScheduler._start_hourly_market_hours_jobs).
    # Not selectable directly via ``schedule:`` -- only reachable through
    # that composite. Unlike "hourly" above (unrestricted hours, relies on
    # the engine's own is_market_open() to no-op outside RTH), this leg is
    # restricted to hour="10-14" so it doesn't waste a cycle/provider-fetch
    # outside market hours at all. Fires on the half-hour, strictly between
    # the "market_open" leg (9:30) and the close-anchor leg below (15:50).
    "_hourly_market_hours_intraday": {"hour": "10-14", "minute": 30, "day_of_week": "mon-fri"},
    # A few minutes before the 16:00 close so the day's final signal / any
    # close-to-open transition trade is deliberately captured, rather than
    # relying on the 14:30 intraday leg (90 minutes earlier) as the day's
    # last word.
    "_hourly_market_hours_close": {"hour": 15, "minute": 50, "day_of_week": "mon-fri"},
}

# Composite schedule spec (set as ``schedule:`` in config/live*.yaml):
# registers three separate CronTrigger jobs (open/intraday/close legs,
# built from the presets above) instead of the usual single "live_cycle"
# job, each passing an explicit ``cycle_type`` down to
# LiveTradingEngine.run_cycle -> Orchestrator.step so LLM-enhanced agent
# reasoning (agent_modes) is restricted to the open/close legs -- running
# the full pipeline hourly during market hours would otherwise multiply
# LLM API cost ~7x/day. See TradingScheduler._start_hourly_market_hours_jobs
# and config/live.yaml's schedule comment (2026-09-18).
HOURLY_MARKET_HOURS = "hourly_market_hours"

_SESSION_SCHEDULES = frozenset({"market_open", "market_close"})

DEFAULT_MARKET_TIMEZONE = "US/Eastern"

# ---------------------------------------------------------------------------
# Extended-hours trading (opt-in, off by default — see the
# ``extended_hours_trading`` config block: {"enabled": false, "premarket":
# {"enabled": false, "start": "...", "end": "...", "schedule": "cron:..."},
# "afterhours": {...}}, added to the systemd auto-start allowlist in
# firm.live.provider_utils). Distinct from HOURLY_MARKET_HOURS's "open"/
# "intraday"/"close" legs, which only ever fire inside *regular* trading
# hours. Two cycle_type values, each scheduled by its own optional
# TradingScheduler job (see TradingScheduler._start_extended_hours_jobs) and
# validated against the configured window by LiveTradingEngine.run_cycle
# (see within_extended_hours_window below) before it's allowed to bypass the
# regular is_market_open() gate.
# ---------------------------------------------------------------------------
EXTENDED_HOURS_CYCLE_TYPES = frozenset({"premarket", "afterhours"})

# Standard US equity convention, used whenever a session's config omits its
# own start/end. Pre-market: 04:00-09:30 ET. After-hours: 16:00-20:00 ET.
_DEFAULT_EXTENDED_HOURS_WINDOWS: dict[str, tuple[str, str]] = {
    "premarket": ("04:00", "09:30"),
    "afterhours": ("16:00", "20:00"),
}


def _parse_hhmm(value: str) -> dt_time:
    hour_str, _, minute_str = str(value).partition(":")
    return dt_time(int(hour_str), int(minute_str or 0))


def extended_hours_session_config(
    extended_hours_cfg: dict[str, Any] | None, cycle_type: str,
) -> dict[str, Any] | None:
    """Return the *enabled* sub-window config block for *cycle_type*
    ("premarket"/"afterhours") out of an ``extended_hours_trading`` config
    dict, or ``None`` when the feature overall, or that specific session,
    isn't enabled.

    Both the scheduler (deciding when to fire an extended-hours cycle) and
    the engine (deciding whether a fired cycle is genuinely still inside its
    window before letting it proceed — see ``within_extended_hours_window``)
    read this same helper, so they can never disagree about what "enabled"
    means.
    """
    cfg = extended_hours_cfg or {}
    if not cfg.get("enabled", False):
        return None
    session_cfg = cfg.get(cycle_type) or {}
    if not session_cfg.get("enabled", False):
        return None
    return session_cfg


def within_extended_hours_window(
    now: datetime,
    cycle_type: str,
    extended_hours_cfg: dict[str, Any] | None,
    *,
    timezone: str = DEFAULT_MARKET_TIMEZONE,
) -> bool:
    """True when *now* genuinely falls inside the configured ``[start, end)``
    window for *cycle_type* ("premarket"/"afterhours"), Mon-Fri only in
    *timezone*.

    Always False when the feature (or that specific session) isn't enabled
    in *extended_hours_cfg* — callers that reached here because a cycle's
    ``cycle_type`` is one of ``EXTENDED_HOURS_CYCLE_TYPES`` must treat a
    False return as "skip this cycle", not as "fall back to the regular
    is_market_open() gate" (see ``LiveTradingEngine.run_cycle``): a
    "premarket"/"afterhours" cycle_type only exists because something
    scheduled it, so silently treating a since-disabled feature as "market
    closed, use the regular check" could let it slip through the regular
    gate at a time (e.g. 6am) the regular gate never anticipates being
    asked about.
    """
    session_cfg = extended_hours_session_config(extended_hours_cfg, cycle_type)
    if session_cfg is None:
        return False
    tz = ZoneInfo(timezone)
    ts = now if now.tzinfo is not None else now.replace(tzinfo=dt_tz.utc)
    local = ts.astimezone(tz)
    if local.weekday() >= 5:
        return False
    default_start, default_end = _DEFAULT_EXTENDED_HOURS_WINDOWS.get(
        cycle_type, ("00:00", "00:00")
    )
    start_t = _parse_hhmm(session_cfg.get("start", default_start))
    end_t = _parse_hhmm(session_cfg.get("end", default_end))
    return start_t <= local.time() < end_t


def trading_day_key(at: datetime, timezone: str = DEFAULT_MARKET_TIMEZONE) -> str:
    """Calendar date for *at* in the trading session timezone (``YYYY-MM-DD``).

    Cycle timestamps are stored as naive UTC; this converts them before
    bucketing daily trade/turnover limits and decision-memory dates.
    """
    from datetime import timezone as dt_tz

    tz = ZoneInfo(timezone)
    ts = at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt_tz.utc)
    return ts.astimezone(tz).strftime("%Y-%m-%d")


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
    warmup_gate: Any | None = None,
) -> None:
    """Run one cycle now if the service started mid-session without today's run.

    Only applies to session-anchored presets (``market_open`` / ``market_close``).
    Skipped when pending manual approvals exist — running a catch-up cycle
    concurrently would compete for the IBKR connection and re-queue orders.
    """
    if schedule_spec not in _SESSION_SCHEDULES:
        return
    if warmup_gate is not None and not getattr(warmup_gate, "is_ready", False):
        from firm.live.pipeline_warmup import warmup_wait_seconds

        log.info("Catch-up waiting for pipeline warmup to finish")
        if not warmup_gate.wait_ready(timeout=warmup_wait_seconds()):
            log.warning(
                "Catch-up cycle skipped — pipeline warmup did not finish in %.0fs",
                warmup_wait_seconds(),
            )
            return
    if _pending_approvals_on_disk(approvals_path):
        log.info(
            "Catch-up cycle skipped — %s has pending approvals awaiting operator",
            approvals_path,
        )
        return
    if not engine.is_running:
        return
    if getattr(engine, "_shutting_down", False):
        log.info("Catch-up cycle skipped — engine is shutting down")
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
            if getattr(engine, "_shutting_down", False):
                return
            engine.run_cycle()
        except Exception:
            log.error("Catch-up cycle failed", exc_info=True)

    threading.Thread(target=_run, name="live-cycle-catch-up", daemon=True).start()


def cycle_had_no_trading_outcome(summary: dict[str, Any]) -> bool:
    """True when a persisted cycle summary shows no trading outcome for a
    reason that plausibly clears within the same session — either every
    proposed order was held by the news guard, or the cycle errored out
    entirely (e.g. a broker disconnect or a hard timeout).

    ``skipped`` cycles are excluded regardless of ``error`` (a "skipped:
    market closed" cycle sets both): market-closed is already covered by
    this module's own market-hours gate before this function is ever
    consulted, and the other skip reasons ("cycle already in progress",
    "engine shutting down") are momentary collisions the existing catch-up
    path already handles, not a lost trading day.

    The news-guard signature specifically: ``_apply_news_guard`` returning
    an empty list makes ``run_cycle`` return before order routing, so
    ``orders_submitted``/``orders_queued``/``orders_failed`` all stay 0
    even though ``orders_generated`` is positive.
    """
    if summary.get("skipped"):
        return False
    if summary.get("error"):
        return True
    return (
        summary.get("orders_generated", 0) > 0
        and summary.get("orders_submitted", 0) == 0
        and summary.get("orders_queued", 0) == 0
        and summary.get("orders_failed", 0) == 0
    )


def maybe_retry_lost_cycle(
    engine: LiveTradingEngine,
    *,
    timezone: str = DEFAULT_MARKET_TIMEZONE,
    cycle_type: str | None = None,
) -> None:
    """Run one more cycle now if today's most recent cycle had no trading
    outcome — either every order was held by the news guard, or the cycle
    errored out (e.g. a broker disconnect).

    A single-cycle-per-day schedule (e.g. ``market_open``) means either
    failure mode would otherwise silently lose the rest of the trading day,
    even with hours of market time left. This is meant to be polled
    periodically (see ``TradingScheduler``); it naturally stops retrying
    once a cycle actually submits or queues orders, since that cycle's
    summary no longer matches :func:`cycle_had_no_trading_outcome`. If the
    underlying cause isn't transient (a persistent bug, an extended broker
    outage), this will keep retrying every interval rather than giving up —
    the same bet the news-guard case already makes, and each attempt is
    cheap and feeds the engine's own consecutive-failure alerting.

    ``cycle_type`` is forwarded to ``engine.run_cycle`` unchanged (see
    ``TradingScheduler``'s call site for why it's hardcoded to "intraday"
    for the "hourly_market_hours" schedule, never inferred here).
    """
    if not engine.is_running or getattr(engine, "_shutting_down", False):
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
            "Lost-cycle retry check skipped — could not determine market hours",
            exc_info=True,
        )
        return

    todays = engine.cycles_today(timezone=timezone)
    if not todays or not cycle_had_no_trading_outcome(todays[-1]):
        return

    last = todays[-1]
    reason = f"error: {last['error']}" if last.get("error") else "held entirely by the news guard"
    log.info(
        "Retry cycle: today's most recent cycle (id=%s) had no trading "
        "outcome (%s) — retrying now instead of waiting for tomorrow's "
        "scheduled open",
        last.get("cycle_id"), reason,
    )

    def _run() -> None:
        try:
            if getattr(engine, "_shutting_down", False):
                return
            engine.run_cycle(cycle_type=cycle_type)
        except Exception:
            log.error("Lost-cycle retry failed", exc_info=True)

    threading.Thread(target=_run, name="live-cycle-lost-retry", daemon=True).start()


def run_order_reconciliation(engine: LiveTradingEngine) -> None:
    """Poll the broker for every locally non-terminal order's true status.

    Runs independently of the trading-cycle schedule since fills/cancels
    happen asynchronously at the broker between cycles, not only at cycle
    time.
    """
    if not engine.is_running or getattr(engine, "_shutting_down", False):
        return
    try:
        engine.reconcile_order_history()
    except Exception:
        log.error("Order-history reconciliation job failed", exc_info=True)


def run_position_reconciliation(
    engine: LiveTradingEngine, state: dict[str, str] | None = None,
) -> None:
    """Diff the broker's real account against the engine's internally
    tracked net position/cash, independent of the trading-cycle schedule.

    Pure observability (see ``LiveTradingEngine.check_reconciliation``,
    which this calls and which never alerts itself): never corrects
    portfolio state or affects order routing, only raises an alert through
    the existing alert path. Runs on its own interval rather than only at
    cycle start, so a once-daily cycle schedule doesn't leave drift
    unnoticed for a full trading day.

    *state* is a small dict the caller owns across ticks, mapping this
    job's name to the last-alerted status, so a sustained mismatch pages
    once on the transition into it -- and once more on recovery -- instead
    of on every tick it remains, the same convention
    ``run_resource_health_check`` uses. A single ongoing broker-data outage
    (or any other sustained cause) would otherwise re-alert the full
    discrepancy list every tick for as long as it persists.
    """
    if state is None:
        state = {}
    if not engine.is_running or getattr(engine, "_shutting_down", False):
        return
    try:
        result = engine.check_reconciliation()
    except Exception:
        log.error("Position reconciliation job failed", exc_info=True)
        return

    status = result.get("status", "unknown")
    previous = state.get("position_reconciliation", "ok")
    if status == previous:
        return
    state["position_reconciliation"] = status
    if status == "ok":
        if previous == "mismatch":
            engine._emit_alert(
                "portfolio_reconciliation_recovered", "info",
                "Broker/internal reconciliation is back in sync.",
            )
        return
    if status != "mismatch":
        return
    discrepancies = result.get("discrepancies", [])
    detail = "; ".join(
        f"{d['type']}"
        f"{' ' + d['symbol'] if 'symbol' in d else ''}"
        f": internal={d['internal']} broker={d['broker']}"
        for d in discrepancies
    )
    engine._emit_alert(
        "portfolio_reconciliation_mismatch", "warning",
        f"Broker/internal reconciliation found {len(discrepancies)} "
        f"discrepancy(ies) beyond tolerance: {detail}",
        discrepancies=discrepancies,
    )


# ---------------------------------------------------------------------------
# Host resource monitoring (disk / memory / CPU).
#
# This process is the only thing watching its own host: there is no
# external uptime monitor today, and every other job in this module reasons
# about the *trading* pipeline, not the box it runs on. A resource crunch
# (disk full, memory pressure, sustained CPU overload) otherwise surfaces
# only once it breaks something downstream — a failed write, a wedged
# process, an OOM kill — with no advance warning.
#
# Stdlib-only (``shutil``/``os.getloadavg``/``/proc/meminfo``): psutil is
# present in this venv but only as a transitive dependency of an unrelated
# package, not a direct one of this project, so nothing operationally
# load-bearing should rely on it still being there tomorrow.
# ---------------------------------------------------------------------------

# Defaults sized for this deployment's actual host (2 cores, a single root
# filesystem shared by the OS, both services' data/ dirs, and the venv) —
# override any of these via ``HEALTH_CHECK_<NAME>`` env vars without a code
# change. Two tiers per resource: "warn" gives an operator time to react
# before things degrade; "crit" is close to the point actual failures start
# (disk: writes start failing; memory: the OOM killer starts picking
# processes; CPU: cycles/requests start timing out).
_HEALTH_CHECK_DEFAULTS: dict[str, float] = {
    # Free space in GB / used-% on the filesystem holding the repo + data
    # dirs. Warn at 5GB free (~25% of a 20GB root disk) leaves real runway;
    # crit at 2GB free sits below the level that actually caused failures
    # rather than exactly at it, so the alert lands before, not during.
    "disk_warn_free_gb": 5.0,
    "disk_crit_free_gb": 2.0,
    "disk_warn_used_pct": 85.0,
    "disk_crit_used_pct": 95.0,
    # % of total RAM still available (MemAvailable, not MemFree — accounts
    # for reclaimable cache instead of alerting on a healthy page cache).
    # Expressed as a percentage rather than a fixed GB figure so the same
    # default still means something if the box is ever resized.
    "mem_warn_avail_pct": 15.0,
    "mem_crit_avail_pct": 8.0,
    # 5-minute load average per core (smoother than the 1-minute figure —
    # a brief burst shouldn't page anyone). >1.0/core means work is queuing;
    # 1.5 gives margin for normal bursts, 3.0 is sustained thrashing on a
    # 2-core box.
    "cpu_warn_load_per_core": 1.5,
    "cpu_crit_load_per_core": 3.0,
}


def _health_threshold(name: str) -> float:
    env_key = f"HEALTH_CHECK_{name.upper()}"
    raw = os.getenv(env_key)
    if raw is None:
        return _HEALTH_CHECK_DEFAULTS[name]
    try:
        return float(raw)
    except ValueError:
        log.warning("Ignoring invalid %s=%r, using default", env_key, raw)
        return _HEALTH_CHECK_DEFAULTS[name]


def _check_disk(path: str) -> dict[str, Any] | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        log.warning("Disk health check: could not stat %s", path, exc_info=True)
        return None
    free_gb = usage.free / (1024 ** 3)
    used_pct = (usage.used / usage.total * 100) if usage.total else 0.0
    severity = "ok"
    if free_gb <= _health_threshold("disk_crit_free_gb") or used_pct >= _health_threshold(
        "disk_crit_used_pct"
    ):
        severity = "critical"
    elif free_gb <= _health_threshold("disk_warn_free_gb") or used_pct >= _health_threshold(
        "disk_warn_used_pct"
    ):
        severity = "warning"
    return {
        "severity": severity,
        "free_gb": round(free_gb, 2),
        "used_pct": round(used_pct, 1),
        "message": f"Disk free: {free_gb:.2f} GB ({used_pct:.0f}% used) on {path}",
    }


def _check_memory() -> dict[str, Any] | None:
    """Linux-only (``/proc/meminfo``) — this deploys exclusively to a
    bare-metal Linux host (see CLAUDE.md), never anywhere that lacks it.
    """
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    info[key] = int(parts[0])  # kB
    except OSError:
        log.warning("Memory health check: could not read /proc/meminfo", exc_info=True)
        return None
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", 0)
    if not total:
        return None
    available_pct = available / total * 100
    severity = "ok"
    if available_pct <= _health_threshold("mem_crit_avail_pct"):
        severity = "critical"
    elif available_pct <= _health_threshold("mem_warn_avail_pct"):
        severity = "warning"
    return {
        "severity": severity,
        "available_pct": round(available_pct, 1),
        "message": f"Memory available: {available_pct:.0f}% ({available / 1_048_576:.2f} GB)",
    }


def _check_cpu() -> dict[str, Any] | None:
    try:
        _, load5, _ = os.getloadavg()
    except OSError:
        log.warning("CPU health check: getloadavg() unavailable", exc_info=True)
        return None
    cpu_count = os.cpu_count() or 1
    load_per_core = load5 / cpu_count
    severity = "ok"
    if load_per_core >= _health_threshold("cpu_crit_load_per_core"):
        severity = "critical"
    elif load_per_core >= _health_threshold("cpu_warn_load_per_core"):
        severity = "warning"
    return {
        "severity": severity,
        "load_per_core": round(load_per_core, 2),
        "message": f"CPU load (5-min avg): {load_per_core:.2f} per core across {cpu_count} core(s)",
    }


def check_resource_health(path: str = ".") -> dict[str, dict[str, Any]]:
    """Classify this host's disk/memory/CPU against the thresholds above.

    Returns one entry per metric that could be measured (keyed
    ``"disk"``/``"memory"``/``"cpu"``), each with a ``severity`` of
    ``"ok"``/``"warning"``/``"critical"``. Each measurement is independent
    so one that can't be taken (e.g. an unreadable ``/proc/meminfo``) never
    blocks the other two.
    """
    metrics: dict[str, dict[str, Any]] = {}
    disk = _check_disk(path)
    if disk is not None:
        metrics["disk"] = disk
    memory = _check_memory()
    if memory is not None:
        metrics["memory"] = memory
    cpu = _check_cpu()
    if cpu is not None:
        metrics["cpu"] = cpu
    return metrics


_HEALTH_ALERT_KIND = {
    "disk": "host_disk_low",
    "memory": "host_memory_low",
    "cpu": "host_cpu_high",
}


def run_resource_health_check(
    engine: LiveTradingEngine, state: dict[str, str] | None = None,
) -> None:
    """Sample this host's disk/memory/CPU and push a real alert through the
    engine's existing alert pipeline (log + ``GET /api/live/alerts`` +
    webhook, see ``LiveTradingEngine._emit_alert`` /
    ``firm.live.notifications``) once a threshold is breached — the same
    channel a kill-switch trip or a broker outage already uses, rather than
    a second, separate notification path.

    Deliberately does not gate on ``engine.is_running``/``_shutting_down``
    the way the other jobs in this module do: the host can run low on disk
    or memory whether or not the trading engine happens to be active right
    now, and ``_emit_alert`` doesn't depend on engine state either.

    *state* is a small dict the caller owns across ticks (one per
    engine/scheduler), mapping metric name -> last-alerted severity, so a
    sustained breach pages once on the transition into it — and once more
    on recovery — instead of every tick it remains there.
    """
    if state is None:
        state = {}
    try:
        metrics = check_resource_health()
    except Exception:
        log.error("Resource health check failed", exc_info=True)
        return
    for name, metric in metrics.items():
        severity = metric["severity"]
        previous = state.get(name, "ok")
        if severity == previous:
            continue
        state[name] = severity
        context = {k: v for k, v in metric.items() if k not in ("severity", "message")}
        if severity == "ok":
            engine._emit_alert(
                f"{_HEALTH_ALERT_KIND[name]}_recovered", "info", metric["message"], **context,
            )
        else:
            engine._emit_alert(
                _HEALTH_ALERT_KIND[name], severity, metric["message"], **context,
            )


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
        dynamic_universe_enabled: bool = False,
        dynamic_universe_state_path: str = "data/dynamic_universe_state.json",
        dynamic_universe_max_symbols: int = 10,
        dynamic_universe_min_dwell_days: int = 5,
        dynamic_universe_sync_hour: int | None = None,
        # FMP-sourced alternative to the danelfin_* dynamic-universe fields
        # above (see firm.live.sp500_universe_sync / sp500_sector_cache) —
        # mutually-exclusive A/B toggle with the Danelfin source, per
        # config/live.yaml's sp500_dynamic_universe block.
        sp500_dynamic_universe_enabled: bool = False,
        sp500_dynamic_universe_state_path: str = "data/dynamic_universe_state.json",
        sp500_sector_cache_path: str = "data/sp500_sector_map.json",
        sp500_dynamic_universe_max_symbols: int = 10,
        sp500_dynamic_universe_min_dwell_days: int = 5,
        sp500_dynamic_universe_incubation_days: int = 5,
        sp500_liquidity_lookback_days: int = 30,
        sp500_sync_hour: int | None = None,
        sp500_sector_cache_refresh_day: str = "sun",
        # Snapshot of the *static* universe's symbol->sector map (e.g.
        # config/live.yaml's risk.sector_map), passed straight through to
        # sp500_universe_sync.sync_once's water-fill selection — not part
        # of danelfin_universe_sync's sync_once signature (it has no notion
        # of sector-balancing), so it has no equivalent kwarg above.
        sp500_static_sector_map: dict[str, str] | None = None,
        # Opt-in premarket/afterhours cycles (off by default — see the
        # module-level EXTENDED_HOURS_CYCLE_TYPES docstring and
        # ``extended_hours_trading`` in config/live*.yaml). Additive: these
        # legs are registered alongside whatever ``schedule`` above already
        # sets up, not a replacement for it.
        extended_hours_trading: dict[str, Any] | None = None,
        # Opt-in daily RAG "news" collection ingestion (off by default —
        # see firm.live.news_ingestion_job and the ``news_ingestion``
        # config block: {"enabled": false, "hour": 7, "days": 3}).
        news_ingestion: dict[str, Any] | None = None,
        # Opt-in adaptive per-sleeve capital reweighting check (off by
        # default — see firm.live.capital_reallocation_job and the
        # ``capital_reallocation`` config block: {"enabled": false,
        # "day_of_week": "sun", "hour": 6, ...}). Sleeved mode only; a
        # no-op in blended mode regardless of this flag (guarded inside
        # the job itself, not here, so this scheduler doesn't need to know
        # the engine's capital_allocation_mode).
        capital_reallocation: dict[str, Any] | None = None,
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
        self._dynamic_universe_enabled = bool(dynamic_universe_enabled)
        self._dynamic_universe_state_path = dynamic_universe_state_path
        self._dynamic_universe_max_symbols = int(dynamic_universe_max_symbols)
        self._dynamic_universe_min_dwell_days = int(dynamic_universe_min_dwell_days)
        # Runs once/day before fundamentals_refresh so the universe is
        # settled before that job's own fetch, unless explicitly overridden.
        self._dynamic_universe_sync_hour = (
            int(dynamic_universe_sync_hour)
            if dynamic_universe_sync_hour is not None
            else max(0, self._fundamentals_refresh_hour - 1)
        )
        self._sp500_dynamic_universe_enabled = bool(sp500_dynamic_universe_enabled)
        self._sp500_dynamic_universe_state_path = sp500_dynamic_universe_state_path
        self._sp500_sector_cache_path = sp500_sector_cache_path
        self._sp500_dynamic_universe_max_symbols = int(sp500_dynamic_universe_max_symbols)
        self._sp500_dynamic_universe_min_dwell_days = int(sp500_dynamic_universe_min_dwell_days)
        self._sp500_dynamic_universe_incubation_days = int(sp500_dynamic_universe_incubation_days)
        self._sp500_liquidity_lookback_days = int(sp500_liquidity_lookback_days)
        self._sp500_sync_hour = (
            int(sp500_sync_hour)
            if sp500_sync_hour is not None
            else max(0, self._fundamentals_refresh_hour - 1)
        )
        self._sp500_sector_cache_refresh_day = sp500_sector_cache_refresh_day
        self._sp500_static_sector_map = dict(sp500_static_sector_map or {})
        self._extended_hours_cfg: dict[str, Any] = dict(extended_hours_trading or {})
        self._news_ingestion_cfg: dict[str, Any] = dict(news_ingestion or {})
        self._capital_reallocation_cfg: dict[str, Any] = dict(capital_reallocation or {})
        self._scheduler: BackgroundScheduler | None = None
        self._job_id = "live_cycle"
        # Extra legs registered only for the "hourly_market_hours" composite
        # schedule (see _start_hourly_market_hours_jobs) -- the open leg
        # reuses self._job_id above so next_run()'s existing default keeps
        # working unchanged for every other schedule.
        self._intraday_job_id = "live_cycle_intraday"
        self._close_job_id = "live_cycle_close"
        self._fundamentals_job_id = "fundamentals_refresh"
        self._news_ingestion_job_id = "news_ingestion"
        self._capital_reallocation_job_id = "capital_reallocation_check"
        self._dynamic_universe_job_id = "danelfin_universe_sync"
        self._sp500_sync_job_id = "sp500_universe_sync"
        self._sp500_sector_refresh_job_id = "sp500_sector_cache_refresh"
        self._lost_cycle_retry_job_id = "lost_cycle_retry"
        self._order_reconciliation_job_id = "order_reconciliation"
        self._position_reconciliation_job_id = "position_reconciliation"
        self._resource_health_job_id = "resource_health_check"
        # Owned by this scheduler instance (one per engine/live instance) so
        # run_resource_health_check's/run_position_reconciliation's
        # alert-on-transition logic persists across ticks rather than
        # resetting every call.
        self._resource_health_state: dict[str, str] = {}
        self._reconciliation_state: dict[str, str] = {}
        # Extended-hours legs (see _start_extended_hours_jobs) — only
        # registered when self._extended_hours_cfg["enabled"] is true.
        self._premarket_job_id = "live_cycle_premarket"
        self._afterhours_job_id = "live_cycle_afterhours"

    def start(self) -> None:
        """Start the background scheduler."""
        self._scheduler = BackgroundScheduler(timezone=self._timezone)
        if self._schedule_spec == HOURLY_MARKET_HOURS:
            self._start_hourly_market_hours_jobs()
        else:
            trigger = self._build_trigger(self._schedule_spec)
            self._scheduler.add_job(
                self._run_cycle_safe,
                trigger=trigger,
                id=self._job_id,
                replace_existing=True,
                # Never let a scheduled cycle overlap itself or pile up missed
                # runs into a burst; the engine also guards against overlap
                # with manual/API triggers via its own re-entrancy lock.
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
        if self._universe and self._news_ingestion_cfg.get("enabled"):
            from firm.live.news_ingestion_job import run_scheduled_news_ingestion

            news_days = int(self._news_ingestion_cfg.get("days", 3))
            self._scheduler.add_job(
                lambda: run_scheduled_news_ingestion(self._universe, days=news_days),
                trigger=CronTrigger(
                    hour=int(self._news_ingestion_cfg.get("hour", 7)),
                    minute=0,
                    day_of_week="mon-fri",
                    timezone=self._timezone,
                ),
                id=self._news_ingestion_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
        if self._capital_reallocation_cfg.get("enabled"):
            from firm.live.capital_reallocation_job import (
                run_scheduled_capital_reallocation_check,
            )

            # Weekly by default -- rolling Sharpe/Sortino over a
            # 30-60 day window barely moves day to day, so a daily check
            # would mostly just log the same recommendation repeatedly.
            # Runs before market open so a human reviewing it has the full
            # trading day to decide whether to apply it.
            self._scheduler.add_job(
                lambda: run_scheduled_capital_reallocation_check(self._engine),
                trigger=CronTrigger(
                    day_of_week=self._capital_reallocation_cfg.get("day_of_week", "sun"),
                    hour=int(self._capital_reallocation_cfg.get("hour", 6)),
                    minute=0,
                    timezone=self._timezone,
                ),
                id=self._capital_reallocation_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
        if self._dynamic_universe_enabled and self._universe:
            from firm.live.danelfin_universe_sync import sync_once

            self._scheduler.add_job(
                lambda: sync_once(
                    self._engine,
                    state_path=self._dynamic_universe_state_path,
                    static_universe=self._universe,
                    max_dynamic_symbols=self._dynamic_universe_max_symbols,
                    min_dwell_days=self._dynamic_universe_min_dwell_days,
                ),
                trigger=CronTrigger(
                    hour=self._dynamic_universe_sync_hour,
                    minute=0,
                    day_of_week="mon-fri",
                    timezone=self._timezone,
                ),
                id=self._dynamic_universe_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
        if self._sp500_dynamic_universe_enabled and self._universe:
            from firm.live import sp500_universe_sync

            self._scheduler.add_job(
                lambda: sp500_universe_sync.sync_once(
                    self._engine,
                    state_path=self._sp500_dynamic_universe_state_path,
                    sector_cache_path=self._sp500_sector_cache_path,
                    static_universe=self._universe,
                    static_sector_map=self._sp500_static_sector_map,
                    max_dynamic_symbols=self._sp500_dynamic_universe_max_symbols,
                    min_dwell_days=self._sp500_dynamic_universe_min_dwell_days,
                    liquidity_lookback_days=self._sp500_liquidity_lookback_days,
                    incubation_days=self._sp500_dynamic_universe_incubation_days,
                ),
                trigger=CronTrigger(
                    hour=self._sp500_sync_hour,
                    minute=0,
                    day_of_week="mon-fri",
                    timezone=self._timezone,
                ),
                id=self._sp500_sync_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            self._scheduler.add_job(
                self._refresh_sp500_sector_cache_safe,
                trigger=CronTrigger(
                    hour=self._sp500_sync_hour,
                    minute=0,
                    day_of_week=self._sp500_sector_cache_refresh_day,
                    timezone=self._timezone,
                ),
                id=self._sp500_sector_refresh_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
        self._scheduler.add_job(
            lambda: run_order_reconciliation(self._engine),
            trigger=IntervalTrigger(minutes=15),
            id=self._order_reconciliation_job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        # Unconditional (no universe/config gate) and independent of the
        # trading schedule, same rationale as order reconciliation above.
        # 30 minutes rather than order reconciliation's 15: position/cash
        # drift is a slower-moving signal (it only changes on a fill), so
        # there's no benefit to polling it as tightly as order status.
        self._scheduler.add_job(
            lambda: run_position_reconciliation(self._engine, self._reconciliation_state),
            trigger=IntervalTrigger(minutes=30),
            id=self._position_reconciliation_job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        # Unconditional (no universe/config gate) and independent of the
        # trading schedule, same as order reconciliation above — the host
        # this runs on needs watching regardless of what's configured to
        # trade on it. 15 minutes matches reconciliation's cadence: cheap
        # stdlib syscalls, no reason to poll less often, and frequent enough
        # that a fast-moving leak (e.g. an unbounded log/cache) is caught
        # with hours of runway rather than found the next time someone looks.
        self._scheduler.add_job(
            lambda: run_resource_health_check(self._engine, self._resource_health_state),
            trigger=IntervalTrigger(minutes=15),
            id=self._resource_health_job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        if self._extended_hours_cfg.get("enabled"):
            self._start_extended_hours_jobs()
        if self._schedule_spec in _SESSION_SCHEDULES or self._schedule_spec == HOURLY_MARKET_HOURS:
            # Meaningful for a single-cycle-per-day schedule (market_open/
            # market_close) since nothing else would run again for the rest
            # of the day otherwise. Also kept for "hourly_market_hours"
            # (2026-09-18): unlike the plain `every_N_minutes`/`hourly`
            # preset (which retries within minutes on its own next tick),
            # this composite's legs are up to ~80 minutes apart (14:30
            # intraday -> 15:50 close-anchor) — still worth a 30-minute
            # safety net rather than silently losing that whole gap to a
            # transient broker/news-guard issue.
            # A retry of a lost cycle always gets cycle_type="intraday"
            # (quant-only) regardless of which leg actually failed: it's
            # the conservative, cost-safe default (never *adds* an LLM call
            # a schedule-driven "open"/"close" cycle wouldn't already have
            # made) and this job has no reliable way to know which leg's
            # cycle it's standing in for.
            retry_cycle_type = "intraday" if self._schedule_spec == HOURLY_MARKET_HOURS else None
            self._scheduler.add_job(
                lambda: maybe_retry_lost_cycle(
                    self._engine, timezone=self._timezone, cycle_type=retry_cycle_type,
                ),
                trigger=IntervalTrigger(minutes=30),
                id=self._lost_cycle_retry_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
        if self._universe:
            log.info(
                "Scheduler started: schedule=%s, tz=%s, fundamentals_refresh=%02d:00%s%s%s",
                self._schedule_spec, self._timezone, self._fundamentals_refresh_hour,
                f", danelfin_universe_sync={self._dynamic_universe_sync_hour:02d}:00"
                if self._dynamic_universe_enabled else "",
                f", sp500_universe_sync={self._sp500_sync_hour:02d}:00"
                f" (sector_cache_refresh={self._sp500_sector_cache_refresh_day})"
                if self._sp500_dynamic_universe_enabled else "",
                f", news_ingestion={int(self._news_ingestion_cfg.get('hour', 7)):02d}:00"
                if self._news_ingestion_cfg.get("enabled") else "",
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

    def next_run(self, job_id: str | None = None) -> datetime | None:
        """Return the next scheduled execution time for *job_id*, or None.

        Defaults to the main trading-cycle job. For a session-anchored
        schedule (``market_open``/``market_close``) this is the *only* thing
        that fires automatically once a day — but ``maybe_retry_lost_cycle``
        (see :meth:`next_lost_cycle_retry`) can run a cycle well before then,
        so callers that want "when will this engine next act" should check
        both rather than just this one.

        For the "hourly_market_hours" composite schedule, *job_id=None*
        instead returns the earliest next fire across all three legs (open/
        intraday/close) — the single most-relevant "next cycle" instant,
        since three separate jobs exist under the hood (see
        ``_start_hourly_market_hours_jobs``). Pass an explicit job id
        (``self._job_id`` / ``self._intraday_job_id`` / ``self._close_job_id``)
        to inspect one leg specifically.
        """
        if self._scheduler is None:
            return None
        if job_id is None and self._schedule_spec == HOURLY_MARKET_HOURS:
            candidates = [
                self.next_run(jid)
                for jid in (self._job_id, self._intraday_job_id, self._close_job_id)
            ]
            live = [c for c in candidates if c is not None]
            return min(live) if live else None
        job = self._scheduler.get_job(job_id or self._job_id)
        if job is None:
            return None
        return getattr(job, "next_run_time", None)

    def next_lost_cycle_retry(self) -> datetime | None:
        """Next fire time of the lost-cycle-retry safety net, or None.

        Registered for session-anchored schedules (``market_open``/
        ``market_close``) and for "hourly_market_hours" (see ``start()``'s
        session-schedule check) — None here means either the scheduler
        isn't running or the schedule is a plain interval/``hourly`` preset,
        which already retries on its own next regular tick without this
        job existing at all.
        """
        return self.next_run(self._lost_cycle_retry_job_id)

    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def _refresh_sp500_sector_cache_safe(self) -> None:
        """Weekly sp500 sector-cache refresh, wrapped so a transient/
        premium-gated FMP failure never kills the scheduler (mirrors
        ``_run_cycle_safe``). Alpha Vantage is used as a bounded per-symbol
        backfill only when ``ALPHAVANTAGE_API_KEY`` is configured — see
        ``firm.live.sp500_sector_cache.refresh_sector_cache``.
        """
        try:
            from firm.data.providers.fmp import FMPProvider
            from firm.live import sp500_sector_cache

            backfill_provider = None
            backfill_limit = 0
            if os.getenv("ALPHAVANTAGE_API_KEY"):
                from firm.data.providers.alphavantage import AlphaVantageProvider

                backfill_provider = AlphaVantageProvider()
                # Bounded, never the full ~500-name pool — see
                # sp500_sector_cache.refresh_sector_cache's docstring.
                backfill_limit = 25

            sp500_sector_cache.refresh_sector_cache(
                self._sp500_sector_cache_path,
                fmp_provider=FMPProvider(),
                seed_map=self._sp500_static_sector_map,
                backfill_provider=backfill_provider,
                backfill_limit=backfill_limit,
                today=datetime.now(ZoneInfo(self._timezone)).date().isoformat(),
            )
        except Exception:
            log.error("sp500 sector-cache refresh job failed", exc_info=True)

    def _start_hourly_market_hours_jobs(self) -> None:
        """Register the "hourly_market_hours" composite schedule's three
        legs as separate APScheduler jobs, each passing an explicit
        ``cycle_type`` through :meth:`_run_cycle_safe` down to
        ``LiveTradingEngine.run_cycle`` -> ``Orchestrator.step`` (added
        2026-09-18 — see module docstring on ``HOURLY_MARKET_HOURS``).

        A fixed ``cycle_type`` baked into each job's own callback (rather
        than one generic job inferring "is this the open/close cycle" from
        the wall clock at run time) keeps that decision explicit and
        traceable at the one place that actually knows which leg just
        fired, per this feature's design constraint — see
        config/live.yaml's schedule comment.

        The open leg is registered under ``self._job_id`` (the same id
        every other schedule's single job uses) so ``next_run()``'s
        existing default keeps resolving to *something* sensible even for
        callers that don't know about the intraday/close legs.
        """
        self._scheduler.add_job(
            lambda: self._run_cycle_safe(cycle_type="open"),
            trigger=self._build_trigger("market_open"),
            id=self._job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.add_job(
            lambda: self._run_cycle_safe(cycle_type="intraday"),
            trigger=self._build_trigger("_hourly_market_hours_intraday"),
            id=self._intraday_job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.add_job(
            lambda: self._run_cycle_safe(cycle_type="close"),
            trigger=self._build_trigger("_hourly_market_hours_close"),
            id=self._close_job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

    def _start_extended_hours_jobs(self) -> None:
        """Register the opt-in premarket/afterhours legs (see the
        ``extended_hours_trading`` config block and module-level
        ``EXTENDED_HOURS_CYCLE_TYPES`` docstring).

        Additive to whatever ``self._schedule_spec`` already registered
        above (a plain preset, an interval, or the "hourly_market_hours"
        composite) — this only runs at all when
        ``self._extended_hours_cfg["enabled"]`` is true, which defaults to
        false, so an operator who never sets it sees no new jobs and no
        behavior change. Each leg's own ``schedule`` key (default
        "cron:08:00" for premarket, "cron:17:00" for afterhours — a few
        minutes inside the session's own window, giving prices a moment to
        populate/settle at the open of that window) is resolved through the
        same ``_build_trigger`` used by every other schedule spec in this
        class, so the same "cron:HH:MM" shorthand / raw-crontab conventions
        apply here too. The *actual* window bounds a fired cycle is
        validated against live in ``extended_hours_cfg["premarket"/
        "afterhours"]["start"/"end"]`` and are checked independently by
        ``LiveTradingEngine.run_cycle`` via ``within_extended_hours_window``
        — this method only decides when APScheduler calls in, not whether
        the engine actually lets that cycle run.
        """
        premarket_cfg = self._extended_hours_cfg.get("premarket") or {}
        if premarket_cfg.get("enabled"):
            self._scheduler.add_job(
                lambda: self._run_cycle_safe(cycle_type="premarket"),
                trigger=self._build_trigger(premarket_cfg.get("schedule", "cron:08:00")),
                id=self._premarket_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
        afterhours_cfg = self._extended_hours_cfg.get("afterhours") or {}
        if afterhours_cfg.get("enabled"):
            self._scheduler.add_job(
                lambda: self._run_cycle_safe(cycle_type="afterhours"),
                trigger=self._build_trigger(afterhours_cfg.get("schedule", "cron:17:00")),
                id=self._afterhours_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )

    def _run_cycle_safe(self, cycle_type: str | None = None) -> None:
        """Wrapper that catches exceptions to avoid killing the scheduler."""
        try:
            self._engine.run_cycle(cycle_type=cycle_type)
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
