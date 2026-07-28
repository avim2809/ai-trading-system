"""news-guard — macro-event blackout gate for the live trading pipeline.

Given an instrument and a time, decide approve/block based on proximity to
high-impact economic events (FOMC, NFP, CPI, central-bank decisions...). The
classic "don't trade into the number" guard, as a callable used as a pre-trade
filter in :class:`~firm.live.engine.LiveTradingEngine`.

Data sources, in order:
  1. Live Forex Factory weekly calendar JSON (free, no API key):
     https://nfs.faireconomy.media/ff_calendar_thisweek.json
  2. Bundled offline fallback CSV (``data/events.csv``) when the network is
     unreachable — so the gate is deterministic and testable offline.

Only HIGH-impact events trigger a blackout. The instrument is mapped to the set
of currencies/regions it is exposed to, and only events in that set count.

Ported from the external trading-suite ``news-guard`` skill; the Twilio SMS
alerting was dropped (firm has its own alert path via the live engine).
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
BUNDLED_CSV = Path(__file__).resolve().parent / "data" / "events.csv"

# Default blackout window: 30 min before the event, 15 min after.
DEFAULT_BEFORE_MIN = 30
DEFAULT_AFTER_MIN = 15

KNOWN_CCYS = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY"}

# Non-FX instruments -> the currencies whose high-impact prints move them.
INSTRUMENT_MAP = {
    "SPY": {"USD"}, "SPX": {"USD"}, "ES": {"USD"}, "US500": {"USD"},
    "QQQ": {"USD"}, "NDX": {"USD"}, "NQ": {"USD"}, "US100": {"USD"},
    "DIA": {"USD"}, "DJI": {"USD"}, "US30": {"USD"}, "YM": {"USD"},
    "IWM": {"USD"}, "RUT": {"USD"},
    "DAX": {"EUR"}, "GER40": {"EUR"},
    "FTSE": {"GBP"}, "UK100": {"GBP"},
    "NIKKEI": {"JPY"}, "JP225": {"JPY"},
    "XAUUSD": {"USD"}, "GOLD": {"USD"}, "XAGUSD": {"USD"}, "SILVER": {"USD"},
    "BTC": {"USD"}, "BTCUSD": {"USD"}, "BTCUSDT": {"USD"}, "XBTUSD": {"USD"},
    "ETH": {"USD"}, "ETHUSD": {"USD"}, "ETHUSDT": {"USD"},
    "SOL": {"USD"}, "SOLUSD": {"USD"},
}

CRYPTO_INSTRUMENTS = {
    "BTC", "BTCUSD", "BTCUSDT", "XBTUSD",
    "ETH", "ETHUSD", "ETHUSDT", "SOL", "SOLUSD",
}
CRYPTO_KEYWORDS = ("crypto", "bitcoin", "btc", "ethereum", "etf", "sec ")


@dataclass(frozen=True)
class Event:
    title: str
    currency: str
    impact: str
    when: datetime  # timezone-aware

    def to_public(self) -> dict:
        return {
            "title": self.title,
            "currency": self.currency,
            "impact": self.impact,
            "time": self.when.astimezone(timezone.utc).isoformat(),
        }


# --------------------------------------------------------------------------
# Time / instrument helpers
# --------------------------------------------------------------------------


def parse_time(value: str) -> datetime:
    """Parse an ISO-8601 time. Accepts a trailing 'Z'. Assumes UTC if naive."""
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def instrument_currencies(instrument: str) -> set[str]:
    """Map an instrument to the set of currencies whose events move it."""
    sym = instrument.upper().replace("/", "").replace("-", "").replace("_", "")
    if sym in INSTRUMENT_MAP:
        return set(INSTRUMENT_MAP[sym])
    if len(sym) == 6:
        base, quote = sym[:3], sym[3:]
        ccys = {c for c in (base, quote) if c in KNOWN_CCYS}
        if len(ccys) == 2:
            return ccys
    if sym in KNOWN_CCYS:
        return {sym}
    # Unknown: fall back to US macro (the most broadly systemic driver).
    return {"USD"}


def is_crypto(instrument: str) -> bool:
    sym = instrument.upper().replace("/", "").replace("-", "").replace("_", "")
    return sym in CRYPTO_INSTRUMENTS


# --------------------------------------------------------------------------
# Calendar loading (live, then bundled fallback)
# --------------------------------------------------------------------------


def _coerce_event(title, country, impact, when_raw) -> Optional[Event]:
    try:
        when = parse_time(str(when_raw))
    except (ValueError, TypeError):
        return None
    return Event(
        title=str(title).strip(),
        currency=str(country).strip().upper(),
        impact=str(impact).strip(),
        when=when,
    )


def load_from_forexfactory(timeout: float = 8.0) -> list[Event]:
    """Fetch the free, keyless Forex Factory weekly JSON. Raises on failure."""
    import requests  # imported lazily so offline use needs no network stack

    resp = requests.get(
        FF_URL, timeout=timeout, headers={"User-Agent": "firm-news-guard/1.0"}
    )
    resp.raise_for_status()
    rows = resp.json()
    out: list[Event] = []
    for r in rows:
        ev = _coerce_event(
            r.get("title"), r.get("country"), r.get("impact"), r.get("date")
        )
        if ev:
            out.append(ev)
    return out


def load_from_csv(path: Path = BUNDLED_CSV) -> list[Event]:
    """Load the bundled offline fallback calendar."""
    out: list[Event] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for r in reader:
            ev = _coerce_event(
                r.get("title"), r.get("currency"), r.get("impact"), r.get("datetime")
            )
            if ev:
                out.append(ev)
    return out


def bundled_csv_age_hours(path: Path = BUNDLED_CSV) -> Optional[float]:
    """Age of the bundled offline calendar file in hours, or ``None`` if it
    can't be stat'd (missing/permissions) — used to size how stale a
    live-calendar-fetch-failure fallback is for the engine's alert."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return (time.time() - mtime) / 3600.0


def load_events(offline: bool = False) -> tuple[list[Event], str]:
    """Return (events, source). Falls back to the bundled CSV on any failure."""
    if not offline:
        try:
            events = load_from_forexfactory()
            if events:
                log.info(
                    "news-guard: loaded %d events from Forex Factory live calendar",
                    len(events),
                )
                return events, "forexfactory"
            log.warning(
                "news-guard: Forex Factory returned no events; "
                "falling back to bundled offline calendar"
            )
        except Exception as exc:  # fall through to the offline calendar
            log.warning(
                "news-guard: live calendar fetch failed (%s); "
                "falling back to bundled offline calendar", exc,
            )
    events = load_from_csv()
    age_hours = bundled_csv_age_hours()
    log.info(
        "news-guard: loaded %d events from bundled offline calendar (%s, age=%s)",
        len(events), BUNDLED_CSV,
        f"{age_hours:.1f}h" if age_hours is not None else "unknown",
    )
    return events, "bundled-csv"


# --------------------------------------------------------------------------
# Core decision
# --------------------------------------------------------------------------


def _relevant(event: Event, ccys: set[str], crypto: bool) -> bool:
    if event.impact.lower() != "high":
        return False
    if event.currency in ccys:
        return True
    if crypto:
        text = event.title.lower()
        return any(k in text for k in CRYPTO_KEYWORDS)
    return False


def decide(
    instrument: str,
    at: datetime,
    events: Iterable[Event],
    before_min: int = DEFAULT_BEFORE_MIN,
    after_min: int = DEFAULT_AFTER_MIN,
    source: str = "unknown",
) -> dict:
    """Pure decision function — no I/O. Block if ``at`` sits inside the blackout
    window of any relevant high-impact event."""
    ccys = instrument_currencies(instrument)
    crypto = is_crypto(instrument)
    relevant = sorted(
        (e for e in events if _relevant(e, ccys, crypto)), key=lambda e: e.when
    )

    before = timedelta(minutes=before_min)
    after = timedelta(minutes=after_min)

    blocking = None
    for ev in relevant:
        if ev.when - before <= at <= ev.when + after:
            blocking = ev
            break

    upcoming = next((e for e in relevant if e.when >= at), None)
    next_event = upcoming.to_public() if upcoming else None
    minutes_until = (
        round((upcoming.when - at).total_seconds() / 60.0, 1) if upcoming else None
    )

    if blocking is not None:
        delta_min = round((blocking.when - at).total_seconds() / 60.0, 1)
        if delta_min > 0:
            timing = f"in {delta_min:g} min"
        elif delta_min < 0:
            timing = f"{abs(delta_min):g} min ago"
        else:
            timing = "now"
        reason = (
            f"BLOCK: {blocking.currency} {blocking.title} ({timing}) is inside the "
            f"{before_min}m-before/{after_min}m-after blackout for {instrument.upper()}."
        )
        log.debug("news-guard decide[%s]: %s (source=%s)", instrument.upper(), reason, source)
        return {
            "decision": "block",
            "reason": reason,
            "instrument": instrument.upper(),
            "currencies": sorted(ccys),
            "at": at.astimezone(timezone.utc).isoformat(),
            "blocking_event": blocking.to_public(),
            "next_event": next_event,
            "minutes_until": minutes_until,
            "source": source,
        }

    if next_event is None:
        reason = (
            f"APPROVE: no high-impact {'/'.join(sorted(ccys))} events found for "
            f"{instrument.upper()} in the loaded calendar."
        )
    else:
        reason = (
            f"APPROVE: next high-impact event ({next_event['currency']} "
            f"{next_event['title']}) is {minutes_until:g} min away — outside the "
            f"{before_min}m blackout for {instrument.upper()}."
        )
    return {
        "decision": "approve",
        "reason": reason,
        "instrument": instrument.upper(),
        "currencies": sorted(ccys),
        "at": at.astimezone(timezone.utc).isoformat(),
        "blocking_event": None,
        "next_event": next_event,
        "minutes_until": minutes_until,
        "source": source,
    }


def evaluate(
    instrument: str,
    at: str | datetime,
    before_min: int = DEFAULT_BEFORE_MIN,
    after_min: int = DEFAULT_AFTER_MIN,
    offline: bool = False,
    events: Iterable[Event] | None = None,
) -> dict:
    """Convenience wrapper: load the calendar (unless *events* is supplied) and
    decide. Passing *events* keeps the call fully offline and deterministic."""
    at_dt = at if isinstance(at, datetime) else parse_time(at)
    if at_dt.tzinfo is None:
        at_dt = at_dt.replace(tzinfo=timezone.utc)
    if events is not None:
        return decide(instrument, at_dt, events, before_min, after_min, "provided")
    loaded, source = load_events(offline=offline)
    return decide(instrument, at_dt, loaded, before_min, after_min, source)
