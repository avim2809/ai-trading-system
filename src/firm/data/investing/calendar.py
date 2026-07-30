"""Investing.com's economic calendar — enriched alternative to the free
Forex Factory feed already used by :mod:`firm.live.news_guard`.

Confirmed (2026-07-30, live check) to be a public page — no Investing.com
Pro subscription or login required, unlike the other three services in this
package. It is, however, still blocked for plain HTTP clients (see
:mod:`firm.data.investing.session`'s module docstring: a bare ``curl`` with
a realistic browser User-Agent got HTTP 403), so it goes through the same
:class:`~firm.data.investing.session.InvestingSession` browser-driven fetch.

Known gap: the HTML parsing below targets the calendar table's historically
documented structure (``tr[event_timestamp]`` rows; ``td.flagCur`` /
``td.event`` / ``td.sentiment`` cells) — it has NOT been verified against
the current live markup (this environment cannot run a browser against the
live site). This is designed to fail safe, not silently wrong: if the
expected structure isn't found, :func:`fetch_calendar` logs a clear warning
and returns an empty list rather than raising or fabricating data — and
:func:`firm.live.news_guard.load_events` already treats an empty/failed
Investing.com fetch as a reason to fall through to Forex Factory, then the
bundled CSV, so a stale parser degrades gracefully rather than blocking
trading. Fix the selectors here once the real markup has been inspected
(e.g. via the Phase 0 smoke test or a one-off authenticated dump).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from firm.data.investing.session import InvestingSession
from firm.data.providers.base import ProviderError
from firm.live.news_guard import Event

log = logging.getLogger(__name__)

_CALENDAR_PATH = "/economic-calendar/"

# Best-effort bull-icon-count -> impact mapping — see the "Known gap" note
# in the module docstring.
_IMPACT_BY_BULL_COUNT = {3: "high", 2: "medium", 1: "low"}
_DEFAULT_IMPACT = "low"


def fetch_calendar(session: InvestingSession | None = None) -> list[Event]:
    """Fetch this week's economic calendar from Investing.com.

    Args:
        session: Reuse an existing, already-authenticated (or not — this
            page needs no login) :class:`InvestingSession`; when omitted, a
            throwaway one is created and closed internally (a fresh browser
            launch just for this one fetch — pass a shared session when
            calling this alongside other Investing.com fetchers to avoid
            paying that cost twice).

    Returns:
        Parsed :class:`~firm.live.news_guard.Event` list, or ``[]`` on any
        parse failure (never raises for a markup-shape mismatch — only for
        the scraper being disabled or the fetch itself failing, both of
        which are ``ProviderError``\\ s the caller should already be
        catching per the existing news-guard fallback ladder).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ProviderError(
            "beautifulsoup4 is not installed — install the 'investing' extra "
            "(pip install -e '.[investing]')."
        ) from exc

    own_session = session is None
    session = session or InvestingSession()
    try:
        resp = session.get(_CALENDAR_PATH)
    finally:
        if own_session:
            session.close()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("tr[event_timestamp]")
    if not rows:
        log.warning(
            "investing_calendar_no_rows_parsed — the table structure may "
            "have changed (see the Known-gap note in "
            "firm.data.investing.calendar); returning no events so the "
            "caller falls back to forexfactory/the bundled CSV."
        )
        return []

    events: list[Event] = []
    for row in rows:
        try:
            event = _parse_row(row)
        except Exception as exc:
            log.debug("investing_calendar_row_parse_failed: %s", exc, exc_info=True)
            continue
        if event is not None:
            events.append(event)

    log.info("investing_calendar_loaded events=%d", len(events))
    return events


def _parse_row(row) -> Event | None:
    ts_raw = row.get("event_timestamp")
    if not ts_raw:
        return None
    when = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)

    currency_cell = row.select_one("td.flagCur")
    currency = currency_cell.get_text(strip=True)[-3:].upper() if currency_cell else ""

    event_cell = row.select_one("td.event")
    title = event_cell.get_text(strip=True) if event_cell else ""
    if not title:
        return None

    impact_cell = row.select_one("td.sentiment")
    bulls = len(impact_cell.select("i[class*='Bullish']")) if impact_cell else 0
    impact = _IMPACT_BY_BULL_COUNT.get(bulls, _DEFAULT_IMPACT)

    return Event(title=title, currency=currency, impact=impact, when=when)
