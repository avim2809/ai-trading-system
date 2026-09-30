"""US equity trading-day calendar (NYSE full-day closures), no dependencies.

The allocation sleeves need "is this the first US trading day of a new
month?" without pulling in ``pandas_market_calendars``/``exchange_calendars``
(neither is installed on the production hosts). NYSE's full-day holiday
rules are stable and short, so they are encoded directly here:

* New Year's Day (Jan 1; Sat -> not observed on Fri Dec 31, Sun -> Mon)
* Martin Luther King Jr. Day (3rd Monday of January, since 1998)
* Washington's Birthday / Presidents Day (3rd Monday of February)
* Good Friday (Friday before Easter Sunday)
* Memorial Day (last Monday of May)
* Juneteenth (June 19, observed, since 2022)
* Independence Day (July 4, observed)
* Labor Day (1st Monday of September)
* Thanksgiving (4th Thursday of November)
* Christmas (Dec 25, observed)

Observed rule (NYSE Rule 7.2): a Saturday holiday closes the preceding
Friday, a Sunday holiday closes the following Monday -- except New Year's
Day on a Saturday, which NYSE does *not* move to Friday Dec 31.

One-off closures (national days of mourning, Hurricane Sandy, ...) are not
modelled; on such a day the broker's own market-hours gate in
``LiveTradingEngine.run_cycle`` already skips the cycle, and the sleeve's
rebalance simply happens on the next cycle that does run.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

US_EASTERN = "US/Eastern"


def to_market_date(at: datetime, tz: str = US_EASTERN) -> date:
    """Calendar date of *at* in the market timezone.

    Naive datetimes are treated as UTC, matching the engine's convention
    (``firm.time_utils.utcnow`` returns naive UTC).
    """
    ts = at if at.tzinfo is not None else at.replace(tzinfo=UTC)
    return ts.astimezone(ZoneInfo(tz)).date()


def _easter_sunday(year: int) -> date:
    """Gregorian Easter (anonymous Gregorian / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The *n*-th (1-based) ``weekday`` (Mon=0) of ``month``."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month // 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:  # Saturday -> Friday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday -> Monday
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE closures for *year*."""
    days: set[date] = set()
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:  # Saturday New Year is not observed on Dec 31
        days.add(_observed(new_year))
    # A Saturday Jan 1 of *next* year is also not observed on this Dec 31,
    # so nothing to add for that case.
    if year >= 1998:
        days.add(_nth_weekday(year, 1, 0, 3))  # MLK day
    days.add(_nth_weekday(year, 2, 0, 3))  # Presidents Day
    days.add(_easter_sunday(year) - timedelta(days=2))  # Good Friday
    days.add(_last_weekday(year, 5, 0))  # Memorial Day
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    days.add(_observed(date(year, 7, 4)))  # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    days.add(_observed(date(year, 12, 25)))  # Christmas
    return frozenset(days)


def is_us_trading_day(d: date) -> bool:
    """True when NYSE holds a regular (full or early-close) session on *d*."""
    if d.weekday() >= 5:
        return False
    return d not in nyse_holidays(d.year)


def first_trading_day_of_month(year: int, month: int) -> date:
    d = date(year, month, 1)
    while not is_us_trading_day(d):
        d += timedelta(days=1)
    return d
