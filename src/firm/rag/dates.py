"""Date normalization for RAG document metadata.

Ingestors receive dates in many shapes (RFC-822 RSS ``pubDate``, ISO-8601
with time, AlphaVantage ``YYYYMMDDTHHMMSS``, fiscal-quarter labels).  To make
point-in-time retrieval correct, every stored ``date`` must be a single
lexicographically-sortable ISO ``YYYY-MM-DD`` *availability* date so a
``date <= asof`` filter behaves identically to the price PIT store.

Unknown/unparseable dates normalize to :data:`UNKNOWN_DATE` (far future) so an
as-of query *excludes* them — failing closed against look-ahead rather than
silently admitting a doc of unknown vintage.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

# Far-future sentinel: a doc with an unknown date is treated as "not yet
# available" by any as-of filter, so it is never leaked into a past decision.
UNKNOWN_DATE = "9999-12-31"

# Min-date sentinel for timeless reference docs (strategy docstrings, config)
# so they remain retrievable at every as-of.
ALWAYS_AVAILABLE_DATE = "1900-01-01"

# Typical lag between a fiscal quarter end and when its results/transcript
# actually become public.  Used to turn a fiscal-period label into a
# conservative availability date.
_EARNINGS_REPORTING_LAG = timedelta(days=45)

_QUARTER_RE = re.compile(r"^(\d{4})[-_ ]?Q([1-4])$", re.IGNORECASE)
_COMPACT_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})")  # YYYYMMDD[THHMMSS]
_QUARTER_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


def normalize_date(value: object) -> str:
    """Return an ISO ``YYYY-MM-DD`` availability date for *value*.

    Returns :data:`UNKNOWN_DATE` when the input is empty or unparseable.
    """
    if value is None:
        return UNKNOWN_DATE
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text:
        return UNKNOWN_DATE

    # Fiscal-quarter label, e.g. "2023-Q1" -> quarter end + reporting lag.
    m = _QUARTER_RE.match(text)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        month, day = _QUARTER_END[q]
        avail = datetime(year, month, day) + _EARNINGS_REPORTING_LAG
        return avail.strftime("%Y-%m-%d")

    # Already an ISO date/datetime (possibly with a trailing 'Z').
    iso_candidate = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate).strftime("%Y-%m-%d")
    except ValueError:
        pass

    # Compact AlphaVantage form: YYYYMMDDTHHMMSS / YYYYMMDD.
    m = _COMPACT_RE.match(text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            return UNKNOWN_DATE

    # RFC-822 RSS pubDate, e.g. "Mon, 02 Jan 2023 14:30:00 GMT".
    try:
        return parsedate_to_datetime(text).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return UNKNOWN_DATE
