"""Shared time helpers.

``datetime.utcnow()`` is deprecated (Python 3.12+, removal slated for 3.16)
in favour of tz-aware ``datetime.now(timezone.utc)``. This codebase's
point-in-time data plumbing (``firm.data.pit_store``, strategy ``asof``
handling, backtest bar dates) is built end-to-end on **naive UTC**
datetimes compared directly against tz-naive pandas date columns; feeding
those comparisons a tz-aware value raises ``TypeError``. ``utcnow()`` below
is the deprecation-safe drop-in replacement that preserves that naive-UTC
contract everywhere it's needed, instead of switching call sites over to
tz-aware values piecemeal (which would silently break PIT filtering).
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC "now" — the deprecation-safe equivalent of ``datetime.utcnow()``."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
