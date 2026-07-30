"""Tests for firm.data.investing.calendar.fetch_calendar.

Uses a small literal HTML fixture matching the assumed (documented as
unverified against the live site — see the module docstring) table
structure, mirroring how tests/test_fmp_provider.py builds fixture JSON
against FMP's documented response shape. This tests the *parsing logic*,
not that the fixture matches investing.com's actual current markup.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from firm.data.investing.calendar import fetch_calendar
from firm.data.investing.session import PageResponse
from firm.data.providers.base import ProviderError
from firm.live.news_guard import Event

_SAMPLE_HTML = """
<table>
<tr event_timestamp="1706108400">
  <td class="left flagCur noWrap"><span class="ceFlags US"></span> USD</td>
  <td class="left event">Non-Farm Payrolls</td>
  <td class="sentiment">
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
  </td>
  <td class="act">250K</td>
  <td class="fore">200K</td>
  <td class="prev">190K</td>
</tr>
<tr event_timestamp="1706112000">
  <td class="left flagCur noWrap"><span class="ceFlags EU"></span> EUR</td>
  <td class="left event">ECB Rate Decision</td>
  <td class="sentiment">
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
  </td>
  <td class="act">&nbsp;</td>
  <td class="fore">4.50%</td>
  <td class="prev">4.50%</td>
</tr>
<tr event_timestamp="1706115600">
  <td class="left flagCur noWrap"><span class="ceFlags JP"></span> JPY</td>
  <td class="left event">Low-impact filler event</td>
  <td class="sentiment"></td>
  <td class="act">&nbsp;</td>
  <td class="fore">&nbsp;</td>
  <td class="prev">&nbsp;</td>
</tr>
<!-- A row missing an event title should be skipped, not crash the parse. -->
<tr event_timestamp="1706119200">
  <td class="left flagCur noWrap"><span class="ceFlags GB"></span> GBP</td>
  <td class="left event"></td>
  <td class="sentiment"></td>
</tr>
</table>
"""


def _session_returning(html: str):
    session = MagicMock()
    session.get.return_value = PageResponse(status_code=200, text=html)
    return session


class TestFetchCalendar:
    def test_parses_expected_rows(self):
        session = _session_returning(_SAMPLE_HTML)
        events = fetch_calendar(session=session)

        assert len(events) == 3  # the title-less 4th row is skipped
        assert all(isinstance(e, Event) for e in events)

        nfp = events[0]
        assert nfp.title == "Non-Farm Payrolls"
        assert nfp.currency == "USD"
        assert nfp.impact == "high"
        assert nfp.when == datetime.fromtimestamp(1706108400, tz=timezone.utc)

        ecb = events[1]
        assert ecb.currency == "EUR"
        assert ecb.impact == "medium"

        filler = events[2]
        assert filler.impact == "low"

    def test_uses_the_provided_url_path(self):
        session = _session_returning(_SAMPLE_HTML)
        fetch_calendar(session=session)
        session.get.assert_called_once_with("/economic-calendar/")

    def test_reuses_provided_session_without_closing_it(self):
        session = _session_returning(_SAMPLE_HTML)
        fetch_calendar(session=session)
        session.close.assert_not_called()

    def test_creates_and_closes_its_own_session_when_none_given(self, monkeypatch):
        fake_session = _session_returning(_SAMPLE_HTML)
        monkeypatch.setattr(
            "firm.data.investing.calendar.InvestingSession",
            MagicMock(return_value=fake_session),
        )
        fetch_calendar()
        fake_session.close.assert_called_once()

    def test_no_matching_rows_returns_empty_list_not_raise(self):
        session = _session_returning("<html><body>completely different markup</body></html>")
        events = fetch_calendar(session=session)
        assert events == []

    def test_malformed_single_row_is_skipped_not_fatal(self):
        html = _SAMPLE_HTML.replace('event_timestamp="1706112000"', 'event_timestamp="not-a-number"')
        session = _session_returning(html)
        events = fetch_calendar(session=session)
        # The malformed row is dropped; the other 2 valid rows still parse.
        assert len(events) == 2

    def test_session_still_closed_when_fetch_raises(self, monkeypatch):
        fake_session = MagicMock()
        fake_session.get.side_effect = ProviderError("blocked")
        monkeypatch.setattr(
            "firm.data.investing.calendar.InvestingSession",
            MagicMock(return_value=fake_session),
        )
        with pytest.raises(ProviderError):
            fetch_calendar()
        fake_session.close.assert_called_once()
