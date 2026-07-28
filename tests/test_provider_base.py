"""Tests for shared provider helpers in `firm.data.providers.base`."""

from __future__ import annotations

import pandas as pd

from firm.data.providers.base import (
    FUNDAMENTALS_PUBLICATION_LAG_DAYS,
    resolve_filing_date,
)


class TestResolveFilingDate:
    def test_prefers_real_filed_date(self):
        ts = resolve_filing_date("2024-09-28", "2024-11-01", symbol="AAPL")
        assert ts == pd.Timestamp("2024-11-01")

    def test_falls_back_to_lag_heuristic_when_filed_is_none(self):
        ts = resolve_filing_date("2024-09-28", None, symbol="AAPL")
        assert ts == pd.Timestamp("2024-09-28") + pd.Timedelta(
            days=FUNDAMENTALS_PUBLICATION_LAG_DAYS
        )

    def test_falls_back_to_lag_heuristic_when_filed_is_empty_string(self):
        ts = resolve_filing_date("2024-09-28", "", symbol="AAPL")
        assert ts == pd.Timestamp("2024-09-28") + pd.Timedelta(
            days=FUNDAMENTALS_PUBLICATION_LAG_DAYS
        )

    def test_falls_back_to_lag_heuristic_on_unparseable_filed_date(self):
        ts = resolve_filing_date("2024-09-28", "not-a-date", symbol="AAPL")
        assert ts == pd.Timestamp("2024-09-28") + pd.Timedelta(
            days=FUNDAMENTALS_PUBLICATION_LAG_DAYS
        )

    def test_truncates_filed_datetime_to_date(self):
        """FMP's acceptedDate-style timestamps include time-of-day; filed
        dates should be truncated to a plain date for merge consistency."""
        ts = resolve_filing_date("2024-09-28", "2024-11-01 18:06:25", symbol="AAPL")
        assert ts == pd.Timestamp("2024-11-01")
