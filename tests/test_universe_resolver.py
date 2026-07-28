"""Unit tests for firm.data.universe.UniverseResolver."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from firm.data.universe import UniverseResolver


def _resolver() -> UniverseResolver:
    constituents = pd.DataFrame({
        "symbol": ["ALWAYS", "LATE_JOINER", "GONE_EARLY", "JOIN_AND_LEAVE"],
        "added_date": [pd.NaT, "2020-06-01", pd.NaT, "2020-03-01"],
        "removed_date": [pd.NaT, pd.NaT, "2020-02-01", "2020-04-01"],
    })
    return UniverseResolver(constituents)


class TestSymbolsAsof:
    def test_always_active_member_included_at_any_date(self):
        r = _resolver()
        assert "ALWAYS" in r.symbols_asof(datetime(2019, 1, 1))
        assert "ALWAYS" in r.symbols_asof(datetime(2025, 1, 1))

    def test_excludes_before_added_date(self):
        r = _resolver()
        assert "LATE_JOINER" not in r.symbols_asof(datetime(2020, 1, 1))
        assert "LATE_JOINER" in r.symbols_asof(datetime(2020, 7, 1))

    def test_excludes_after_removed_date(self):
        r = _resolver()
        assert "GONE_EARLY" in r.symbols_asof(datetime(2020, 1, 15))
        assert "GONE_EARLY" not in r.symbols_asof(datetime(2020, 3, 1))


class TestSymbolsBetween:
    def test_includes_late_joiner_when_window_covers_its_membership(self):
        r = _resolver()
        union = r.symbols_between(datetime(2020, 1, 1), datetime(2020, 12, 31))
        assert "LATE_JOINER" in union

    def test_excludes_late_joiner_when_window_ends_before_it_joins(self):
        r = _resolver()
        union = r.symbols_between(datetime(2020, 1, 1), datetime(2020, 3, 1))
        assert "LATE_JOINER" not in union

    def test_includes_name_whose_entire_membership_is_inside_the_window(self):
        """JOIN_AND_LEAVE is a member only 2020-03-01..2020-04-01 — a plain
        start/end snapshot union would miss it entirely, but symbols_between
        (window-overlap test) must not."""
        r = _resolver()
        union = r.symbols_between(datetime(2020, 1, 1), datetime(2020, 12, 31))
        assert "JOIN_AND_LEAVE" in union
        # Sanity: it's genuinely absent from both the start and end snapshots.
        assert "JOIN_AND_LEAVE" not in r.symbols_asof(datetime(2020, 1, 1))
        assert "JOIN_AND_LEAVE" not in r.symbols_asof(datetime(2020, 12, 31))

    def test_excludes_name_whose_membership_is_fully_outside_the_window(self):
        r = _resolver()
        union = r.symbols_between(datetime(2020, 5, 1), datetime(2020, 12, 31))
        assert "GONE_EARLY" not in union
        assert "JOIN_AND_LEAVE" not in union

    def test_always_active_member_always_included(self):
        r = _resolver()
        union = r.symbols_between(datetime(2020, 5, 1), datetime(2020, 6, 1))
        assert "ALWAYS" in union


class TestDelistedBetween:
    def test_finds_removals_in_range(self):
        r = _resolver()
        removed = r.delisted_between(datetime(2020, 1, 1), datetime(2020, 12, 31))
        assert "GONE_EARLY" in removed
        assert "JOIN_AND_LEAVE" in removed
        assert "ALWAYS" not in removed
