"""Tests for src/firm/strategies/insider_cluster.py — pure event-detection
logic on small synthetic frames. No network, no PitView wiring exercised
(that path is a documented no-op placeholder — see its own test below)."""

from __future__ import annotations

import pandas as pd
import pytest

from firm.strategies.insider_cluster import (
    InsiderClusterStrategy,
    build_owner_activity_index,
    classify_owner,
    compute_cluster_events,
)


def _purchase(issuer_cik, owner_cik, trans_date, known_lag_days=2, ticker="ABCD", issuer_name="ABC CORP"):
    trans_date = pd.Timestamp(trans_date)
    return {
        "issuer_cik": issuer_cik,
        "issuer_name": issuer_name,
        "ticker": ticker,
        "owner_cik": owner_cik,
        "trans_date": trans_date,
        "known_date": trans_date + pd.Timedelta(days=known_lag_days),
    }


class TestClassifyOwner:
    def test_routine_when_traded_same_month_prior_3_years(self):
        activity = {"O1": {"2019-03", "2020-03", "2021-03", "2022-03"}}
        assert classify_owner("O1", pd.Timestamp("2022-03-15"), activity) == "routine"

    def test_opportunistic_when_missing_any_prior_year(self):
        activity = {"O1": {"2020-03", "2021-03"}}  # missing 2019-03
        assert classify_owner("O1", pd.Timestamp("2022-03-15"), activity) == "opportunistic"

    def test_opportunistic_when_owner_unknown(self):
        assert classify_owner("O_NEW", pd.Timestamp("2022-03-15"), {}) == "opportunistic"

    def test_build_owner_activity_index(self):
        activity_df = pd.DataFrame({
            "owner_cik": ["O1", "O1", "O2"],
            "year_month": ["2020-01", "2020-02", "2020-01"],
        })
        idx = build_owner_activity_index(activity_df)
        assert idx["O1"] == {"2020-01", "2020-02"}
        assert idx["O2"] == {"2020-01"}


class TestComputeClusterEvents:
    def test_three_distinct_insiders_within_30_days_fires(self):
        purchases = pd.DataFrame([
            _purchase("ISSUER1", "O1", "2020-01-01"),
            _purchase("ISSUER1", "O2", "2020-01-10"),
            _purchase("ISSUER1", "O3", "2020-01-20"),
        ])
        activity = pd.DataFrame(columns=["owner_cik", "year_month"])  # all opportunistic (no history)
        events = compute_cluster_events(purchases, activity)
        assert len(events) == 1
        assert events.iloc[0]["n_distinct_insiders"] == 3
        assert bool(events.iloc[0]["has_opportunistic"]) is True

    def test_two_insiders_does_not_fire(self):
        purchases = pd.DataFrame([
            _purchase("ISSUER1", "O1", "2020-01-01"),
            _purchase("ISSUER1", "O2", "2020-01-10"),
        ])
        activity = pd.DataFrame(columns=["owner_cik", "year_month"])
        events = compute_cluster_events(purchases, activity)
        assert events.empty

    def test_outside_30_day_window_does_not_count(self):
        purchases = pd.DataFrame([
            _purchase("ISSUER1", "O1", "2020-01-01"),
            _purchase("ISSUER1", "O2", "2020-01-10"),
            _purchase("ISSUER1", "O3", "2020-03-01"),  # >30 days after O1
        ])
        activity = pd.DataFrame(columns=["owner_cik", "year_month"])
        events = compute_cluster_events(purchases, activity)
        assert events.empty

    def test_same_owner_twice_does_not_count_as_two_distinct(self):
        purchases = pd.DataFrame([
            _purchase("ISSUER1", "O1", "2020-01-01"),
            _purchase("ISSUER1", "O1", "2020-01-05"),  # same owner again
            _purchase("ISSUER1", "O2", "2020-01-10"),
            _purchase("ISSUER1", "O3", "2020-01-15"),
        ])
        activity = pd.DataFrame(columns=["owner_cik", "year_month"])
        events = compute_cluster_events(purchases, activity)
        assert len(events) == 1
        assert events.iloc[0]["n_distinct_insiders"] == 3

    def test_rising_edge_only_one_event_per_sustained_cluster(self):
        # 5 distinct insiders all within the window: without rising-edge
        # de-duplication this would fire on the 3rd, 4th AND 5th purchase.
        purchases = pd.DataFrame([
            _purchase("ISSUER1", f"O{i}", f"2020-01-{i:02d}") for i in range(1, 6)
        ])
        activity = pd.DataFrame(columns=["owner_cik", "year_month"])
        events = compute_cluster_events(purchases, activity)
        assert len(events) == 1

    def test_new_cluster_after_count_drops_and_rises_again(self):
        # First cluster: O1/O2/O3 in early Jan. Then a long gap so the
        # trailing window empties out, then a second, independent cluster
        # O4/O5/O6 in March should fire as its own event.
        purchases = pd.DataFrame([
            _purchase("ISSUER1", "O1", "2020-01-01"),
            _purchase("ISSUER1", "O2", "2020-01-05"),
            _purchase("ISSUER1", "O3", "2020-01-10"),
            _purchase("ISSUER1", "O4", "2020-03-01"),
            _purchase("ISSUER1", "O5", "2020-03-05"),
            _purchase("ISSUER1", "O6", "2020-03-10"),
        ])
        activity = pd.DataFrame(columns=["owner_cik", "year_month"])
        events = compute_cluster_events(purchases, activity)
        assert len(events) == 2

    def test_no_opportunistic_insider_excludes_the_cluster(self):
        purchases = pd.DataFrame([
            _purchase("ISSUER1", "O1", "2020-01-01"),
            _purchase("ISSUER1", "O2", "2020-01-10"),
            _purchase("ISSUER1", "O3", "2020-01-20"),
        ])
        # All three owners are classified routine (traded the matching
        # calendar month in each of the prior 3 years) -> no opportunistic
        # insider in the window -> CMP's own tested hypothesis excludes it.
        activity = pd.DataFrame({
            "owner_cik": ["O1", "O1", "O1", "O2", "O2", "O2", "O3", "O3", "O3"],
            "year_month": ["2017-01", "2018-01", "2019-01"] * 3,
        })
        events = compute_cluster_events(purchases, activity)
        assert events.empty

    def test_known_date_is_max_across_qualifying_window_not_trigger_purchase(self):
        purchases = pd.DataFrame([
            _purchase("ISSUER1", "O1", "2020-01-01", known_lag_days=2),
            _purchase("ISSUER1", "O2", "2020-01-10", known_lag_days=20),  # files very late
            _purchase("ISSUER1", "O3", "2020-01-20", known_lag_days=2),
        ])
        activity = pd.DataFrame(columns=["owner_cik", "year_month"])
        events = compute_cluster_events(purchases, activity)
        assert len(events) == 1
        # O2's filing (2020-01-30) is the latest known_date in the window,
        # even though O3's purchase is the chronologically last transaction.
        assert events.iloc[0]["known_date"] == pd.Timestamp("2020-01-30")

    def test_distinct_issuers_are_independent(self):
        purchases = pd.DataFrame([
            _purchase("ISSUER1", "O1", "2020-01-01"),
            _purchase("ISSUER1", "O2", "2020-01-10"),
            _purchase("ISSUER2", "O3", "2020-01-01"),
            _purchase("ISSUER2", "O4", "2020-01-05"),
            _purchase("ISSUER2", "O5", "2020-01-10"),
        ])
        activity = pd.DataFrame(columns=["owner_cik", "year_month"])
        events = compute_cluster_events(purchases, activity)
        assert len(events) == 1
        assert events.iloc[0]["issuer_cik"] == "ISSUER2"

    def test_missing_required_column_raises(self):
        with pytest.raises(ValueError):
            compute_cluster_events(pd.DataFrame({"issuer_cik": ["X"]}), pd.DataFrame())

    def test_empty_input(self):
        events = compute_cluster_events(
            pd.DataFrame(columns=["issuer_cik", "owner_cik", "trans_date", "known_date"]),
            pd.DataFrame(columns=["owner_cik", "year_month"]),
        )
        assert events.empty


class TestInsiderClusterStrategyPlaceholder:
    """The BaseStrategy wrapper must be a safe no-op until PitView grows an
    insider_cluster_events() accessor — it is not wired into any live
    config today."""

    def test_generate_returns_empty_without_pitview_support(self):
        class DummyPitView:
            asof = pd.Timestamp("2020-01-01")
            universe = ["ABCD"]

        strategy = InsiderClusterStrategy()
        signals = strategy.generate(DummyPitView())
        assert signals == []
