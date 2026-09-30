"""Tests for scripts/eodhd_s2_breadth_overlay_preregistered_bars.py --
mechanics only (DRAFT status, fingerprint determinism, tier-classification
precedence, the declared variant grid, and the episode_count helper used for
the power analysis). This file is DRAFT (module docstring: no return has
been computed yet), so -- like test_insider_cluster_preregistered_bars.py --
there is no return-series simulator to test here."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_s2_breadth_overlay_preregistered_bars as prereg  # noqa: E402


class TestDraftStatus:
    def test_is_still_draft(self):
        # Phase 1 of this candidate: design frozen against AVAILABILITY only,
        # no return computed -- must not claim to be frozen yet.
        assert prereg.DRAFT is True
        assert prereg.PREREGISTERED_AT is None

    def test_cleaning_fingerprint_matches_the_frozen_rule(self):
        import eodhd_clean as ec
        assert prereg.CLEANING_FINGERPRINT == ec.cleaning_fingerprint()


class TestFingerprint:
    def test_deterministic(self):
        assert prereg.bars_fingerprint() == prereg.bars_fingerprint()
        assert len(prereg.bars_fingerprint()) == 64

    def test_changes_if_a_variant_changes(self, monkeypatch):
        before = prereg.bars_fingerprint()
        monkeypatch.setitem(prereg.VARIANTS["V1_primary"], "threshold", 0.55)
        after = prereg.bars_fingerprint()
        assert before != after


class TestVariantGrid:
    def test_n_variants_is_small_and_matches_the_dict(self):
        assert prereg.N_VARIANTS == len(prereg.VARIANTS)
        assert prereg.N_VARIANTS <= 4

    def test_every_variant_declares_measure_threshold_direction_destination(self):
        valid_measures = {prereg.BREADTH["primary"]["id"], prereg.BREADTH["alternative"]["id"]}
        for vid, v in prereg.VARIANTS.items():
            for key in ("measure", "threshold", "direction", "destination"):
                assert key in v, (vid, key)
            assert v["measure"] in valid_measures, (vid, v["measure"])
            assert v["destination"] in ("IEF", "BIL")
            assert v["direction"] == "below"

    def test_at_least_one_variant_uses_each_declared_measure(self):
        measures = {v["measure"] for v in prereg.VARIANTS.values()}
        assert measures == {prereg.BREADTH["primary"]["id"], prereg.BREADTH["alternative"]["id"]}

    def test_primary_threshold_is_the_natural_midpoint_of_a_proportion(self):
        # V1 is the primary variant; 0.50 is a declared, pre-return convention
        # for a measure bounded in [0, 1], not a fit threshold.
        assert prereg.VARIANTS["V1_primary"]["threshold"] == 0.50
        assert prereg.VARIANTS["V1_primary"]["measure"] == prereg.BREADTH["primary"]["id"]


class TestTierBars:
    def test_s2_only_defensive_bar_is_present(self):
        ids = [b["id"] for b in prereg.TIER_A_BARS]
        assert "A8" in ids
        a8 = next(b for b in prereg.TIER_A_BARS if b["id"] == "A8")
        assert "drawdown" in a8["rule"].lower() and "calmar" in a8["rule"].lower()

    def test_alpha_is_the_shared_protocol_value_not_bonferroni_over_5(self):
        # The shared protocol fixes alpha at 0.05/5 = 0.01 for every shortlist
        # candidate -- this file must not recompute its own 0.05/N_VARIANTS.
        assert prereg.BOOTSTRAP["alpha_one_sided"] == pytest.approx(0.01)

    def test_placebo_rule_matches_the_protocols_s2_specific_text(self):
        assert "63" in prereg.PLACEBO["rule"] and "permut" in prereg.PLACEBO["rule"]

    def test_dsr_prior_trials_is_the_frozen_placeholder(self):
        # Exact placeholder text mandated for phase 1 (not a number yet --
        # the other 4 shortlist candidates' grids aren't all frozen).
        assert prereg.DSR["prior_trials"] == "190 + other shortlist variants (fixed at freeze)"


class TestClassifyPrecedence:
    @pytest.mark.parametrize("a_ok,d,b_ok,want", [
        (True, False, True, "A"),
        (True, True, True, "A"),
        (False, True, True, "D"),
        (False, False, True, "B"),
        (False, False, False, "C"),
    ])
    def test_classify_precedence(self, a_ok, d, b_ok, want):
        bars = {"A1": a_ok, "A8": True}
        tier_b = {"B_a": b_ok, "B_b": True}
        assert prereg.classify(bars, d, tier_b) == want

    def test_empty_bars_is_never_tier_a(self):
        assert prereg.classify({}, False, {"B_a": True}) == "B"
        assert prereg.classify({}, False, {}) == "C"


class TestEpisodeCount:
    def test_no_state_ever_on(self):
        r = prereg.episode_count([False] * 12)
        assert r == {"n_months": 12, "n_flips": 0, "n_on_episodes": 0, "pct_on": 0.0}

    def test_always_on_is_one_episode_zero_flips(self):
        r = prereg.episode_count([True] * 12)
        assert r["n_flips"] == 0
        assert r["n_on_episodes"] == 1
        assert r["pct_on"] == 1.0

    def test_single_on_off_cycle(self):
        # off, off, on, on, on, off, off -- one episode, two flips (on then off)
        r = prereg.episode_count([False, False, True, True, True, False, False])
        assert r["n_on_episodes"] == 1
        assert r["n_flips"] == 2

    def test_two_separate_episodes(self):
        r = prereg.episode_count([False, True, False, True, False])
        assert r["n_on_episodes"] == 2
        assert r["n_flips"] == 4  # off->on, on->off, off->on, on->off

    def test_starting_on_counts_as_an_episode(self):
        r = prereg.episode_count([True, False, False])
        assert r["n_on_episodes"] == 1
        assert r["n_flips"] == 1

    def test_empty_series(self):
        r = prereg.episode_count([])
        assert r["n_months"] == 0
        assert r["pct_on"] == 0.0


class TestBenchmarksAndPrimary:
    def test_three_benchmarks_declared(self):
        assert set(prereg.BENCHMARKS) == {"BM1_SPY", "BM2_60_40", "BM3_SPY_VT"}

    def test_primary_benchmark_is_plain_60_40(self):
        assert prereg.PRIMARY_BENCHMARK == "BM2_60_40"


class TestWindowAndObserved:
    def test_window_start_is_not_before_the_protocol_floor(self):
        start = pd.Timestamp(prereg.WINDOW["start"])
        floor = pd.Timestamp("1993-02-01")
        assert start >= floor

    def test_window_start_is_justified_by_raw_coverage_not_screened_eligible_count(self):
        rationale = prereg.WINDOW["start_rationale"]
        assert "raw" in rationale.lower()
        assert "1996" in rationale and "1997" in rationale  # the vendor-depth cliff year pair

    def test_raw_coverage_table_shows_the_pre_1998_cliff(self):
        cov = prereg.OBSERVED["raw_ticker_file_coverage_by_year"]
        # the documented vendor-depth artifact: a near-3x jump in a single year
        assert cov[1997] > 2 * cov[1996]
        # 1995 coverage independently matches the coordinator's ~1,900 estimate
        assert 1700 <= cov[1995] <= 2100

    def test_raw_coverage_is_stable_from_the_chosen_start_year_onward(self):
        cov = prereg.OBSERVED["raw_ticker_file_coverage_by_year"]
        window_years = [y for y in cov if y >= 2002]
        lo, hi = min(cov[y] for y in window_years), max(cov[y] for y in window_years)
        mean = sum(cov[y] for y in window_years) / len(window_years)
        # tight band (well within the +/-3% this file claims), unlike the pre-2002 ramp
        assert (hi - lo) / mean < 0.10

    def test_eligible_count_at_window_start_is_positive_and_below_the_long_run_median(self):
        # window start should be past the ramp-up, not at its very first nonzero day
        assert prereg.OBSERVED["eligible_count_at_window_start"] > 0
        assert (prereg.OBSERVED["eligible_count_at_window_start"]
                < prereg.OBSERVED["eligible_count_median_full_history"])

    def test_midpoint_is_between_start_and_end(self):
        start = pd.Timestamp(prereg.WINDOW["start"])
        end = pd.Timestamp(prereg.WINDOW["end"])
        mid = pd.Timestamp(prereg.WINDOW["midpoint"])
        assert start < mid < end

    def test_variant_state_change_counts_are_internally_consistent(self):
        # V1 and V2 share a signal/threshold (only the destination differs),
        # so their flip/episode counts must be identical.
        by_variant = prereg.OBSERVED["n_state_changes_by_variant"]
        assert by_variant["V1_primary"] == by_variant["V2_cash_destination"]
        # the stricter threshold (V3) must flip less often than the primary (V1)
        assert by_variant["V3_stricter_threshold"]["n_flips"] < by_variant["V1_primary"]["n_flips"]
        # n_flips == 2 * n_on_episodes, or 2*n_on_episodes - 1 if still "on" at the window's end
        for v in by_variant.values():
            assert v["n_flips"] in (2 * v["n_on_episodes"], 2 * v["n_on_episodes"] - 1)

    def test_primary_variant_episode_count_is_below_the_rule_of_thumb_minimum(self):
        # the power analysis's own headline: V1/V2/V3 are underpowered by the
        # 20-30-episode rule of thumb stated in POWER_NOTES.
        by_variant = prereg.OBSERVED["n_state_changes_by_variant"]
        assert by_variant["V1_primary"]["n_on_episodes"] < 20
        assert "underpowered" in prereg.POWER_NOTES["observed_2026_09_30"].lower()


class TestUniverseScreenMatchesBuilder:
    def test_thresholds_match_the_breadth_builder_module(self):
        import eodhd_breadth as eb
        assert prereg.UNIVERSE["sma_window"] == eb.UNIVERSE_SCREEN["sma_window"]
        assert prereg.UNIVERSE["price_min_usd"] == eb.UNIVERSE_SCREEN["price_min_usd"]
        assert prereg.UNIVERSE["adv20_min_usd"] == eb.UNIVERSE_SCREEN["adv20_min_usd"]
