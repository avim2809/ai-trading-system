"""Tests for scripts/eodhd_s4_52wk_high_preregistered_bars.py -- mechanics only
(fingerprint determinism, variant cap, tier-classification precedence, power-report
shape). PHASE 1 DRAFT: no return series exists yet, so there is nothing here about
candidate/benchmark performance -- mirrors test_insider_cluster_preregistered_bars.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_s4_52wk_high_preregistered_bars as prereg


class TestDraftStatus:
    def test_is_draft_not_frozen(self):
        assert prereg.DRAFT is False
        assert prereg.PREREGISTERED_AT == "2026-09-30T19:18:54Z"

    def test_variant_cap(self):
        assert prereg.N_VARIANTS == len(prereg.CANDIDATES)
        assert prereg.N_VARIANTS <= 4

    def test_exactly_one_primary_candidate(self):
        primaries = [c for c in prereg.CANDIDATES.values() if c.get("is_primary")]
        assert len(primaries) == 1

    def test_at_most_one_simpler_hold_variant(self):
        non_primary_holds = {h for h, spec in prereg.HOLD_STRUCTURES.items() if not spec["is_primary"]}
        assert len(non_primary_holds) == 1

    def test_cleaning_fingerprint_is_v2(self):
        import eodhd_clean as ec
        assert prereg.OBSERVED["cleaning_fingerprint_v2"] == ec.cleaning_fingerprint()
        assert ec.CLEANING_RULES["version"] == 2


class TestFingerprint:
    def test_deterministic(self):
        assert prereg.bars_fingerprint() == prereg.bars_fingerprint()
        assert len(prereg.bars_fingerprint()) == 64

    def test_changes_if_a_design_dict_changes(self, monkeypatch):
        before = prereg.bars_fingerprint()
        monkeypatch.setitem(prereg.UNIVERSE, "N_primary", 750)
        after = prereg.bars_fingerprint()
        assert before != after


class TestUniverseDesign:
    def test_price_screen_uses_raw_close_not_adjusted(self):
        assert "raw" in prereg.UNIVERSE["per_ticker_screens_at_month_end"]["price_field"].lower()

    def test_liquidity_cut_declares_both_n_values(self):
        assert prereg.UNIVERSE["N_primary"] == 500
        assert prereg.UNIVERSE["N_variant"] == 1000


class TestWindow:
    def test_window_starts_after_the_credible_coverage_transition(self):
        import pandas as pd
        start = pd.Timestamp(prereg.WINDOW["start"])
        cliff = pd.Timestamp(prereg.OBSERVED["credible_coverage_transition"]["first_credible_month_end"])
        assert start > cliff

    def test_window_ordering_and_midpoint(self):
        import pandas as pd
        start = pd.Timestamp(prereg.WINDOW["start"])
        mid = pd.Timestamp(prereg.WINDOW["midpoint"])
        end = pd.Timestamp(prereg.WINDOW["end"])
        assert start < mid < end

    def test_n_choices_supported_throughout_window(self):
        assert prereg.WINDOW["n_eligible_min_in_window"] > prereg.UNIVERSE["N_variant"]
        assert prereg.WINDOW["N_primary_and_N_variant_supported_throughout"] is True

    def test_topn_delisted_fraction_is_low_before_the_cliff_and_high_after(self):
        # This is the survivorship-tilt signature the coordinator asked to check:
        # pre-cliff years should look nothing like the post-cliff / long-run range.
        by_year = prereg.OBSERVED["topN_delisted_fraction_by_year"]
        pre_cliff = [by_year[str(y)]["frac_delisted_top500"] for y in range(1994, 1998)]
        post_cliff = [by_year[str(y)]["frac_delisted_top500"] for y in range(1999, 2006)]
        assert max(pre_cliff) < min(post_cliff)


class TestClassifyPrecedence:
    @pytest.mark.parametrize("a_ok,d,b_ok,want", [
        (True, False, True, "A"),
        (True, True, True, "A"),
        (False, True, True, "D"),
        (False, False, True, "B"),
        (False, False, False, "C"),
    ])
    def test_classify_precedence(self, a_ok, d, b_ok, want):
        bars = {"A1": a_ok, "A2": True}
        tier_b = {"B_a": b_ok, "B_b": True}
        assert prereg.classify(bars, d, tier_b) == want


class TestPowerReport:
    def test_shape(self):
        prereg.POWER["available_window_months"] = 300  # synthetic, for shape-only testing
        report = prereg.power_report()
        rows = report["rows"]
        n_holds = len(prereg.POWER["effective_independent_months_per_calendar_month"])
        n_effects = len(prereg.POWER["effect_size_bps_per_month_gross_decile_spread"])
        n_sigmas = len(prereg.POWER["sigma_monthly_bps_scenarios"])
        assert len(rows) == n_holds * n_effects * n_sigmas

    def test_larger_effect_needs_fewer_effective_months(self):
        prereg.POWER["available_window_months"] = 300
        report = prereg.power_report()
        by_key = {(r["hold"], r["effect"], r["sigma_scenario"]): r for r in report["rows"]}
        for hold in prereg.POWER["effective_independent_months_per_calendar_month"]:
            for sigma in prereg.POWER["sigma_monthly_bps_scenarios"]:
                gross = by_key[(hold, "original_gh_gross", sigma)]
                decayed = by_key[(hold, "decayed_post_publication", sigma)]
                assert gross["n_required_80pct_power_effective_months"] < decayed["n_required_80pct_power_effective_months"]

    def test_overlapping_hold_has_fewer_effective_available_months_than_1month(self):
        prereg.POWER["available_window_months"] = 300
        report = prereg.power_report()
        by_key = {(r["hold"], r["effect"], r["sigma_scenario"]): r for r in report["rows"]}
        row_6mo = by_key[("6_month_overlapping", "original_gh_gross", "central")]
        row_1mo = by_key[("1_month", "original_gh_gross", "central")]
        assert row_6mo["avail_effective_months"] < row_1mo["avail_effective_months"]

    def test_none_available_months_yields_none_powered_flag(self):
        prereg.POWER["available_window_months"] = None
        report = prereg.power_report()
        assert all(r["adequately_powered"] is None for r in report["rows"])


class TestRequiredN:
    def test_larger_effect_needs_fewer_observations(self):
        n_small = prereg.required_n(mu_bps=15, sigma_bps=400, alpha_one_sided=0.01, power=0.8)
        n_large = prereg.required_n(mu_bps=45, sigma_bps=400, alpha_one_sided=0.01, power=0.8)
        assert n_large < n_small


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
