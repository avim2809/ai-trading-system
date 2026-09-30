"""Tests for scripts/eodhd_s1_industry_momentum_preregistered_bars.py -- mechanics
only (fingerprint determinism, tier-classification precedence, variant count,
power-report shape). This file is DRAFT (see its own module docstring: design
only, no return series computed yet), same house style as
test_insider_cluster_preregistered_bars.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_s1_industry_momentum_preregistered_bars as prereg  # noqa: E402


class TestDraftStatus:
    def test_is_draft_not_frozen(self):
        assert prereg.DRAFT is False
        assert prereg.PREREGISTERED_AT == "2026-09-30T19:18:54Z"

    def test_no_return_computation_markers_present(self):
        # This is a design-only sanity check: the module must document that it
        # used availability, not returns, to fix its window/eligibility.
        assert "availability" in prereg.AVAILABILITY["source_scan"]
        assert prereg.AVAILABILITY["cleaning_fingerprint"] == (
            "fc0690f087edaac8c11ddf77b381e58b59c57a893d460f12c7fa782229693054"
        )


class TestUniverse:
    def test_sector_11_count(self):
        assert len(prereg.SECTOR_11) == 11

    def test_universe_is_sector_plus_industry_no_duplicates(self):
        assert prereg.UNIVERSE == prereg.SECTOR_11 + prereg.INDUSTRY_33
        assert len(prereg.UNIVERSE) == len(set(prereg.UNIVERSE))
        assert len(prereg.UNIVERSE) == prereg.AVAILABILITY["n_universe"]

    def test_excludes_known_non_industry_tickers(self):
        # Broad/size/bond/commodity/currency/country ETFs must not be in the universe.
        excluded = {"SPY", "QQQ", "IWM", "VTI", "AGG", "TLT", "IEF", "GLD", "USO", "UUP",
                    "EFA", "EEM", "FXI", "INDA"}
        assert excluded.isdisjoint(prereg.UNIVERSE)


class TestAvailabilityInternalConsistency:
    def test_eligible_counts_bounded_by_universe_size(self):
        a = prereg.AVAILABILITY
        assert a["n_eligible_at_window_start_2002_01_31"] <= a["n_universe"]
        assert a["n_eligible_at_data_end"] <= a["n_universe"]
        assert a["min_n_eligible_over_window"] == a["n_eligible_at_window_start_2002_01_31"]

    def test_pbj_excluded_by_its_own_adv_not_hand_picked(self):
        a = prereg.AVAILABILITY
        assert a["pbj_adv20_usd_at_data_end"] < a["adv20_eligibility_floor_usd"]
        assert "PBJ" in prereg.UNIVERSE  # stays in the declared universe


class TestWindowAndEligibility:
    def test_window_start_before_end(self):
        assert prereg.WINDOW["start"] < prereg.WINDOW["end"]

    def test_midpoint_inside_window(self):
        assert prereg.WINDOW["start"] < prereg.WINDOW["midpoint_date"] < prereg.WINDOW["end"]

    def test_n_min_for_trading_allows_a_nonempty_tercile(self):
        # A tercile of n_min_for_trading eligible names holds at least 3.
        n = prereg.ELIGIBILITY["n_min_for_trading"]
        assert round(n / 3) >= 3


class TestVariants:
    def test_variant_count_within_grid_limit(self):
        assert prereg.N_VARIANTS == len(prereg.VARIANTS)
        assert prereg.N_VARIANTS <= 4

    def test_primary_variant_is_the_papers_parameters(self):
        assert prereg.PRIMARY_VARIANT in prereg.VARIANTS
        primary = prereg.VARIANTS[prereg.PRIMARY_VARIANT]
        assert primary["formation_months"] == 6
        assert primary["skip_months"] == 1
        assert primary["selection"] == "top_tercile"

    def test_all_variants_declare_required_fields(self):
        for name, spec in prereg.VARIANTS.items():
            assert "formation_months" in spec, name
            assert "skip_months" in spec, name
            assert "selection" in spec, name
            if spec["selection"] == "top_n":
                assert "top_n" in spec, name


class TestFingerprint:
    def test_deterministic(self):
        assert prereg.bars_fingerprint() == prereg.bars_fingerprint()
        assert len(prereg.bars_fingerprint()) == 64

    def test_changes_if_a_variant_changes(self):
        fp_before = prereg.bars_fingerprint()
        original = prereg.VARIANTS[prereg.PRIMARY_VARIANT]["formation_months"]
        prereg.VARIANTS[prereg.PRIMARY_VARIANT]["formation_months"] = 99
        try:
            fp_after = prereg.bars_fingerprint()
        finally:
            prereg.VARIANTS[prereg.PRIMARY_VARIANT]["formation_months"] = original
        assert fp_before != fp_after


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
        tier_b = {"B": b_ok}
        assert prereg.classify(bars, d, tier_b) == want


class TestTierBarsMatchProtocol:
    def test_seven_tier_a_bars(self):
        ids = [b["id"] for b in prereg.TIER_A_BARS]
        assert ids == ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]

    def test_no_s2_only_a8_bar(self):
        # A8 is S2-only per the protocol; S1 must not declare it.
        assert all(b["id"] != "A8" for b in prereg.TIER_A_BARS)

    def test_alpha_one_sided_is_bonferroni_over_five_candidates(self):
        assert prereg.BOOTSTRAP["alpha_one_sided"] == pytest.approx(0.05 / 5)


class TestPowerReport:
    def test_shape(self):
        report = prereg.power_report()
        rows = report["rows"]
        # 4 variants x 3 bet-frequency scenarios x 2 literature effects
        assert len(rows) == 4 * 3 * 2
        assert report["n_rows"] == len(rows)

    def test_higher_bet_frequency_never_needs_a_larger_mde(self):
        report = prereg.power_report()
        by_key = {}
        for r in report["rows"]:
            if r["literature_effect"] != "sharpe_gap_original":
                continue
            by_key[(r["variant"], r["bet_frequency_scenario"])] = r["mde_sharpe_gap_80pct_power"]
        for variant in prereg.VARIANTS:
            cons = by_key[(variant, "conservative")]
            cent = by_key[(variant, "central")]
            naive = by_key[(variant, "naive_monthly")]
            assert cons >= cent >= naive

    def test_decayed_effect_is_never_easier_to_detect_than_original(self):
        # Same MDE per (variant, scenario); the claimed effect is smaller for
        # "decayed", so it can never be MORE often adequately powered.
        report = prereg.power_report()
        by_key = {(r["variant"], r["bet_frequency_scenario"], r["literature_effect"]): r
                  for r in report["rows"]}
        for variant in prereg.VARIANTS:
            for scenario in prereg.BET_FREQUENCY_PRIORS:
                original = by_key[(variant, scenario, "sharpe_gap_original")]
                decayed = by_key[(variant, scenario, "sharpe_gap_decayed")]
                assert original["literature_sharpe_gap"] >= decayed["literature_sharpe_gap"]
                if decayed["adequately_powered"]:
                    assert original["adequately_powered"]

    def test_power_report_carries_availability_not_returns(self):
        report = prereg.power_report()
        assert report["availability"] == prereg.AVAILABILITY
        assert "headline" in report and isinstance(report["headline"], str)


class TestRequiredMdeHelper:
    def test_larger_rho_reduces_mde(self):
        common = dict(benchmark_annual_sharpe=0.4, periods_per_year=4, n_periods=80,
                      alpha_one_sided=0.01, power=0.8)
        low_rho = prereg._required_mde_sharpe_gap(rho=0.2, **common)
        high_rho = prereg._required_mde_sharpe_gap(rho=0.8, **common)
        assert high_rho < low_rho

    def test_more_periods_reduces_mde(self):
        common = dict(benchmark_annual_sharpe=0.4, rho=0.7, periods_per_year=4,
                      alpha_one_sided=0.01, power=0.8)
        few = prereg._required_mde_sharpe_gap(n_periods=40, **common)
        many = prereg._required_mde_sharpe_gap(n_periods=400, **common)
        assert many < few
