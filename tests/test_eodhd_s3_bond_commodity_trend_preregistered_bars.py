"""Tests for scripts/eodhd_s3_bond_commodity_trend_preregistered_bars.py —
mechanics only (fingerprint determinism, tier-classification precedence,
variant count, power-report shape). This file is DRAFT (Phase 1: design only,
no return series exists for this candidate yet), so there is no return-series
simulator to test here, matching the convention in
test_insider_cluster_preregistered_bars.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_s3_bond_commodity_trend_preregistered_bars as prereg  # noqa: E402


class TestDraftStatus:
    def test_is_draft_not_frozen(self):
        assert prereg.DRAFT is False
        assert prereg.PREREGISTERED_AT == "2026-09-30T19:18:54Z"

    def test_cleaning_fingerprint_matches_frozen_rule(self):
        # v2, after Amendment 1 — same value asserted in tests/test_eodhd_clean.py
        # for the shared, frozen cleaning rule this design commits to using.
        assert prereg.CLEANING["fingerprint"] == "fc0690f087edaac8c11ddf77b381e58b59c57a893d460f12c7fa782229693054"
        assert prereg.CLEANING["version"] == 2

    def test_amendment_1_is_recorded(self):
        assert prereg.AMENDMENT["id"] == "protocol_amendment_1"
        assert prereg.AMENDMENT["changes_adopted"]
        assert prereg.AMENDMENT["s3_specific_changes_in_this_pass"]


class TestFingerprint:
    def test_deterministic(self):
        assert prereg.bars_fingerprint() == prereg.bars_fingerprint()
        assert len(prereg.bars_fingerprint()) == 64

    def test_changes_if_a_design_dict_changes(self, monkeypatch):
        before = prereg.bars_fingerprint()
        monkeypatch.setitem(prereg.SATELLITE, "total_weight", 0.20)
        after = prereg.bars_fingerprint()
        assert before != after


class TestVariantGrid:
    def test_n_variants_is_five_and_under_the_task_cap_of_six(self):
        assert prereg.N_VARIANTS == len(prereg.VARIANTS) == 5
        assert prereg.N_VARIANTS <= 6

    def test_every_variant_declares_its_sleeve(self):
        for name, v in prereg.VARIANTS.items():
            assert v["sleeve"] in ("bond", "commodity", "combined"), name

    def test_bond_side_is_not_gridded(self):
        # The brief instructs: do not extend Sihvonen's 1-month lookback.
        bond_variants = [v for v in prereg.VARIANTS.values() if v["sleeve"] == "bond"]
        assert len(bond_variants) == 1
        assert bond_variants[0]["lookback_months"] == 1

    def test_combined_variants_reference_declared_sub_variants(self):
        combined = {k: v for k, v in prereg.VARIANTS.items() if v["sleeve"] == "combined"}
        assert len(combined) == 2
        for v in combined.values():
            for dep in v["uses"]:
                assert dep in prereg.VARIANTS

    def test_commodity_primary_is_now_the_12_1_month_peer_reviewed_lookback(self):
        # Coordinator review: 12-1 month (Moskowitz-Ooi-Pedersen / Asness-Moskowitz-Pedersen)
        # is primary; the 3-month practitioner backtest is now the sensitivity variant.
        primary = prereg.VARIANTS["commodity_v1_primary"]
        sensitivity = prereg.VARIANTS["commodity_v2_sensitivity"]
        assert "12-1 month" in primary["signal"]
        assert "reported_not_primary" not in primary
        assert "3-month" in sensitivity["signal"]
        assert sensitivity.get("reported_not_primary") is True


class TestUniverse:
    def test_bond_universe_is_treasuries_only(self):
        assert prereg.UNIVERSE["bond_buckets"]["treasuries_only"] is True
        assert set(prereg.UNIVERSE["bond_buckets"]["tickers"]) == {"SHY", "IEF", "TLT"}

    def test_commodity_universe_size_and_k(self):
        n = prereg.UNIVERSE["commodity_basket"]["n"]
        tickers = prereg.UNIVERSE["commodity_basket"]["tickers"]
        assert len(tickers) == n == 6
        assert prereg.UNIVERSE["commodity_basket"]["k"] == 3

    def test_satellite_weights_sum_to_declared_total(self):
        s = prereg.SATELLITE
        assert s["split"]["bond_sleeve"] + s["split"]["commodity_sleeve"] == pytest.approx(s["total_weight"])


class TestAvailabilityConsistency:
    def test_bond_common_start_is_the_latest_bucket_start(self):
        starts = prereg.AVAILABILITY["bond_tickers_first_clean_date"]
        assert prereg.AVAILABILITY["bond_common_start"] == max(starts.values())

    def test_commodity_common_start_is_the_latest_ticker_start(self):
        starts = prereg.AVAILABILITY["commodity_tickers_first_clean_date"]
        assert prereg.AVAILABILITY["commodity_common_start"] == max(starts.values())

    def test_combined_window_starts_no_earlier_than_either_sleeve(self):
        assert prereg.WINDOWS["combined"]["start"] >= prereg.WINDOWS["bond"]["start"]
        assert prereg.WINDOWS["combined"]["start"] >= prereg.WINDOWS["commodity"]["start"]

    def test_windows_end_at_data_end(self):
        for w in prereg.WINDOWS.values():
            assert w["end"] == prereg.DATA_END

    def test_midpoint_strictly_inside_window(self):
        for name, w in prereg.WINDOWS.items():
            assert w["start"] < w["midpoint"] < w["end"], name


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
        tier_b = {"B_a": b_ok}
        assert prereg.classify(bars, d, tier_b) == want

    def test_empty_bars_is_not_a(self):
        assert prereg.classify({}, False, {"B_a": True}) == "B"
        assert prereg.classify({}, False, {}) == "C"


class TestPowerReport:
    def test_shape(self):
        rows = prereg.power_report()["rows"]
        # 2 sub-candidates x 2 effect scenarios x 2 bet-rate scenarios
        assert len(rows) == 2 * 2 * 2

    def test_bond_is_underpowered_except_the_single_most_optimistic_pair(self):
        # After Amendment 1's longer window (2002-07-26, +24.1y vs the original
        # draft's 2007-01-11, +19.7y), the optimistic-effect x central-bet-rate
        # pair just clears the bar (579.4 available vs 564.5 required) -- every
        # other pair remains underpowered. The report must say so plainly, not
        # claim the bond side is powered across the board.
        rows = {(r["effect"], r["bet_rate_scenario"]): r for r in prereg.power_report()["rows"]
                if r["sub_candidate"] == "bond"}
        assert rows[("optimistic_assumed", "central")]["adequately_powered"] is True
        assert rows[("optimistic_assumed", "conservative")]["adequately_powered"] is False
        assert rows[("conservative_assumed", "conservative")]["adequately_powered"] is False
        assert rows[("conservative_assumed", "central")]["adequately_powered"] is False

    def test_commodity_only_adequately_powered_under_the_most_optimistic_pair(self):
        rows = {(r["effect"], r["bet_rate_scenario"]): r for r in prereg.power_report()["rows"]
                if r["sub_candidate"] == "commodity"}
        assert rows[("optimistic_assumed", "central")]["adequately_powered"] is True
        assert rows[("conservative_assumed", "conservative")]["adequately_powered"] is False
        assert rows[("conservative_assumed", "central")]["adequately_powered"] is False
        assert rows[("optimistic_assumed", "conservative")]["adequately_powered"] is False

    def test_larger_effect_needs_fewer_bets_for_both_sub_candidates(self):
        rows = {(r["sub_candidate"], r["effect"], r["bet_rate_scenario"]): r
                for r in prereg.power_report()["rows"]}
        for sub in ("bond", "commodity"):
            cons = rows[(sub, "conservative_assumed", "central")]
            opt = rows[(sub, "optimistic_assumed", "central")]
            assert opt["n_required_80pct_power"] < cons["n_required_80pct_power"]

    def test_central_bet_rate_never_needs_more_available_bets_than_conservative(self):
        rows = {(r["sub_candidate"], r["effect"], r["bet_rate_scenario"]): r
                for r in prereg.power_report()["rows"]}
        for sub in ("bond", "commodity"):
            for effect in ("conservative_assumed", "optimistic_assumed"):
                cons = rows[(sub, effect, "conservative")]
                cent = rows[(sub, effect, "central")]
                assert cent["n_available"] >= cons["n_available"]
                assert cons["n_required_80pct_power"] == cent["n_required_80pct_power"]


class TestRequiredN:
    def test_larger_effect_needs_fewer_bets(self):
        n_small = prereg.required_n(mu_bps=50, sigma_bps=2000, alpha_one_sided=0.05, power=0.8)
        n_large = prereg.required_n(mu_bps=200, sigma_bps=2000, alpha_one_sided=0.05, power=0.8)
        assert n_large < n_small

    def test_higher_power_needs_more_bets(self):
        n_low_power = prereg.required_n(mu_bps=50, sigma_bps=2000, alpha_one_sided=0.05, power=0.6)
        n_high_power = prereg.required_n(mu_bps=50, sigma_bps=2000, alpha_one_sided=0.05, power=0.9)
        assert n_high_power > n_low_power


class TestAlphaMatchesSharedProtocol:
    def test_alpha_is_the_fixed_shortlist_wide_value(self):
        # protocol §4: 0.05 / 5 shortlist candidates = 0.01, fixed across S1-S5,
        # not derived from this file's own N_VARIANTS.
        assert prereg.BOOTSTRAP["alpha_one_sided"] == pytest.approx(0.01)
        assert prereg.POWER["alpha_one_sided"] == prereg.BOOTSTRAP["alpha_one_sided"]
