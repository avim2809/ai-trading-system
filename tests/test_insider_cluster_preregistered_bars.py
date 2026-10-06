"""Tests for scripts/insider_cluster_preregistered_bars.py — mechanics only
(fingerprint determinism, tier-classification precedence, power-report
shape). This file is DRAFT (see its own module docstring: no price data
exists yet for this candidate), so there is no return-series simulator to
test here, unlike test_alt_premia_evaluation.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import insider_cluster_preregistered_bars as prereg


class TestFrozenStatus:
    def test_is_frozen_with_timestamp(self):
        # Frozen 2026-09-30 once EODHD prices existed and before any event
        # return was computed (see FREEZE_NOTES).
        assert prereg.DRAFT is False
        assert prereg.PREREGISTERED_AT == "2026-09-30T17:15:00Z"

    def test_freeze_notes_fix_the_open_design_choices(self):
        notes = prereg.FREEZE_NOTES
        for key in ("universe_filter", "event_sets", "overlap_rule", "returns", "benchmark_primary",
                    "delisting_stress", "bootstrap", "calendar_time_portfolio", "tier_b"):
            assert key in notes, key
        assert any(b["id"] == "A7" for b in prereg.TIER_A_BARS)


class TestFingerprint:
    def test_deterministic(self):
        assert prereg.bars_fingerprint() == prereg.bars_fingerprint()
        assert len(prereg.bars_fingerprint()) == 64


class TestObservedInputsAreInternallyConsistent:
    def test_event_ticker_count_le_all_purchase_ticker_count(self):
        o = prereg.OBSERVED
        assert o["n_distinct_tickers_in_events"] <= o["n_distinct_tickers_all_purchases"]

    def test_events_in_current_universe_is_tiny(self):
        # The whole point of this candidate: the current 25-name mega-cap
        # book essentially never produces a cluster event.
        o = prereg.OBSERVED
        assert o["events_in_current_25_name_universe"] / o["n_cluster_events_total"] < 0.01


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
    def test_shape_and_monotonicity(self):
        report = prereg.power_report()
        rows = report["rows"]
        assert len(rows) == 2 * 2 * 2  # 2 holds x 2 effect sizes x 2 bet-rate scenarios

        # Higher bet-rate scenario must never need MORE available bets to
        # be adequately powered than a lower one, for the same (hold, effect).
        by_key = {(r["hold"], r["effect"], r["bet_rate_scenario"]): r for r in rows}
        for hold in ("3mo", "6mo"):
            for effect in (f"original_cmp_{hold}", f"decayed_replication_{hold}"):
                cons = by_key[(hold, effect, "conservative")]
                cent = by_key[(hold, effect, "central")]
                assert cent["n_available_2008_2026"] >= cons["n_available_2008_2026"]
                # n_required_80pct_power depends only on (mu, sigma), not the bet rate.
                assert cons["n_required_80pct_power"] == cent["n_required_80pct_power"]

    def test_original_cmp_effect_is_easier_to_detect_than_decayed(self):
        report = prereg.power_report()
        by_key = {(r["hold"], r["effect"]): r for r in report["rows"] if r["bet_rate_scenario"] == "central"}
        for hold in ("3mo", "6mo"):
            original = by_key[(hold, f"original_cmp_{hold}")]
            decayed = by_key[(hold, f"decayed_replication_{hold}")]
            assert original["n_required_80pct_power"] < decayed["n_required_80pct_power"]


class TestRequiredN:
    def test_larger_effect_needs_fewer_bets(self):
        n_small_effect = prereg.required_n(mu_bps=50, sigma_bps=2000, alpha_one_sided=0.05, power=0.8)
        n_large_effect = prereg.required_n(mu_bps=200, sigma_bps=2000, alpha_one_sided=0.05, power=0.8)
        assert n_large_effect < n_small_effect
