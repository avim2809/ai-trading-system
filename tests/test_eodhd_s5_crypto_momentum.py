"""Tests for the DRAFT S5 crypto cross-sectional momentum pre-registration
(scripts/eodhd_s5_crypto_momentum_preregistered_bars.py).

Pure mechanics only -- no network, no real crypto series, no return
computation. This candidate is still DRAFT (see module docstring): these
tests check the *design*'s internal consistency (fingerprint determinism,
declared variant count, universe-exclusion/ticker-collision helpers, tercile
counting, Bonferroni divisor, tier-classification precedence, and the power
analysis arithmetic), not any backtest outcome.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_s5_crypto_momentum_preregistered_bars as s5  # noqa: E402


class TestDraftStatus:
    def test_is_draft_not_frozen(self):
        assert s5.DRAFT is False
        assert s5.PREREGISTERED_AT == "2026-09-30T19:18:54Z"

    def test_cleaning_fingerprint_matches_v2(self):
        # Pinned to the amendment-1 value the coordinator confirmed
        # (docs/eodhd_shortlist_protocol_2026_10.md); a change here without an
        # intentional protocol amendment is a regression, not a refresh.
        assert s5.DATA["cleaning_fingerprint"] == (
            "fc0690f087edaac8c11ddf77b381e58b59c57a893d460f12c7fa782229693054"
        )


class TestFingerprint:
    def test_deterministic(self):
        assert s5.bars_fingerprint() == s5.bars_fingerprint()
        assert len(s5.bars_fingerprint()) == 64

    def test_changes_if_a_design_dict_changes(self, monkeypatch):
        before = s5.bars_fingerprint()
        monkeypatch.setitem(s5.COSTS, "crypto_bps_per_side", 999.0)
        after = s5.bars_fingerprint()
        assert before != after


class TestVariants:
    def test_n_variants_within_protocol_cap(self):
        assert s5.N_VARIANTS == len(s5.VARIANTS)
        assert s5.N_VARIANTS <= 4

    def test_exactly_one_primary_variant(self):
        primaries = [v for v in s5.VARIANTS.values() if v["primary"]]
        assert len(primaries) == 1
        assert primaries[0]["n_universe"] == 20
        assert primaries[0]["absolute_filter"] is True

    def test_variant_grid_covers_universe_size_and_filter_toggle(self):
        sizes = {v["n_universe"] for v in s5.VARIANTS.values()}
        filters = {v["absolute_filter"] for v in s5.VARIANTS.values()}
        assert sizes == {20, 30}
        assert filters == {True, False}


class TestUniverseExclusion:
    @pytest.mark.parametrize("code", [
        "USDT-USD", "USDC-USD", "DAI-USD", "WBTC-USD", "WETH-USD", "STETH-USD",
        "XAUT-USD", "PAXG-USD", "XRPBULL-USD", "XRPDOWN-USD",
    ])
    def test_known_exclusions(self, code):
        assert s5.is_excluded(code)

    @pytest.mark.parametrize("code", ["BTC-USD", "ETH-USD", "SOL-USD", "LUNA-USD", "MIR-USD"])
    def test_real_coins_not_excluded(self, code):
        assert not s5.is_excluded(code)

    def test_ticker_collision_suffix_normalises_to_same_base(self):
        # Observed at the gate: TAO-USD / TAO22974-USD, CBBTC32994-USD -> CBBTC.
        assert s5.base_symbol("TAO-USD") == s5.base_symbol("TAO22974-USD") == "TAO"
        assert s5.base_symbol("CBBTC32994-USD") == "CBBTC"
        assert s5.is_excluded("CBBTC32994-USD")  # inherits CBBTC's exclusion via the suffix rule

    def test_short_numeric_suffix_is_not_stripped(self):
        # A 1-2 digit trailing run is not a collision suffix (the base_symbol
        # regex threshold is >=3 digits) -- guards against false-positive
        # stripping on a short, otherwise-real ticker.
        assert s5.base_symbol("X2-USD") == "X2"


class TestLiquidityFloorAndBiases:
    def test_liquidity_floor_is_one_million_usd(self):
        # Required per coordinator review 2026-10-01: implementability-based
        # floor, not a tuned threshold.
        assert s5.UNIVERSE["liquidity_floor_usd"] == pytest.approx(1_000_000.0)

    def test_biases_dict_declares_all_four_required_items(self):
        for key in ("residual_survivorship", "wash_trading_volume",
                    "ticker_collision_bare", "segment_break_adjacent"):
            assert key in s5.BIASES
            assert isinstance(s5.BIASES[key], str) and len(s5.BIASES[key]) > 20

    def test_survivorship_bias_direction_is_named_upward(self):
        assert "UPWARD" in s5.BIASES["residual_survivorship"]

    def test_wash_trading_bias_names_volume_inflation(self):
        assert "inflates" in s5.BIASES["wash_trading_volume"]

    def test_segment_break_adjacent_matches_lookback_days(self):
        # The 28-day post-break exclusion is meant to line up exactly with
        # SIGNAL['lookback_days'], not be an independently chosen number.
        assert str(s5.SIGNAL["lookback_days"]) in s5.BIASES["segment_break_adjacent"]

    def test_fingerprint_changes_if_biases_change(self, monkeypatch):
        before = s5.bars_fingerprint()
        monkeypatch.setitem(s5.BIASES, "residual_survivorship", "changed")
        after = s5.bars_fingerprint()
        assert before != after


class TestTercileCount:
    @pytest.mark.parametrize("n,expected", [(20, 7), (30, 10), (3, 1), (1, 0), (2, 1)])
    def test_round_half_up(self, n, expected):
        assert s5.tercile_count(n) == expected


class TestBootstrapAndDSR:
    def test_alpha_is_shortlist_wide_bonferroni(self):
        assert s5.BOOTSTRAP["alpha_one_sided"] == pytest.approx(0.05 / 5)

    def test_dsr_trials_equals_declared_variant_count(self):
        assert s5.DSR["trials"] == s5.N_VARIANTS

    def test_dsr_prior_trials_is_the_task_placeholder_not_a_guessed_int(self):
        assert s5.DSR["prior_trials"] == 207


class TestClassifyPrecedence:
    @pytest.mark.parametrize("a_ok,d,b_ok,want", [
        (True, False, True, "A"),
        (True, True, True, "A"),     # A takes precedence over everything
        (False, True, True, "D"),    # measurably worse beats "safer"
        (False, False, True, "B"),
        (False, False, False, "C"),
    ])
    def test_precedence(self, a_ok, d, b_ok, want):
        bars = {"A1": a_ok, "A2": True}
        tier_b = {"B_a": b_ok, "B_b": True}
        assert s5.classify(bars, d, tier_b) == want


class TestPower:
    def test_required_blocks_decreases_with_larger_sharpe_gap(self):
        n_small_gap = s5.required_blocks(0.15, 0.01, 0.80, 91)
        n_large_gap = s5.required_blocks(0.5, 0.01, 0.80, 91)
        assert n_small_gap > n_large_gap > 0

    def test_required_blocks_matches_hand_solved_value(self):
        # z_{0.99} ~ 2.3263, z_{0.80} ~ 0.8416 -> (z_a+z_b) ~ 3.1679
        # N = (3.1679 / 0.5)^2 * (365/91) ~ 161.0
        n = s5.required_blocks(0.5, 0.01, 0.80, 91)
        assert n == pytest.approx(161.0, abs=0.5)

    def test_report_flags_both_scenarios_underpowered_on_available_history(self):
        # Honest power finding this design must surface, not hide: even the
        # optimistic literature-anchored Sharpe gap (0.5, borrowed from the
        # live C1 rule's own prior) is underpowered on the ~12.7-year window,
        # and the illustrative post-2020-decay scenario is far more so.
        rep = s5.power_report()
        assert rep["rows"], "power_report produced no rows"
        assert all(not row["adequately_powered"] for row in rep["rows"])

    def test_power_report_windows_are_smaller_post_2020(self):
        rep = s5.power_report()
        by_scenario_window = {(r["scenario"], r["window"]): r["n_blocks_available"] for r in rep["rows"]}
        for scenario in s5.POWER["scenarios"]:
            assert (by_scenario_window[(scenario, "post_2020_decay_window")]
                    < by_scenario_window[(scenario, "full_window")])


class TestWindowsAndImplementability:
    def test_window_start_precedes_midpoint_precedes_end(self):
        import pandas as pd
        start = pd.Timestamp(s5.WINDOWS["start"])
        mid = pd.Timestamp(s5.WINDOWS["midpoint"])
        end = pd.Timestamp(s5.WINDOWS["end"])
        assert start < mid < end

    def test_implementability_is_marked_unverified_and_not_a_bar(self):
        assert "UNVERIFIED" in s5.IMPLEMENTABILITY["status"]
        assert "not a bar" in s5.IMPLEMENTABILITY["caveat"]
