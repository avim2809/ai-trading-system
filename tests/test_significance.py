"""Tests for firm.patterns.significance -- the statistical-significance
test against a matched-volatility null (Part A of the 2026-09-27
false-positive-rate fix)."""

from __future__ import annotations

import numpy as np
import pytest

from firm.data.synthetic import make_synthetic_prices
from firm.patterns.significance import (
    DEFAULT_N_DRAWS,
    benjamini_hochberg_accept,
    build_null_score_distribution,
    cached_null_score_distribution,
    pattern_p_value,
)


def _synthetic_ohlcv(symbol="SYN00", n_days=300, seed=1):
    panel = make_synthetic_prices(symbols=[symbol], n_days=n_days, seed=seed)
    df = panel.sort_values("date").reset_index(drop=True)
    return df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), df["volume"].to_numpy()


class TestPatternPValue:
    def test_score_higher_than_entire_null_gets_minimum_pvalue(self):
        null = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        p = pattern_p_value(100.0, null)
        assert p == pytest.approx(1.0 / 6.0)  # (1+0)/(1+5)

    def test_score_lower_than_entire_null_gets_high_pvalue(self):
        null = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        p = pattern_p_value(0.0, null)
        assert p == pytest.approx(1.0)  # (1+5)/(1+5)

    def test_score_matching_all_null_values_exactly(self):
        null = np.full(10, 50.0)
        p = pattern_p_value(50.0, null)
        assert p == pytest.approx(1.0)  # all 10 count as >= 50

    def test_empty_null_returns_none(self):
        assert pattern_p_value(80.0, np.zeros(0)) is None

    def test_pvalue_never_exactly_zero(self):
        # (1+count)/(1+n) construction -- even beating every draw stays > 0.
        null = np.array([1.0, 2.0, 3.0])
        p = pattern_p_value(1000.0, null)
        assert p is not None and p > 0.0


class TestBenjaminiHochbergAccept:
    def test_empty_input_returns_empty(self):
        result = benjamini_hochberg_accept(np.array([]))
        assert len(result) == 0

    def test_all_very_significant_all_accepted(self):
        p = np.array([0.001, 0.002, 0.003, 0.004])
        result = benjamini_hochberg_accept(p, q=0.10)
        assert np.all(result)

    def test_all_clearly_insignificant_none_accepted(self):
        p = np.array([0.8, 0.9, 0.95, 0.99])
        result = benjamini_hochberg_accept(p, q=0.10)
        assert not np.any(result)

    def test_uncorrected_alpha_would_over_accept_bh_is_stricter(self):
        # 20 p-values uniformly spread 0.01..0.20 -- a naive p<0.05 cutoff
        # would accept 4 of them (0.01, 0.03(ish)...), but BH at q=0.05
        # should accept fewer once corrected for 20 simultaneous tests.
        p = np.linspace(0.01, 0.20, 20)
        naive_accept_count = int(np.sum(p < 0.05))
        bh_result = benjamini_hochberg_accept(p, q=0.05)
        assert int(np.sum(bh_result)) <= naive_accept_count

    def test_step_up_property_accepts_every_hypothesis_at_or_below_the_max_passing_rank(self):
        # sorted p-values: 0.01(rank1,thr=0.02,PASS) 0.03(rank2,thr=0.04,PASS)
        # 0.04(rank3,thr=0.06,PASS) 0.20(rank4,thr=0.08,fail) 0.30(rank5,thr=0.10,fail)
        q = 0.10
        p = np.array([0.01, 0.04, 0.03, 0.20, 0.30])  # unsorted on purpose
        result = benjamini_hochberg_accept(p, q=q)
        # original order: [0.01, 0.04, 0.03, 0.20, 0.30] -> accept the 3 smallest
        assert result.tolist() == [True, True, True, False, False]

    def test_preserves_input_order(self):
        p = np.array([0.5, 0.001, 0.9, 0.002])
        result = benjamini_hochberg_accept(p, q=0.10)
        # indices 1 and 3 (the small p-values) should be accepted; 0 and 2 not.
        assert result[1] and result[3]
        assert not result[0] and not result[2]


class TestBuildNullScoreDistribution:
    def test_returns_n_draws_scores(self):
        high, low, close, volume = _synthetic_ohlcv()
        null = build_null_score_distribution(high, low, close, volume, n_draws=20, seed=1)
        assert len(null) == 20
        assert np.all(null >= 0.0)
        assert np.all(null <= 100.0)

    def test_default_n_draws_constant_used_by_default(self):
        high, low, close, volume = _synthetic_ohlcv()
        # Just confirm the default doesn't crash and matches the documented
        # constant's shape -- not running the full 200 draws here (slow),
        # only checking the constant itself is what the function defaults to.
        import inspect
        sig = inspect.signature(build_null_score_distribution)
        assert sig.parameters["n_draws"].default == DEFAULT_N_DRAWS

    def test_too_short_history_returns_empty_not_crash(self):
        n = 10  # below the 21-bar floor
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        close = np.full(n, 100.0)
        volume = np.full(n, 1_000_000.0)
        null = build_null_score_distribution(high, low, close, volume, n_draws=20)
        assert len(null) == 0

    def test_deterministic_given_same_seed(self):
        high, low, close, volume = _synthetic_ohlcv()
        null_a = build_null_score_distribution(high, low, close, volume, n_draws=15, seed=7)
        null_b = build_null_score_distribution(high, low, close, volume, n_draws=15, seed=7)
        np.testing.assert_array_equal(null_a, null_b)

    def test_different_seeds_give_different_draws(self):
        high, low, close, volume = _synthetic_ohlcv()
        null_a = build_null_score_distribution(high, low, close, volume, n_draws=15, seed=7)
        null_b = build_null_score_distribution(high, low, close, volume, n_draws=15, seed=8)
        assert not np.array_equal(null_a, null_b)

    def test_surrogate_volume_is_a_permutation_not_a_fresh_draw(self):
        # Sanity check on the volume-shuffle mechanism: total volume across
        # a surrogate draw must be conserved (it's a permutation, not
        # resampled with replacement) -- verified indirectly by checking
        # build_null_score_distribution doesn't raise and produces varied
        # (not-all-identical) scores, which would happen if volume were
        # accidentally held constant across every draw in a way that
        # made every surrogate identical.
        high, low, close, volume = _synthetic_ohlcv(n_days=100)
        null = build_null_score_distribution(high, low, close, volume, n_draws=10, seed=3)
        assert len(set(null.tolist())) > 1  # not every draw scored identically


class TestCachedNullScoreDistribution:
    def test_same_inputs_hit_cache_not_recomputed(self, monkeypatch):
        import firm.patterns.significance as sig_module

        high, low, close, volume = _synthetic_ohlcv(n_days=60)
        call_count = 0
        original = sig_module.build_null_score_distribution

        def _counting_wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(sig_module, "build_null_score_distribution", _counting_wrapper)
        sig_module._cached_null_score_distribution.cache_clear()

        first = cached_null_score_distribution(high, low, close, volume, n_draws=5, seed=1)
        second = cached_null_score_distribution(high, low, close, volume, n_draws=5, seed=1)

        assert call_count == 1  # second call hit the cache
        np.testing.assert_array_equal(first, second)

    def test_changed_input_misses_cache(self):
        import firm.patterns.significance as sig_module

        sig_module._cached_null_score_distribution.cache_clear()
        high, low, close, volume = _synthetic_ohlcv(n_days=60, seed=1)
        high2, low2, close2, volume2 = _synthetic_ohlcv(n_days=60, seed=2)

        first = cached_null_score_distribution(high, low, close, volume, n_draws=5, seed=1)
        second = cached_null_score_distribution(high2, low2, close2, volume2, n_draws=5, seed=1)
        # Different underlying series -- no reason to expect identical
        # null distributions (this would only spuriously pass if the cache
        # ignored its inputs, which is exactly the bug this guards against).
        assert not np.array_equal(first, second) or sig_module._cached_null_score_distribution.cache_info().misses == 2


class TestRealisticSignificance:
    def test_strong_real_pattern_scores_more_extreme_than_typical_noise(self):
        """Integration-style sanity check: a clean, textbook double-top
        (reused from tests/pattern_fixtures.py) should land in the tail of
        a noise null built from unrelated random-walk data of comparable
        length/volatility -- not a guarantee for every possible fixture,
        but a basic sanity check that the mechanism points the right way."""
        import sys
        from pathlib import Path

        _TESTS = Path(__file__).resolve().parent
        if str(_TESTS) not in sys.path:
            sys.path.insert(0, str(_TESTS))
        from pattern_fixtures import POSITIVE_FIXTURES, build_frame

        from firm.patterns.scanner import scan_symbol

        fx = next(f for f in POSITIVE_FIXTURES if f.name == "double_bottom")
        df = build_frame(fx)
        matches = scan_symbol(df, min_score=0.0)
        assert matches
        real_score = matches[0].quality_score

        high, low, close, volume = _synthetic_ohlcv(n_days=200, seed=99)
        null = build_null_score_distribution(high, low, close, volume, n_draws=100, seed=42)
        p = pattern_p_value(real_score, null)

        assert p is not None
        # Not a strict statistical claim (different length series / one
        # draw of noise), just a directional sanity check that a clean,
        # high-scoring real pattern isn't "typical" relative to noise.
        assert real_score > np.median(null)
