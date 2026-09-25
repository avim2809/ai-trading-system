"""Tests for firm.patterns.scorer.

Covers the original five-component behaviour (still the default when the
two new optional-arg-driven components aren't supplied) plus the new
``breakout_distance`` / ``pre_breakout_compression`` components: saturation
bounds, graceful degradation when their optional inputs are omitted or
unusable, and that ``PatternScore.total`` / ``.as_dict()`` reflect them.
"""

from __future__ import annotations

import numpy as np
import pytest

from firm.patterns.scorer import PatternScore, score_pattern

# ---------------------------------------------------------------------------
# Backward compatibility: existing 5-kwarg call sites (e.g. scanner.py today)
# ---------------------------------------------------------------------------


def test_score_pattern_without_new_kwargs_never_crashes():
    """Every existing caller of score_pattern (just scanner.py today) only
    ever passes the original 5 kwargs. This must keep working unchanged.
    """
    score = score_pattern(
        geometry_tolerance_used=0.8,
        fit_quality=0.7,
        volume_ratio=1.6,
        duration_bars=30,
        follow_through_atr=1.0,
    )
    assert isinstance(score, PatternScore)
    assert score.breakout_distance == 0.0
    assert score.pre_breakout_compression == 0.0


def test_score_pattern_rewards_clean_setup_without_new_components():
    """Mirrors the original (pre-rebalance) 'clean setup' case: all 5
    original inputs maxed. With the two new components omitted (zero
    contribution, by contract), the new effective ceiling for a caller that
    hasn't upgraded is 85, not 100 -- the documented weight-rebalance
    trade-off (see scorer.py's module docstring).
    """
    score = score_pattern(
        geometry_tolerance_used=1.0,
        fit_quality=1.0,
        volume_ratio=2.0,
        duration_bars=40,
        follow_through_atr=3.0,
    )
    assert score.total == pytest.approx(85.0, abs=0.01)
    assert score.geometry == pytest.approx(30.0)
    assert score.trendline_fit == pytest.approx(15.0)
    assert score.volume_confirmation == pytest.approx(20.0)
    assert score.duration == pytest.approx(10.0)
    assert score.follow_through == pytest.approx(10.0)


def test_score_pattern_penalizes_weak_setup_without_new_components():
    score = score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,  # at the 20d average -- no confirmation at all
        duration_bars=1,  # implausibly short
        follow_through_atr=0.0,  # barely crossed the level
    )
    assert score.total == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Fully saturated 7-component score sums to a clean 100
# ---------------------------------------------------------------------------


def test_score_pattern_fully_saturated_all_seven_components_sums_to_100():
    # confirm_index = 25; pre_breakout_compression looks at idx = 24 and its
    # trailing 20-bar window [5:25); breakout_distance looks at atr[25]
    # directly. Kept independent of each other by construction.
    atr_series = np.full(30, 10.0)
    atr_series[24] = 6.0  # ratio = 6 / mean([...19x10.0, 6.0]) well below 0.8
    atr_series[25] = 2.0  # ATR at the confirmation bar itself

    score = score_pattern(
        geometry_tolerance_used=1.0,
        fit_quality=1.0,
        volume_ratio=2.0,
        duration_bars=40,
        follow_through_atr=3.0,
        close_at_confirm=102.0,
        level_at_confirm=100.0,  # distance = 2.0 -> 2.0/2.0 = 1.0x ATR -> saturates
        atr_series=atr_series,
        confirm_index=25,
    )
    assert score.breakout_distance == pytest.approx(10.0)
    assert score.pre_breakout_compression == pytest.approx(5.0)
    assert score.total == pytest.approx(100.0, abs=0.01)
    assert score.as_dict()["total"] == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# breakout_distance
# ---------------------------------------------------------------------------


def _score_with_breakout_distance(close_at_confirm, level_at_confirm, atr_at_confirm, confirm_index=10):
    atr_series = np.full(confirm_index + 1, 5.0)
    atr_series[confirm_index] = atr_at_confirm
    return score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,
        duration_bars=1,
        follow_through_atr=0.0,
        close_at_confirm=close_at_confirm,
        level_at_confirm=level_at_confirm,
        atr_series=atr_series,
        confirm_index=confirm_index,
    )


def test_breakout_distance_saturates_at_one_atr():
    # distance = 2.0, atr = 2.0 -> exactly 1.0x ATR -> full 10 pts.
    score = _score_with_breakout_distance(close_at_confirm=102.0, level_at_confirm=100.0, atr_at_confirm=2.0)
    assert score.breakout_distance == pytest.approx(10.0)


def test_breakout_distance_beyond_saturation_stays_clipped_at_10():
    # distance = 4.0, atr = 2.0 -> 2.0x ATR, well past saturation -> still 10.
    score = _score_with_breakout_distance(close_at_confirm=104.0, level_at_confirm=100.0, atr_at_confirm=2.0)
    assert score.breakout_distance == pytest.approx(10.0)


def test_breakout_distance_half_saturation_gives_half_credit():
    # distance = 1.0, atr = 2.0 -> 0.5x ATR -> half credit (5 pts).
    score = _score_with_breakout_distance(close_at_confirm=101.0, level_at_confirm=100.0, atr_at_confirm=2.0)
    assert score.breakout_distance == pytest.approx(5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(close_at_confirm=None, level_at_confirm=100.0, atr_series=np.full(5, 1.0), confirm_index=2),
        dict(close_at_confirm=101.0, level_at_confirm=None, atr_series=np.full(5, 1.0), confirm_index=2),
        dict(close_at_confirm=101.0, level_at_confirm=100.0, atr_series=None, confirm_index=2),
        dict(close_at_confirm=101.0, level_at_confirm=100.0, atr_series=np.full(5, 1.0), confirm_index=None),
    ],
)
def test_breakout_distance_missing_any_input_defaults_to_zero(kwargs):
    score = score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,
        duration_bars=1,
        follow_through_atr=0.0,
        **kwargs,
    )
    assert score.breakout_distance == 0.0


def test_breakout_distance_confirm_index_out_of_range_defaults_to_zero():
    score = score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,
        duration_bars=1,
        follow_through_atr=0.0,
        close_at_confirm=110.0,
        level_at_confirm=100.0,
        atr_series=np.full(5, 1.0),
        confirm_index=99,  # way past the end of atr_series
    )
    assert score.breakout_distance == 0.0


def test_breakout_distance_nan_atr_defaults_to_zero():
    atr_series = np.full(5, 1.0)
    atr_series[2] = float("nan")
    score = score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,
        duration_bars=1,
        follow_through_atr=0.0,
        close_at_confirm=110.0,
        level_at_confirm=100.0,
        atr_series=atr_series,
        confirm_index=2,
    )
    assert score.breakout_distance == 0.0


def test_breakout_distance_zero_atr_defaults_to_zero():
    atr_series = np.full(5, 0.0)
    score = score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,
        duration_bars=1,
        follow_through_atr=0.0,
        close_at_confirm=110.0,
        level_at_confirm=100.0,
        atr_series=atr_series,
        confirm_index=2,
    )
    assert score.breakout_distance == 0.0


# ---------------------------------------------------------------------------
# pre_breakout_compression
# ---------------------------------------------------------------------------


def _score_with_compression(atr_series, confirm_index):
    return score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,
        duration_bars=1,
        follow_through_atr=0.0,
        atr_series=atr_series,
        confirm_index=confirm_index,
    )


def test_pre_breakout_compression_full_credit_when_clearly_compressed():
    # idx = confirm_index - 1 = 24; window = atr_series[5:25] (20 bars).
    # 19 bars at 10.0, idx itself (24) = 6.0 -> ratio well below 0.8.
    atr_series = np.full(30, 10.0)
    atr_series[24] = 6.0
    score = _score_with_compression(atr_series, confirm_index=25)
    assert score.pre_breakout_compression == pytest.approx(5.0)


def test_pre_breakout_compression_zero_credit_when_elevated():
    atr_series = np.full(30, 10.0)
    atr_series[24] = 20.0  # ratio = 20 / mean(19x10 + 20) = 20/10.5 ~= 1.905 -> >= 1.2
    score = _score_with_compression(atr_series, confirm_index=25)
    assert score.pre_breakout_compression == 0.0


def test_pre_breakout_compression_linear_interpolation_at_midpoint():
    # Uniform ATR -> ratio == 1.0, exactly midway between 0.8 and 1.2 ->
    # half credit (2.5 of the 5-pt max).
    atr_series = np.full(30, 10.0)
    score = _score_with_compression(atr_series, confirm_index=25)
    assert score.pre_breakout_compression == pytest.approx(2.5)


def test_pre_breakout_compression_exact_full_credit_boundary_ratio():
    # Construct the 20-bar window [5:25) so its mean is *exactly* 10.0 in
    # floating point (sum == 200.0) and the current bar (idx 24) is exactly
    # 8.0 -> ratio == 8.0 / 10.0 == 0.8 exactly (no rounding ambiguity),
    # which the docstring says should still earn full credit ("<=").
    atr_series = np.full(30, 10.0)
    atr_series[23] = 12.0  # offsets idx 24 below so the window sum stays 200.0
    atr_series[24] = 8.0
    assert atr_series[5:25].sum() == pytest.approx(200.0)
    score = _score_with_compression(atr_series, confirm_index=25)
    assert score.pre_breakout_compression == pytest.approx(5.0)


def test_pre_breakout_compression_insufficient_history_defaults_to_zero():
    # idx = confirm_index - 1 = 4; window_start = 4 - 20 + 1 = -15 < 0.
    atr_series = np.full(10, 10.0)
    score = _score_with_compression(atr_series, confirm_index=5)
    assert score.pre_breakout_compression == 0.0


def test_pre_breakout_compression_nan_in_window_defaults_to_zero():
    atr_series = np.full(30, 10.0)
    atr_series[10] = float("nan")  # falls inside the [5:25) window for confirm_index=25
    score = _score_with_compression(atr_series, confirm_index=25)
    assert score.pre_breakout_compression == 0.0


def test_pre_breakout_compression_missing_inputs_default_to_zero():
    score = score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,
        duration_bars=1,
        follow_through_atr=0.0,
    )
    assert score.pre_breakout_compression == 0.0

    score2 = score_pattern(
        geometry_tolerance_used=0.0,
        fit_quality=0.0,
        volume_ratio=1.0,
        duration_bars=1,
        follow_through_atr=0.0,
        atr_series=np.full(30, 10.0),
        confirm_index=None,
    )
    assert score2.pre_breakout_compression == 0.0


def test_pre_breakout_compression_confirm_index_at_start_of_array_defaults_to_zero():
    atr_series = np.full(30, 10.0)
    score = _score_with_compression(atr_series, confirm_index=0)  # idx = -1
    assert score.pre_breakout_compression == 0.0


# ---------------------------------------------------------------------------
# PatternScore.total / .as_dict()
# ---------------------------------------------------------------------------


def test_pattern_score_default_construction_has_zero_new_components():
    score = PatternScore(geometry=1.0, trendline_fit=2.0, volume_confirmation=3.0, duration=4.0, follow_through=5.0)
    assert score.breakout_distance == 0.0
    assert score.pre_breakout_compression == 0.0
    assert score.total == pytest.approx(15.0)


def test_pattern_score_as_dict_includes_new_component_keys():
    score = PatternScore(
        geometry=1.0,
        trendline_fit=2.0,
        volume_confirmation=3.0,
        duration=4.0,
        follow_through=5.0,
        breakout_distance=6.0,
        pre_breakout_compression=7.0,
    )
    d = score.as_dict()
    assert d["breakout_distance"] == 6.0
    assert d["pre_breakout_compression"] == 7.0
    assert d["total"] == pytest.approx(28.0)
    assert d["total"] == pytest.approx(score.total)
