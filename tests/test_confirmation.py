"""Tests for firm.patterns.confirmation.

Covers the pre-existing find_confirmation (a light smoke test -- this
module previously only had indirect coverage via the rule modules in
tests/test_patterns.py) plus the new retest_outcome / retest_score_modifier
building blocks: held/failed/no_retest for both directions, boundary
conditions, and degenerate/empty-array inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from firm.patterns.confirmation import find_confirmation, retest_outcome, retest_score_modifier

# ---------------------------------------------------------------------------
# find_confirmation -- smoke test only (unchanged; thoroughly exercised
# indirectly via tests/test_patterns.py's rule-detector tests)
# ---------------------------------------------------------------------------


def test_find_confirmation_finds_most_recent_cross_above():
    close = np.array([100.0, 100.0, 100.0, 105.0, 95.0, 106.0])
    idx = find_confirmation(close, lambda _i: 100.0, "above", lookback_bars=3, min_index=0)
    assert idx == 5


def test_find_confirmation_returns_negative_one_when_not_found():
    close = np.array([100.0, 99.0, 98.0, 97.0])
    idx = find_confirmation(close, lambda _i: 100.0, "above", lookback_bars=3, min_index=0)
    assert idx == -1


# ---------------------------------------------------------------------------
# retest_outcome -- "above" (bullish breakout) direction
# ---------------------------------------------------------------------------


def _arrays(n: int, base: float = 110.0, spread: float = 1.0):
    close = np.full(n, base)
    high = close + spread
    low = close - spread
    return close, low, high


def test_retest_outcome_above_held():
    n = 20
    close, low, high = _arrays(n)
    confirm_index = 5
    level = 100.0
    low[6] = 99.0  # retest: dips back to/through the level
    close[6] = 99.5
    close[confirm_index + 10] = 105.0  # last bar in the 10-bar window: back above -> held
    outcome = retest_outcome(close, low, high, level, "above", confirm_index, lookback_bars=10)
    assert outcome == "held"


def test_retest_outcome_above_failed():
    n = 20
    close, low, high = _arrays(n)
    confirm_index = 5
    level = 100.0
    low[6] = 99.0  # retest
    close[6] = 99.5
    close[confirm_index + 10] = 95.0  # last bar in window: stayed below -> failed
    outcome = retest_outcome(close, low, high, level, "above", confirm_index, lookback_bars=10)
    assert outcome == "failed"


def test_retest_outcome_above_no_retest():
    n = 20
    close, low, high = _arrays(n, base=110.0, spread=1.0)  # low never dips near 100
    confirm_index = 5
    level = 100.0
    outcome = retest_outcome(close, low, high, level, "above", confirm_index, lookback_bars=10)
    assert outcome == "no_retest"


# ---------------------------------------------------------------------------
# retest_outcome -- "below" (bearish breakdown) direction, mirrored
# ---------------------------------------------------------------------------


def test_retest_outcome_below_held():
    n = 20
    close, low, high = _arrays(n, base=90.0, spread=1.0)
    confirm_index = 5
    level = 100.0
    high[6] = 101.0  # retest: bounces back up to/through the level
    close[6] = 100.5
    close[confirm_index + 10] = 95.0  # last bar: back below -> held
    outcome = retest_outcome(close, low, high, level, "below", confirm_index, lookback_bars=10)
    assert outcome == "held"


def test_retest_outcome_below_failed():
    n = 20
    close, low, high = _arrays(n, base=90.0, spread=1.0)
    confirm_index = 5
    level = 100.0
    high[6] = 101.0  # retest
    close[6] = 100.5
    close[confirm_index + 10] = 105.0  # last bar: stayed above -> failed
    outcome = retest_outcome(close, low, high, level, "below", confirm_index, lookback_bars=10)
    assert outcome == "failed"


def test_retest_outcome_below_no_retest():
    n = 20
    close, low, high = _arrays(n, base=90.0, spread=1.0)  # high never rises near 100
    confirm_index = 5
    level = 100.0
    outcome = retest_outcome(close, low, high, level, "below", confirm_index, lookback_bars=10)
    assert outcome == "no_retest"


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def test_retest_outcome_retest_on_very_next_bar():
    n = 20
    close, low, high = _arrays(n)
    confirm_index = 5
    level = 100.0
    low[6] = 99.0  # very next bar after confirm_index retests
    close[6] = 99.5
    close[confirm_index + 10] = 108.0  # held
    outcome = retest_outcome(close, low, high, level, "above", confirm_index, lookback_bars=10)
    assert outcome == "held"


def test_retest_outcome_retest_exactly_at_lookback_edge_is_included():
    n = 20
    close, low, high = _arrays(n)
    confirm_index = 5
    level = 100.0
    lookback_bars = 3
    edge_index = confirm_index + lookback_bars  # 8 -- last bar still inside the window
    low[edge_index] = 99.0  # retest right at the edge
    close[edge_index] = 101.0  # held
    outcome = retest_outcome(close, low, high, level, "above", confirm_index, lookback_bars=lookback_bars)
    assert outcome == "held"


def test_retest_outcome_retest_one_bar_past_lookback_edge_is_excluded():
    n = 20
    close, low, high = _arrays(n)
    confirm_index = 5
    level = 100.0
    lookback_bars = 3
    past_edge_index = confirm_index + lookback_bars + 1  # 9 -- one bar outside the window
    low[past_edge_index] = 99.0  # would retest, but it's outside the lookback window
    outcome = retest_outcome(close, low, high, level, "above", confirm_index, lookback_bars=lookback_bars)
    assert outcome == "no_retest"


def test_retest_outcome_confirm_index_at_last_bar_has_no_window():
    close, low, high = _arrays(6)
    outcome = retest_outcome(close, low, high, 100.0, "above", confirm_index=5, lookback_bars=10)
    assert outcome == "no_retest"


def test_retest_outcome_confirm_index_negative_returns_no_retest():
    close, low, high = _arrays(10)
    outcome = retest_outcome(close, low, high, 100.0, "above", confirm_index=-1, lookback_bars=10)
    assert outcome == "no_retest"


def test_retest_outcome_confirm_index_beyond_array_returns_no_retest():
    close, low, high = _arrays(10)
    outcome = retest_outcome(close, low, high, 100.0, "above", confirm_index=50, lookback_bars=10)
    assert outcome == "no_retest"


def test_retest_outcome_empty_arrays_returns_no_retest():
    empty = np.array([])
    outcome = retest_outcome(empty, empty, empty, 100.0, "above", confirm_index=0, lookback_bars=10)
    assert outcome == "no_retest"


def test_retest_outcome_lookback_bars_zero_returns_no_retest():
    close, low, high = _arrays(10)
    outcome = retest_outcome(close, low, high, 100.0, "above", confirm_index=3, lookback_bars=0)
    assert outcome == "no_retest"


# ---------------------------------------------------------------------------
# retest_score_modifier
# ---------------------------------------------------------------------------


def test_retest_score_modifier_held_is_positive():
    assert retest_score_modifier("held") == 1.0


def test_retest_score_modifier_failed_is_negative():
    assert retest_score_modifier("failed") == -1.0


def test_retest_score_modifier_no_retest_is_neutral():
    assert retest_score_modifier("no_retest") == 0.0


def test_retest_score_modifier_unrecognized_defaults_to_neutral():
    assert retest_score_modifier("something_unexpected") == 0.0


@pytest.mark.parametrize("direction", ["above", "below"])
def test_retest_outcome_return_value_is_always_a_valid_modifier_input(direction):
    close, low, high = _arrays(20)
    outcome = retest_outcome(close, low, high, 100.0, direction, confirm_index=5, lookback_bars=10)
    assert outcome in ("held", "failed", "no_retest")
    # Must never raise / warn-degrade for retest_outcome's own literal outputs.
    assert retest_score_modifier(outcome) in (1.0, -1.0, 0.0)
