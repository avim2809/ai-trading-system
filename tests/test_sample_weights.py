"""Tests for firm.patterns.ml.sample_weights -- de Prado ch.4 average-
uniqueness weighting for overlapping triple-barrier-labeled events (Part B
item 3, 2026-09-27)."""

from __future__ import annotations

import numpy as np
import pytest

from firm.patterns.ml.sample_weights import average_uniqueness, sample_weights_by_group


class TestAverageUniqueness:
    def test_empty_input_returns_empty_array(self):
        assert average_uniqueness([]).tolist() == []

    def test_single_event_is_fully_unique(self):
        weights = average_uniqueness([(0, 5)])
        assert weights.tolist() == [1.0]

    def test_two_completely_non_overlapping_events_are_both_fully_unique(self):
        weights = average_uniqueness([(0, 4), (10, 14)])
        assert weights.tolist() == pytest.approx([1.0, 1.0])

    def test_two_fully_overlapping_identical_spans_get_half_weight_each(self):
        # Both events occupy exactly [0, 4] -> concurrency=2 at every bar
        # either touches -> mean(1/2) = 0.5 for both.
        weights = average_uniqueness([(0, 4), (0, 4)])
        assert weights.tolist() == pytest.approx([0.5, 0.5])

    def test_partial_overlap_is_weighted_between_the_two_extremes(self):
        # Event A: [0, 9]. Event B: [5, 14] -- overlap on bars 5..9 (5 bars
        # out of A's 10 and B's 10), concurrency=2 there, =1 elsewhere.
        weights = average_uniqueness([(0, 9), (5, 14)])
        # A: 5 bars at concurrency 1 (1/1=1.0) + 5 bars at concurrency 2
        # (1/2=0.5) -> mean = (5*1.0 + 5*0.5) / 10 = 0.75
        assert weights[0] == pytest.approx(0.75)
        assert weights[1] == pytest.approx(0.75)  # symmetric overlap

    def test_zero_length_spans_are_valid(self):
        # Two single-bar events on the SAME bar -> concurrency 2 there.
        weights = average_uniqueness([(3, 3), (3, 3)])
        assert weights.tolist() == pytest.approx([0.5, 0.5])

    def test_zero_length_span_with_no_neighbors_is_fully_unique(self):
        weights = average_uniqueness([(3, 3), (100, 100)])
        assert weights.tolist() == pytest.approx([1.0, 1.0])

    def test_heavy_cluster_of_overlapping_events_all_get_low_weight(self):
        # 5 events all spanning the exact same window -> each is 1/5 unique.
        spans = [(0, 9)] * 5
        weights = average_uniqueness(spans)
        assert weights.tolist() == pytest.approx([0.2] * 5)

    def test_weights_are_always_in_unit_interval(self):
        rng = np.random.RandomState(0)
        spans = []
        for _ in range(50):
            t0 = int(rng.randint(0, 100))
            t1 = t0 + int(rng.randint(0, 20))
            spans.append((t0, t1))
        weights = average_uniqueness(spans)
        assert np.all((weights > 0.0) & (weights <= 1.0))

    def test_order_of_input_spans_does_not_affect_each_events_own_weight(self):
        spans = [(0, 9), (5, 14), (20, 25)]
        weights = average_uniqueness(spans)
        reordered = [spans[2], spans[0], spans[1]]
        reordered_weights = average_uniqueness(reordered)
        assert reordered_weights[0] == pytest.approx(weights[2])
        assert reordered_weights[1] == pytest.approx(weights[0])
        assert reordered_weights[2] == pytest.approx(weights[1])


class TestSampleWeightsByGroup:
    def test_empty_input_returns_empty_array(self):
        assert sample_weights_by_group([], []).tolist() == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            sample_weights_by_group(["AAPL"], [(0, 5), (10, 15)])

    def test_groups_are_weighted_independently(self):
        # AAPL's two events fully overlap (weight 0.5 each); MSFT's single
        # event is unaffected by AAPL's overlap despite sharing bar indices.
        groups = ["AAPL", "AAPL", "MSFT"]
        spans = [(0, 9), (0, 9), (0, 9)]
        weights = sample_weights_by_group(groups, spans)
        assert weights[0] == pytest.approx(0.5)
        assert weights[1] == pytest.approx(0.5)
        assert weights[2] == pytest.approx(1.0)

    def test_preserves_input_row_order_regardless_of_group_interleaving(self):
        groups = ["MSFT", "AAPL", "MSFT", "AAPL"]
        spans = [(0, 4), (0, 9), (0, 4), (0, 9)]
        weights = sample_weights_by_group(groups, spans)
        # MSFT rows (indices 0, 2) fully overlap each other -> 0.5 each;
        # AAPL rows (indices 1, 3) fully overlap each other -> 0.5 each.
        assert weights.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])

    def test_single_row_per_group_is_fully_unique(self):
        groups = ["AAPL", "MSFT", "NVDA"]
        spans = [(0, 5), (100, 105), (3, 8)]
        weights = sample_weights_by_group(groups, spans)
        assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0])
