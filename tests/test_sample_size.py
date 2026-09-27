"""Tests for firm.patterns.sample_size -- the per-pattern minimum-sample
confidence discount (Part A of the 2026-09-27 false-positive-rate fix)."""

from __future__ import annotations

import pytest

from firm.patterns.sample_size import (
    DEFAULT_MIN_RELIABLE_SAMPLES,
    confidence_discount,
    load_sample_counts,
    save_sample_counts,
)


class TestConfidenceDiscount:
    def test_no_counts_available_is_a_no_op(self):
        assert confidence_discount("rising_wedge", None) == 1.0

    def test_well_above_threshold_gets_full_confidence(self):
        counts = {"rising_wedge": 1351}
        assert confidence_discount("rising_wedge", counts, min_reliable_samples=30) == 1.0

    def test_exactly_at_threshold_gets_full_confidence(self):
        counts = {"double_top": 30}
        assert confidence_discount("double_top", counts, min_reliable_samples=30) == pytest.approx(1.0)

    def test_below_threshold_scales_proportionally(self):
        counts = {"double_top": 11}
        assert confidence_discount("double_top", counts, min_reliable_samples=30) == pytest.approx(11 / 30)

    def test_single_sample_gets_heavy_discount(self):
        counts = {"cup_handle": 1}
        assert confidence_discount("cup_handle", counts, min_reliable_samples=30) == pytest.approx(1 / 30)

    def test_pattern_absent_from_counts_is_a_true_zero_not_skipped(self):
        counts = {"rising_wedge": 1351}  # "pennant" never seen at all
        assert confidence_discount("pennant", counts, min_reliable_samples=30) == 0.0

    def test_zero_min_reliable_samples_is_a_no_op_not_divide_by_zero(self):
        counts = {"double_top": 11}
        assert confidence_discount("double_top", counts, min_reliable_samples=0) == 1.0

    def test_default_threshold_constant(self):
        assert DEFAULT_MIN_RELIABLE_SAMPLES == 30


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "counts.json"
        counts = {"rising_wedge": 1351, "cup_handle": 1, "double_top": 11}
        save_sample_counts(counts, path)
        loaded = load_sample_counts(path)
        assert loaded == counts

    def test_missing_file_returns_none_not_raise(self, tmp_path):
        assert load_sample_counts(tmp_path / "does_not_exist.json") is None

    def test_corrupt_file_returns_none_not_raise(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json")
        assert load_sample_counts(path) is None

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "counts.json"
        save_sample_counts({"double_top": 5}, path)
        assert path.exists()
