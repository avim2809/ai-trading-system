"""Tests for scripts/benchmark_pattern_detectors.py.

Tests the pure report-building logic (run_fixture's grading rule,
build_report's aggregation) against controlled inputs -- not "does the
whole corpus currently pass" (real detector behavior can legitimately
shift as scorer weights/tolerances evolve; that's exactly what this
benchmark exists to surface, not something a green test suite should
mask by asserting today's exact pass/fail split is permanent). Imports
the script module directly by adding scripts/ to sys.path, mirroring
tests/test_pattern_recognition_rollout_gate.py (scripts/ is not a
package).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from benchmark_pattern_detectors import apply_significance_and_fdr, build_report, run_fixture  # noqa: E402

from pattern_fixtures import (  # noqa: E402
    BOUNDARY_FIXTURES,
    NEGATIVE_FIXTURES,
    POSITIVE_FIXTURES,
    build_frame,
)


class TestRunFixture:
    def test_positive_fixture_graded_correct_when_detected(self):
        fx = next(f for f in POSITIVE_FIXTURES if f.name == "double_top")
        result = run_fixture(fx)
        assert result["should_detect_something"] is True
        assert result["did_detect_something"] is True
        assert result["correct"] is True
        assert result["detected_pattern"] == "double_top"

    def test_negative_fixture_graded_correct_when_nothing_detected(self):
        fx = NEGATIVE_FIXTURES[0]
        result = run_fixture(fx)
        assert result["should_detect_something"] is False
        assert result["did_detect_something"] is False
        assert result["correct"] is True

    def test_boundary_fixture_graded_correct_when_nothing_detected(self):
        fx = BOUNDARY_FIXTURES[0]
        result = run_fixture(fx)
        assert result["correct"] is True

    def test_higher_min_score_can_turn_a_hit_into_a_miss(self):
        fx = next(f for f in POSITIVE_FIXTURES if f.name == "rising_wedge")
        lenient = run_fixture(fx, min_score=0.0)
        strict = run_fixture(fx, min_score=99.9)
        assert lenient["did_detect_something"] is True
        assert strict["did_detect_something"] is False

    def test_wrong_pattern_name_graded_incorrect(self):
        from pattern_fixtures import PatternFixture

        fx = next(f for f in POSITIVE_FIXTURES if f.name == "double_top")
        wrong = PatternFixture(fx.name, fx.category, fx.anchors, fx.total_bars, fx.spike_at, frozenset({"cup_handle"}))
        result = run_fixture(wrong)
        assert result["did_detect_something"] is True  # a pattern WAS found...
        assert result["correct"] is False  # ...just not the one expected


class TestBuildReport:
    def _fake_results(self):
        return [
            {"name": "p1", "category": "positive", "family": "reversal", "expected_patterns": ["double_top"],
             "detected_pattern": "double_top", "detected_quality_fraction": 0.9,
             "should_detect_something": True, "did_detect_something": True, "correct": True},
            {"name": "p2", "category": "positive", "family": "triangle", "expected_patterns": ["rectangle"],
             "detected_pattern": None, "detected_quality_fraction": 0.0,
             "should_detect_something": True, "did_detect_something": False, "correct": False},
            {"name": "n1", "category": "negative", "family": None, "expected_patterns": [],
             "detected_pattern": None, "detected_quality_fraction": 0.0,
             "should_detect_something": False, "did_detect_something": False, "correct": True},
            {"name": "n2", "category": "negative", "family": None, "expected_patterns": [],
             "detected_pattern": "triple_top", "detected_quality_fraction": 0.7,
             "should_detect_something": False, "did_detect_something": True, "correct": False},
        ]

    def test_overall_accuracy(self):
        report = build_report(self._fake_results())
        assert report["overall"]["n"] == 4
        assert report["overall"]["n_correct"] == 2
        assert report["overall"]["accuracy"] == 0.5

    def test_by_category_breakdown(self):
        report = build_report(self._fake_results())
        assert report["by_category"]["positive"]["n"] == 2
        assert report["by_category"]["positive"]["n_correct"] == 1
        assert report["by_category"]["negative"]["n"] == 2
        assert report["by_category"]["negative"]["n_correct"] == 1

    def test_classification_report_present(self):
        report = build_report(self._fake_results())
        c = report["classification"]
        # y_true=[1,1,0,0], y_pred=[1,0,0,1] -> TP=1, FP=1, FN=1
        assert c["precision"] == 0.5
        assert c["recall"] == 0.5

    def test_empty_results_does_not_raise(self):
        report = build_report([])
        assert report["overall"]["n"] == 0
        assert report["overall"]["accuracy"] is None


class TestApplySignificanceAndFdr:
    def _fake_result(self, name, *, should_detect, p_value, detected="double_top"):
        return {
            "name": name, "category": "positive" if should_detect else "negative",
            "family": "reversal", "expected_patterns": [detected] if should_detect else [],
            "detected_pattern": detected, "detected_quality_fraction": 0.8,
            "should_detect_something": should_detect, "did_detect_something": True,
            "correct": should_detect, "p_value": p_value,
        }

    def test_significant_result_survives(self):
        results = [self._fake_result("a", should_detect=True, p_value=0.001)]
        adjusted = apply_significance_and_fdr(results, q=0.05)
        assert adjusted[0]["did_detect_something"] is True
        assert adjusted[0]["correct"] is True

    def test_insignificant_result_is_rejected(self):
        results = [self._fake_result("a", should_detect=False, p_value=0.9)]
        adjusted = apply_significance_and_fdr(results, q=0.05)
        assert adjusted[0]["did_detect_something"] is False
        assert adjusted[0]["detected_pattern"] is None
        assert adjusted[0]["correct"] is True  # correctly rejected a false positive

    def test_results_with_no_pvalue_pass_through_unchanged(self):
        result = self._fake_result("a", should_detect=False, p_value=None)
        result["did_detect_something"] = False
        result["correct"] = True
        adjusted = apply_significance_and_fdr([result], q=0.05)
        assert adjusted[0] == result

    def test_empty_results_does_not_raise(self):
        assert apply_significance_and_fdr([], q=0.05) == []

    def test_does_not_mutate_input(self):
        results = [self._fake_result("a", should_detect=False, p_value=0.9)]
        original = dict(results[0])
        apply_significance_and_fdr(results, q=0.05)
        assert results[0] == original


class TestFullFixtureBuildFrame:
    def test_build_frame_matches_run_fixture_result(self):
        fx = next(f for f in POSITIVE_FIXTURES if f.name == "ascending_triangle")
        df = build_frame(fx)
        assert list(df.columns) == ["high", "low", "close", "volume"]
        assert len(df) == fx.total_bars
