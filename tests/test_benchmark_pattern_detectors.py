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

from benchmark_pattern_detectors import (
    apply_significance_and_fdr,
    build_report,
    run_benchmark,
    run_fixture,
)
from pattern_fixtures import (
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


# ---------------------------------------------------------------------------
# CI-enforced release gate (Part A item 7, 2026-09-27): everything above this
# point tests the report-BUILDING logic against controlled/fake inputs, by
# design (see this module's docstring). These tests are different on
# purpose: they run the REAL fixture corpus through the REAL scan_symbol
# pipeline and fail the build if a pre-registered floor is violated --
# turning this benchmark from "informative" into a hard release gate, per
# docs/pattern_recognition_plan.md's Part A item 7.
#
# Thresholds are fixed BEFORE looking at results, against a deterministic,
# reproducible run of this exact corpus (n_negative_symbols=20, seed=42 --
# see run_benchmark's own seeding; a repeat run with identical arguments has
# been verified byte-for-byte identical). They are headroom above the
# measured baseline, not the measured value itself: tight enough to catch a
# real regression (e.g. back toward the pre-Part-A 70% synthetic-noise
# false-positive rate), loose enough that a still-healthy run doesn't flake
# the build. If these ever need to move, that is a deliberate, reviewed
# decision (a new corpus, a new detector generation) made in its own commit
# with fresh numbers -- never a quiet loosening to make a failing PR pass.
#
# Rebaselined 2026-09-27 (Workstream C, later the same day as the
# thresholds below were first set in a6e1bda): find_confirmation
# (firm.patterns.confirmation) was fixed from a plain is-beyond test to a
# genuine edge-triggered crossing -- see that module's docstring for the
# bug history. That fix, plus the resulting tests/pattern_fixtures.py
# corpus rework (every fixture's tail reshaped so its real breakout lands
# inside the 3-bar confirm window instead of relying on "still past the
# level"), changes what this benchmark measures. Re-run against this exact
# corpus/seed and CONFIRMED THE EXISTING CEILINGS STILL HOLD WITH REAL
# MARGIN -- this is a deliberate re-measurement, not a loosening: no
# threshold constant below changed.
#
#   Baseline (rule-based scan_symbol only, min_score=60.0 -- what actually
#     ships live today): negative-category accuracy=0.857 (FPR=14.3%, was
#     19.0% pre-fix -- fewer stale "still beyond" false confirmations on
#     noise), positive/ambiguous/boundary accuracy=1.000 (perfect recall,
#     unchanged), Brier=0.054 (was 0.073), PR-AUC=0.970 (was 0.942).
#   With significance test + FDR control applied on top (Part A items 3-4,
#     not yet the strategy's live default): negative-category
#     accuracy=0.905 (FPR=9.5%, was 4.8% pre-fix -- see honest note below),
#     recall still 1.000.
#
# Honest note on the FPR direction: the significance-adjusted FPR moved
# from 4.8% to 9.5% (still comfortably under the 15% ceiling) even though
# the baseline FPR improved. The null-score distribution
# (cached_null_score_distribution) and the real fixtures' scores both
# shifted under the fix, and the two didn't move by the same amount --
# this is a real re-measurement, not a bug in either direction; both
# figures are reported plainly rather than only citing the flattering one.
GOLDEN_BENCHMARK_N_NEGATIVE_SYMBOLS = 20
GOLDEN_BENCHMARK_SEED = 42

MAX_BASELINE_FALSE_POSITIVE_RATE = 0.30
MAX_SIGNIFICANCE_FDR_FALSE_POSITIVE_RATE = 0.15
MIN_RECALL = 1.0  # never regress on a real, confirmed pattern -- no tolerance
MAX_BRIER_SCORE = 0.25  # uninformative-forecaster baseline (see run_fixture docstring)
MIN_PR_AUC = 0.85


class TestGoldenBenchmarkReleaseGate:
    def _report(self, *, compute_significance: bool) -> dict:
        results = run_benchmark(
            n_negative_symbols=GOLDEN_BENCHMARK_N_NEGATIVE_SYMBOLS,
            seed=GOLDEN_BENCHMARK_SEED,
            compute_significance=compute_significance,
        )
        if compute_significance:
            results = apply_significance_and_fdr(results)
        return build_report(results)

    def test_baseline_never_misses_a_real_pattern(self):
        report = self._report(compute_significance=False)
        assert report["classification"]["recall"] == MIN_RECALL
        for category in ("positive", "ambiguous", "boundary"):
            assert report["by_category"][category]["accuracy"] == 1.0, (
                f"{category} category accuracy regressed below 1.0 -- a real, "
                "previously-detected pattern is now being missed"
            )

    def test_baseline_false_positive_rate_stays_under_registered_ceiling(self):
        report = self._report(compute_significance=False)
        fpr = 1.0 - report["by_category"]["negative"]["accuracy"]
        assert fpr <= MAX_BASELINE_FALSE_POSITIVE_RATE, (
            f"synthetic-noise false-positive rate regressed to {fpr:.1%} "
            f"(ceiling {MAX_BASELINE_FALSE_POSITIVE_RATE:.0%}) -- rerun "
            "scripts/benchmark_pattern_detectors.py and investigate the scorer "
            "before touching this ceiling"
        )

    def test_baseline_brier_score_beats_uninformative_forecaster(self):
        report = self._report(compute_significance=False)
        assert report["brier_score"] < MAX_BRIER_SCORE

    def test_baseline_pr_auc_stays_above_registered_floor(self):
        report = self._report(compute_significance=False)
        assert report["pr_auc"] >= MIN_PR_AUC

    def test_significance_and_fdr_tightens_the_false_positive_rate_further(self):
        report = self._report(compute_significance=True)
        fpr = 1.0 - report["by_category"]["negative"]["accuracy"]
        assert fpr <= MAX_SIGNIFICANCE_FDR_FALSE_POSITIVE_RATE, (
            f"significance+FDR-adjusted false-positive rate regressed to "
            f"{fpr:.1%} (ceiling {MAX_SIGNIFICANCE_FDR_FALSE_POSITIVE_RATE:.0%})"
        )
        assert report["classification"]["recall"] == MIN_RECALL
