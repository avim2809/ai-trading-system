"""Tests for scripts/pattern_ml_preregistered_bars.py -- the pre-registered
pass/fail bar-checking logic for Workstream D
(``docs/pattern_recognition_plan.md``). Pure functions, exercised entirely
against synthetic measurements so this coverage needs no real backtest --
mirrors tests/test_validate_pattern_cnn_walkforward.py's convention of
importing a scripts/ module directly (scripts/ is not a package).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pattern_ml_preregistered_bars import (  # noqa: E402
    BARS,
    PARAM_GRID,
    bars_fingerprint,
    check_bar,
    evaluate_all,
)

# A measurement set that clears every bar -- the baseline every "flip one
# thing" test below mutates a single field away from.
_ALL_PASS = {
    "executable_expectancy_r_positive": 0.15,
    "expectancy_ci_excludes_zero": True,
    "expectancy_beats_placebo": True,
    "deflated_sharpe_min": 0.97,
    "pbo_max": 0.30,
    "pbo_min_candidates": len(PARAM_GRID),
    "fold_consistency_min": 0.875,
}


class TestCheckBar:
    def test_gt_comparison_passes_above_threshold(self):
        assert check_bar("deflated_sharpe_min", 0.96) is True

    def test_gt_comparison_fails_at_or_below_threshold(self):
        assert check_bar("deflated_sharpe_min", 0.95) is False
        assert check_bar("deflated_sharpe_min", 0.5) is False

    def test_lt_comparison_passes_below_threshold(self):
        assert check_bar("pbo_max", 0.49) is True

    def test_lt_comparison_fails_at_or_above_threshold(self):
        assert check_bar("pbo_max", 0.50) is False
        assert check_bar("pbo_max", 0.90) is False

    def test_bool_comparison(self):
        assert check_bar("expectancy_beats_placebo", True) is True
        assert check_bar("expectancy_beats_placebo", False) is False

    def test_none_value_is_none_not_false(self):
        """An uncomputable bar must read as None (fail-closed at the
        aggregate level), never silently coerced to a pass or a plain
        fail-as-if-measured."""
        assert check_bar("deflated_sharpe_min", None) is None

    def test_nan_value_is_none(self):
        assert check_bar("pbo_max", float("nan")) is None


class TestEvaluateAll:
    def test_all_pass_gives_overall_pass(self):
        result = evaluate_all(_ALL_PASS)
        assert result["overall_pass"] is True
        assert all(b["passed"] is True for b in result["per_bar"].values())

    def test_single_failing_bar_fails_overall(self):
        measurements = {**_ALL_PASS, "deflated_sharpe_min": 0.10}
        result = evaluate_all(measurements)
        assert result["overall_pass"] is False
        assert result["per_bar"]["deflated_sharpe_min"]["passed"] is False
        # Every other bar is unaffected.
        assert result["per_bar"]["pbo_max"]["passed"] is True

    def test_placebo_bar_failing_fails_overall_even_with_positive_expectancy(self):
        """The plan's own framing: if the placebo does as well or better,
        that's a fail regardless of the raw expectancy number."""
        measurements = {**_ALL_PASS, "expectancy_beats_placebo": False}
        result = evaluate_all(measurements)
        assert result["overall_pass"] is False

    def test_missing_measurement_fails_closed(self):
        """A bar with no entry in ``measurements`` at all (not just an
        explicit None) must still fail the overall verdict, not be skipped."""
        measurements = dict(_ALL_PASS)
        del measurements["fold_consistency_min"]
        result = evaluate_all(measurements)
        assert result["per_bar"]["fold_consistency_min"]["passed"] is None
        assert result["overall_pass"] is False

    def test_pbo_two_candidate_grid_fails_min_candidates_bar(self):
        """cscv_pbo degenerates to a coin flip at N=2 candidates -- even a
        good-looking PBO number must not pass on a 2-candidate grid."""
        measurements = {**_ALL_PASS, "pbo_max": 0.10, "pbo_min_candidates": 2}
        result = evaluate_all(measurements)
        assert result["per_bar"]["pbo_min_candidates"]["passed"] is False
        assert result["overall_pass"] is False

    def test_result_includes_fingerprint_and_timestamp(self):
        result = evaluate_all(_ALL_PASS)
        assert result["bars_fingerprint"] == bars_fingerprint()
        assert result["preregistered_at"]


class TestBarsFingerprint:
    def test_deterministic(self):
        assert bars_fingerprint() == bars_fingerprint()

    def test_changes_if_bars_change(self):
        """Not a mutation test of the committed file (that would defeat its
        own one-shot purpose) -- just confirms the hash is actually a
        function of BARS' content, by comparing against a hand-computed
        hash of a deliberately different payload."""
        import hashlib
        import json

        real = bars_fingerprint()
        tampered_payload = json.dumps(
            {"PREREGISTERED_AT": "2000-01-01T00:00:00Z", "BARS": {}}, sort_keys=True
        )
        tampered = hashlib.sha256(tampered_payload.encode("utf-8")).hexdigest()
        assert real != tampered


class TestParamGrid:
    def test_at_least_five_candidates(self):
        """PBO computed over < 5 candidates is not admissible evidence per
        this task's own instruction (N=2 degenerates to a coin flip)."""
        assert len(PARAM_GRID) >= 5

    def test_candidates_are_genuinely_distinct(self):
        import json

        serialized = [json.dumps(c, sort_keys=True) for c in PARAM_GRID]
        assert len(set(serialized)) == len(PARAM_GRID)

    def test_every_bar_has_a_description(self):
        for name, spec in BARS.items():
            assert spec["description"], name
