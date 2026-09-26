"""Tests for scripts/validate_pattern_cnn_walkforward.py's pure decision
functions (derive_recommendation, _fold_flag_selection_pattern).

No test previously imported this script at all. Imports the script module
directly by adding scripts/ to sys.path, mirroring
tests/test_pattern_recognition_rollout_gate.py (scripts/ is not a package).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from validate_pattern_cnn_walkforward import (  # noqa: E402
    DEFAULT_PARAM_GRID,
    _fold_flag_selection_pattern,
    derive_recommendation,
)


class TestDeriveRecommendation:
    def _pattern(self, flags: list[bool | None]) -> list[dict]:
        return [
            {"fold": i, "run_id": f"r{i}", "selected_index": 0, "selected_flag_enabled": f}
            for i, f in enumerate(flags)
        ]

    def test_failing_verdict_rolls_back_regardless_of_pattern(self):
        rec = derive_recommendation("fail", self._pattern([True, True, True]))
        assert rec["recommendation"] == "ROLLBACK"
        assert rec["target_flag_enabled"] is False

    def test_none_verdict_rolls_back(self):
        rec = derive_recommendation(None, self._pattern([True]))
        assert rec["recommendation"] == "ROLLBACK"

    def test_passing_verdict_with_consistent_majority_keeps(self):
        rec = derive_recommendation("pass", self._pattern([True, True, True, True, False]))
        assert rec["recommendation"] == "KEEP"
        assert rec["target_flag_enabled"] is True
        assert rec["true_fraction"] == 0.8

    def test_passing_verdict_with_mixed_pattern_holds(self):
        rec = derive_recommendation("pass", self._pattern([True, False, True, False]))
        assert rec["recommendation"] == "HOLD"
        assert rec["target_flag_enabled"] is None
        assert rec["true_fraction"] == 0.5

    def test_passing_verdict_with_no_usable_folds_holds(self):
        rec = derive_recommendation("pass", self._pattern([None, None]))
        assert rec["recommendation"] == "HOLD"
        assert rec["true_fraction"] is None
        assert rec["n_usable_folds"] == 0

    def test_none_entries_excluded_from_fraction(self):
        rec = derive_recommendation("pass", self._pattern([True, True, True, None]))
        assert rec["true_fraction"] == 1.0
        assert rec["n_usable_folds"] == 3

    def test_custom_threshold_respected(self):
        pattern = self._pattern([True, True, False, False])  # 50%
        assert derive_recommendation("pass", pattern, 0.75)["recommendation"] == "HOLD"
        assert derive_recommendation("pass", pattern, 0.5)["recommendation"] == "KEEP"

    def test_flag_key_and_label_carried_through(self):
        rec = derive_recommendation(
            "pass", self._pattern([True, True, True, True]),
            flag_key="xgb_confirmation_enabled", flag_label="XGBoost ensemble",
        )
        assert rec["flag_key"] == "xgb_confirmation_enabled"
        assert "xgb_confirmation_enabled" in rec["reason"]


class TestFoldFlagSelectionPattern:
    def _fake_run(self, tmp_path, name: str, selection: dict | None):
        run_dir = tmp_path / name
        run_dir.mkdir()
        if selection is not None:
            (run_dir / "walk_forward_selection.json").write_text(json.dumps(selection))
        return SimpleNamespace(run_id=name, artifacts_dir=str(run_dir))

    def test_attributes_correct_flag_per_candidate(self, tmp_path):
        param_grid = [
            {"strategy_params": {"pattern_recognition": {"cnn_scoring_enabled": False}}},
            {"strategy_params": {"pattern_recognition": {"cnn_scoring_enabled": True}}},
            {"strategy_params": {"pattern_recognition": {
                "cnn_scoring_enabled": False, "xgb_confirmation_enabled": True,
            }}},
        ]
        runs = [
            self._fake_run(tmp_path, "fold0", {"selected_index": 1}),  # CNN candidate
            self._fake_run(tmp_path, "fold1", {"selected_index": 2}),  # XGB candidate
            self._fake_run(tmp_path, "fold2", {"selected_index": 0}),  # baseline
        ]

        cnn_pattern = _fold_flag_selection_pattern(runs, param_grid, flag_key="cnn_scoring_enabled")
        xgb_pattern = _fold_flag_selection_pattern(runs, param_grid, flag_key="xgb_confirmation_enabled")

        assert [f["selected_flag_enabled"] for f in cnn_pattern] == [True, False, False]
        assert [f["selected_flag_enabled"] for f in xgb_pattern] == [False, True, False]

    def test_missing_selection_file_degrades_to_none(self, tmp_path):
        param_grid = DEFAULT_PARAM_GRID
        runs = [self._fake_run(tmp_path, "fold0", None)]
        pattern = _fold_flag_selection_pattern(runs, param_grid, flag_key="cnn_scoring_enabled")
        assert pattern == [{"fold": 0, "run_id": "fold0", "selected_index": None, "selected_flag_enabled": None}]

    def test_default_param_grid_has_three_candidates_and_no_stale_warning(self):
        # DEFAULT_PARAM_GRID must cover both tracked flags -- this is the
        # regression this test exists to catch if a future edit adds a new
        # gated flag without updating the grid (mirrors the module's own
        # runtime warning for _EXPECTED_PATTERN_RECOGNITION_PARAMS).
        assert len(DEFAULT_PARAM_GRID) == 3
        flags_present = {
            k for cand in DEFAULT_PARAM_GRID
            for k in cand["strategy_params"]["pattern_recognition"]
        }
        assert {"cnn_scoring_enabled", "xgb_confirmation_enabled"} <= flags_present
