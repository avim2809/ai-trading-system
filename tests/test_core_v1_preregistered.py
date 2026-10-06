"""core_v1 frozen pre-registration module: fingerprint, grid vs the draft, file hashes, charter gate. No data is read."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import core_v1_preregistered as pre  # noqa: E402

DRAFT = ROOT / "plan" / "drafts" / "P3-11" / "core_v1_prereg_DRAFT.yaml"
# Pinned at freeze: any edit to the frozen constants changes this and must be a new pre-registration.
PINNED_FINGERPRINT = "495e6d70b6ea20994328582888ba8e466ac31ca27908a0712c9fc42f0c24550e"


def test_fingerprint_stable_and_pinned():
    fp = pre.bars_fingerprint()
    assert fp == pre.bars_fingerprint() and re.fullmatch(r"[0-9a-f]{64}", fp)
    assert fp == PINNED_FINGERPRINT


def test_fingerprint_changes_with_a_constant(monkeypatch):
    before = pre.bars_fingerprint()
    monkeypatch.setattr(pre, "SEED", pre.SEED + 1)
    assert pre.bars_fingerprint() != before


def test_status_timestamp_and_declarations():
    assert pre.STATUS and "PLACEHOLDER" not in pre.STATUS.upper()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", pre.PREREGISTERED_AT)
    assert pre.CLEANING_VERSION == "v3" and "5 bars" in pre.CLEANING_LOOKAHEAD_DISCLOSURE
    assert "2026-09-30" in pre.POST_SEAL_DATA_DECLARATION and pre.WINDOW[1] == "2026-09-30"
    assert isinstance(pre.SEED, int) and pre.EMBARGO_PCT == 0.01


def test_grid_equals_the_draft_grid():
    d = yaml.safe_load(DRAFT.read_text())
    axes = d["param_grid"]
    assert pre.GRID_AXES == axes
    assert len(pre.GRID) == 12 == d["max_grid_size"] == pre.MAX_GRID_SIZE
    assert len({tuple(sorted(g.items())) for g in pre.GRID}) == 12
    assert pre.DEFAULT_CONFIG in pre.GRID
    assert all(set(g) == set(axes) for g in pre.GRID)          # no tau / gross cap dimension
    assert pre.GATES_SHA256 == d["gates_hash"]
    assert pre.UNIVERSE == d["universe"] and list(pre.WINDOW) == d["date_range"]
    assert pre.POST_SEAL_DATA_DECLARATION == d["post_seal_data_declaration"]


def test_file_hashes_equal_current_files_and_the_draft_header():
    pre.verify_frozen_inputs(ROOT)
    header = DRAFT.read_text()
    for rel, h in pre.FILE_HASHES.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h
        if rel != pre.GATES_FILE:
            assert h in header


def test_changed_file_is_refused(tmp_path):
    (tmp_path / "config").mkdir()
    for rel in pre.FILE_HASHES:
        (tmp_path / rel).write_bytes((ROOT / rel).read_bytes())
    pre.verify_frozen_inputs(tmp_path)
    (tmp_path / "config/tax_il.yaml").write_text("changed\n")
    with pytest.raises(pre.PreregError, match="tax_il"):
        pre.verify_frozen_inputs(tmp_path)


def test_universe_matches_config_and_weights_hash():
    u = yaml.safe_load((ROOT / "config/universe_etf.yaml").read_text())
    assert [i["symbol"] for i in u["instruments"]] == pre.UNIVERSE
    assert sum(pre.INSTRUMENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert pre.instrument_weights_hash() == pre.INSTRUMENT_WEIGHTS_SHA256
    assert pre.instrument_weights_hash({**pre.INSTRUMENT_WEIGHTS, "SPY": 0.5}) != pre.INSTRUMENT_WEIGHTS_SHA256


def test_expected_ledger_rows_consistent_with_configs():
    assert set(pre.EXPECTED_LEDGER_ROWS) == {
        "constant_estimation", "robustness", "cost_stress", "stress_suite", "benchmark", "comparison"}
    g = yaml.safe_load((ROOT / pre.GATES_FILE).read_text())["robustness_parameters"]
    units = sum(len(v["value"]) if isinstance(v["value"], list) else 1 for v in g.values())
    assert pre.N_ROBUSTNESS_PARAMS == units
    assert pre.EXPECTED_LEDGER_ROWS["robustness"] == 2 * units
    sp = yaml.safe_load((ROOT / "config/stress_periods.yaml").read_text())
    episodes = next(v for v in sp.values() if isinstance(v, list) and v and "name" in v[0])
    assert pre.EXPECTED_LEDGER_ROWS["stress_suite"] == len(episodes)
    assert pre.EXPECTED_LEDGER_ROWS["constant_estimation"] == 11 + 14 + 14 + 1


def test_charter_missing_is_refused(tmp_path):
    with pytest.raises(pre.PreregError, match="charter missing"):
        pre.verify_charter(tmp_path)
    if not (ROOT / pre.CHARTER_PATH).exists():
        with pytest.raises(pre.PreregError, match="charter missing"):
            pre.verify_before_run(ROOT)       # no charter is committed yet


def test_charter_invalid_is_refused(tmp_path):
    d = tmp_path / "research" / "charters"
    d.mkdir(parents=True)
    (d / "core_v1.md").write_text("no front matter\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    with pytest.raises(pre.PreregError, match="charter (invalid|unparseable)"):
        pre.verify_charter(tmp_path)


def test_no_hardcoded_charter_hash():
    src = (_SCRIPTS / "core_v1_preregistered.py").read_text()
    assert "approved_commit" in src                                    # read at run time
    assert set(re.findall(r"[0-9a-f]{64}", src)) == set(pre.FILE_HASHES.values())   # only config hashes are literals


def test_charter_with_placeholder_front_matter_is_refused(tmp_path):
    d = tmp_path / "research" / "charters"
    d.mkdir(parents=True)
    (d / "core_v1.md").write_text("---\nfamily: core_v1\napproved_commit: <fill>\n---\n# Mechanism\n# Falsification\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    with pytest.raises(pre.PreregError, match="charter invalid"):
        pre.verify_charter(tmp_path)
