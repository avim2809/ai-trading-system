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

import core_v1_preregistered as pre

DRAFT = ROOT / "plan" / "drafts" / "P3-11" / "core_v1_prereg_DRAFT.yaml"
# Pinned at freeze: any edit to the frozen constants changes this and must be a new pre-registration.
PINNED_FINGERPRINT = "6274c43219d3c7fe1fe9d9686bd6a8cfe3c643fc32aaf68d86116279746c2bba"   # batch-20 re-freeze (supersedes 7826fb03...)
PINNED_DRAFT_SPEC_HASH = "b964385057cf7a7ef25b5f14f36ce2551f0511adff86dbca968ce1874a9bdbf0"
PINNED_WEIGHTS_SHA256 = "6eb94e426e90a832b4a7c1778f36212bd2e1503d6cdb3e89141d45f8df3d0d83"


def test_fingerprint_stable_and_pinned():
    fp = pre.bars_fingerprint()
    assert fp == pre.bars_fingerprint() and re.fullmatch(r"[0-9a-f]{64}", fp)
    assert fp == PINNED_FINGERPRINT


def test_gate7_benchmark_variant_pinned_annual_monthly_sensitivity():
    assert (pre.BENCHMARK_GATE_VARIANT, pre.BENCHMARK_SENSITIVITY_VARIANT) == ("annual", "monthly")
    b = next(v["benchmark"] for v in yaml.safe_load((ROOT / pre.GATES_FILE).read_text()).values()
             if isinstance(v, dict) and isinstance(v.get("benchmark"), dict))
    assert b["uses_higher_after_tax_sharpe_of_the_two"] is False and b["gate_variant"] == "annual"


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
    assert pre.instrument_weights_hash({**pre.INSTRUMENT_WEIGHTS, "SPY": 0.5}) != pre.INSTRUMENT_WEIGHTS_SHA256


def _class_groups():
    u = yaml.safe_load((ROOT / "config/universe_etf.yaml").read_text())
    return u["asset_classes"]


def test_weights_are_one_group_per_asset_class():
    classes = _class_groups()
    assert pre.INSTRUMENT_WEIGHT_SCHEME == "handcrafted_one_group_per_asset_class"
    w = pre.INSTRUMENT_WEIGHTS
    assert list(w) == pre.UNIVERSE or set(w) == set(pre.UNIVERSE)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
    assert sorted(s for m in classes.values() for s in m) == sorted(pre.UNIVERSE)
    class_w = {c: sum(w[s] for s in m) for c, m in classes.items()}
    assert len(classes) == 10
    assert all(v == pytest.approx(1 / len(classes), abs=1e-12) for v in class_w.values())   # equal across classes
    for c, members in classes.items():                                                      # equal within each class
        assert all(w[s] == pytest.approx(class_w[c] / len(members), abs=1e-12) for s in members)
    assert w["SPY"] == pytest.approx(1 / 30) and w["EFA"] == pytest.approx(0.1)
    assert w["SPY"] != pytest.approx(1 / len(pre.UNIVERSE))                                 # not the old equal 1/14


def test_weights_match_firm_portfolio_weights_and_hash():
    from firm.portfolio.weights import handcraft_weights
    classes = _class_groups()
    ref = handcraft_weights({c: {c: list(m)} for c, m in classes.items()})
    assert {s: float(ref[s]) for s in ref.index} == pre.INSTRUMENT_WEIGHTS
    assert pre.instrument_weights_hash() == pre.INSTRUMENT_WEIGHTS_SHA256
    assert re.fullmatch(r"[0-9a-f]{64}", pre.INSTRUMENT_WEIGHTS_SHA256)
    assert pre.INSTRUMENT_WEIGHTS_SHA256 == PINNED_WEIGHTS_SHA256


def test_family_n_and_raw_count_semantics_follow_gates():
    gates = yaml.safe_load((ROOT / pre.GATES_FILE).read_text())
    counts = gates["n_counts"]
    assert pre.FAMILY_N == counts["family_provisional"] == pre.FAMILY_N_LEGACY + pre.FAMILY_N_GRID_MAX
    assert pre.FAMILY_N_GRID_MAX == pre.MAX_GRID_SIZE == len(pre.GRID) == 12
    assert pre.FAMILY_N_LEGACY == 1 + 10 + 5 + 3 + 0 == 19       # trend, alt_premia, S3, S5, futures_trend per n_rule
    assert "the core_v1 grid (up to 12)" in gates["n_rule"]
    # diagnostics are raw-count only; the module does not add them to the family N
    diag = sum(pre.EXPECTED_LEDGER_ROWS.values())
    assert diag == pre.DIAGNOSTIC_ROWS_RAW_ONLY == 110
    assert pre.NEW_RAW_ROWS == diag + len(pre.GRID)
    assert pre.FAMILY_N_ROW_KINDS == ("grid",) and not set(pre.FAMILY_N_ROW_KINDS) & set(pre.EXPECTED_LEDGER_ROWS)
    assert counts["raw_with_estimates"] == 463      # verdicts also state this raw count (it predates core_v1 rows)


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


# ---- batch-20 re-freeze: gate-5 coverage floor and the P4-03 limits layer -------------------------------------------------------------------------
def test_refreeze_values_and_their_sources():
    gates = yaml.safe_load((ROOT / pre.GATES_FILE).read_text())
    rp = gates["robustness_parameters"]
    assert pre.MIN_ACTIVE_FRACTION == 0.5 and 0 < pre.MIN_ACTIVE_FRACTION <= 1
    assert pre.VOL_EWMA_SPAN == 252 and isinstance(pre.VOL_EWMA_SPAN, int)
    assert rp["vol_ewma_span"]["value"] is None                     # still null in the gates; frozen here
    assert pre.MAX_VOL_SCALE == rp["max_vol_scale"]["value"] == 1.5
    assert pre.INSTRUMENT_RISK_CAP_MULTIPLE == rp["instrument_risk_cap_multiple"]["value"] == 2.0
    assert pre.CLASS_RISK_CAP_APPLIED == 1.0
    assert pre.FROZEN_LIMITS == {"vol_ewma_span": 252}
    risk = yaml.safe_load((ROOT / "config" / "risk.yaml").read_text())
    assert risk["max_vol_scale"] == pre.MAX_VOL_SCALE and risk["max_instrument_risk_mult"] == pre.INSTRUMENT_RISK_CAP_MULTIPLE
    assert pre._risk_yaml_mismatches() == []


def test_every_new_value_is_in_the_fingerprint(monkeypatch):
    before = pre.bars_fingerprint()
    for name, val in (("MIN_ACTIVE_FRACTION", 0.6), ("VOL_EWMA_SPAN", 126), ("MAX_VOL_SCALE", 2.0),
                      ("INSTRUMENT_RISK_CAP_MULTIPLE", 3.0), ("CLASS_RISK_CAP_APPLIED", 0.4)):
        monkeypatch.setattr(pre, name, val)
        assert pre.bars_fingerprint() != before, name
        monkeypatch.undo()
    assert pre.bars_fingerprint() == before


def test_min_active_fraction_is_structurally_sound():
    """Universe metadata only (no data): the floor needs at least 3 asset classes, and every named episode meets it at its start."""
    import pandas as pd

    uni = yaml.safe_load((ROOT / "config" / "universe_etf.yaml").read_text())
    n = len(pre.UNIVERSE)
    need = -(-int(pre.MIN_ACTIVE_FRACTION * n * 1000) // 1000)                      # ceil(0.5 * 14) = 7
    sizes = sorted((len(m) for m in pre.ASSET_CLASSES.values()), reverse=True)
    assert need == 7 and sum(sizes[:2]) < need                                       # any 7 instruments span >= 3 classes
    assert need >= 3 and -(-1 // 0.40) <= 3                                          # 0.40 class cap needs >= 3 held classes
    first = {str(i["symbol"]): pd.Timestamp(i["first_trade_date"]) for i in uni["instruments"]}
    sp = yaml.safe_load((ROOT / "config/stress_periods.yaml").read_text())
    episodes = next(v for v in sp.values() if isinstance(v, list) and v and "name" in v[0])
    for e in episodes:
        start = pd.Timestamp(e["start"])
        active = sum(1 for t in first.values() if t + pd.offsets.BDay(pre.ENTRY_GATE_DAYS) <= start)
        assert active / n >= pre.MIN_ACTIVE_FRACTION, e["name"]


def test_all_27_gate6_parameters_are_covered_by_the_frozen_module():
    from firm.research import core_v1_pipeline as P

    gates = yaml.safe_load((ROOT / pre.GATES_FILE).read_text())
    names = P.perturbable_names(gates, frozen=pre.FROZEN_LIMITS)
    assert names["unassessed"] == [] and len(names["assessed"]) == pre.N_ROBUSTNESS_PARAMS == 27
    assert pre.EXPECTED_LEDGER_ROWS["robustness"] == 54 and pre.NEW_RAW_ROWS == 122


def test_unchanged_by_the_refreeze():
    """Everything the owner said to keep: weights scheme, seed, family-N semantics, universe, gates hash, annual gate variant, grid."""
    assert pre.SEED == 20261005 and pre.FAMILY_N == 31 and pre.FAMILY_N_ROW_KINDS == ("grid",)
    assert pre.INSTRUMENT_WEIGHT_SCHEME == "handcrafted_one_group_per_asset_class" and pre.INSTRUMENT_WEIGHTS_SHA256 == PINNED_WEIGHTS_SHA256
    assert pre.GATES_SHA256 == "bcecaec9ef44163596e27459779a124747d382c8068bf0607d4793df79f71540"
    assert pre.BENCHMARK_GATE_VARIANT == "annual" and len(pre.GRID) == 12 and len(pre.UNIVERSE) == 14
    assert pre.FILE_HASHES["config/universe_etf.yaml"] == "379cb0d4611f46561b1125c72276803ea590ab3de136bcfb3d0ce6a9696d3514"


def test_draft_yaml_is_a_valid_new_spec_with_the_refreeze_content():
    from firm.research import prereg as PR

    d = yaml.safe_load(DRAFT.read_text())
    assert d["approver"] == "" and d["approved_at_utc"] == ""                       # the owner fills both
    spec = PR.spec_from_dict({**d, "approver": "owner", "approved_at_utc": "2026-10-07T00:00:00Z"})
    assert PR.validate_spec(spec) == [] and PR.spec_hash(spec) == PINNED_DRAFT_SPEC_HASH
    assert spec.gates_hash == pre.GATES_SHA256 and "min_active_fraction 0.5" in spec.falsification
    assert "robustness_all_27_parameters_perturbed" in spec.metrics
    old = yaml.safe_load((ROOT / "research" / "preregistration" / "20261006_core_v1.yaml").read_text())
    assert PR.spec_hash(PR.spec_from_dict(old)) != PINNED_DRAFT_SPEC_HASH            # a genuinely new pre-registration
    assert d["param_grid"] == old["param_grid"] and d["universe"] == old["universe"] and d["family"] == old["family"] == "core_v1"


def test_verify_before_run_refuses_a_drifted_risk_limit(tmp_path):
    (tmp_path / "config").mkdir()
    for rel in pre.FILE_HASHES:
        (tmp_path / rel).write_bytes((ROOT / rel).read_bytes())
    risk = yaml.safe_load((ROOT / "config" / "risk.yaml").read_text())
    risk["max_vol_scale"] = 2.0
    (tmp_path / "config" / "risk.yaml").write_text(yaml.safe_dump(risk))
    with pytest.raises(pre.PreregError, match="frozen limits changed"):
        pre.verify_before_run(tmp_path)
    risk["max_vol_scale"], risk["vol_ewma_span"] = 1.5, 63
    (tmp_path / "config" / "risk.yaml").write_text(yaml.safe_dump(risk))
    with pytest.raises(pre.PreregError, match="vol_ewma_span"):
        pre.verify_before_run(tmp_path)
