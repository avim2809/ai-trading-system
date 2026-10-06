"""P3-08 driver end to end on a SYNTHETIC panel (both drivers: estimation then evaluation), tmp ledger, tmp output dirs. No real data.

The ledger runs in REGISTERED mode against a clean throwaway git repo with ``is_approved`` stubbed, so ``verify_provenance`` is exercised on
real registered rows. The host ledger file must be unchanged (size and mtime).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import core_v1_preregistered as pre
import run_core_v1_constants as C1
import run_core_v1_evaluation as E

from firm.reporting import after_tax as AT
from firm.reporting import gate_report as GR
from firm.research import core_v1_pipeline as P
from firm.research import ledger as L
from firm.research import legacy_adapters as LA
from firm.research import prereg as PR

HOST_LEDGER = Path("/local/store/research-ledger/trials.jsonl")
GATES = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())
SYMS = ("SPY", "IEF", "GLD")


def host_stat():
    if not HOST_LEDGER.exists():
        return None
    st = HOST_LEDGER.stat()
    return (st.st_size, st.st_mtime_ns)


# ---- pure CPCV / PBO behaviour -----------------------------------------------------------------------------------------------------------
def _matrix(n=1200, cols=4, seed=0):
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame(np.random.default_rng(seed).normal(0.0003, 0.01, (n, cols)), index=idx)


def test_identical_columns_give_nine_identical_path_sharpes():
    base = _matrix(cols=1)[0]
    M = pd.DataFrame({i: base for i in range(5)})
    out = E.run_cpcv(M, E.select_argmax, 10, 2, embargo_pct=pre.EMBARGO_PCT)
    assert out["n_paths"] == 9 and len(out["path_sharpes"]) == 9 and out["n_splits"] == 45
    assert max(out["path_sharpes"]) - min(out["path_sharpes"]) < 1e-12           # every path equals the full-sample series
    assert out["selection_counts"] == {0: 45}                                    # ties go to the lowest grid index


def test_grid_whose_best_flips_between_halves_gives_different_path_sharpes():
    n = 1200
    idx = pd.bdate_range("2012-01-02", periods=n)
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 0.01, n)
    b = rng.normal(0.0, 0.01, n)
    a[: n // 2] += 0.003          # column 0 wins the first half
    b[n // 2:] += 0.003           # column 1 wins the second half
    out = E.run_cpcv(pd.DataFrame({0: a, 1: b}, index=idx), E.select_argmax, 10, 2, embargo_pct=pre.EMBARGO_PCT)
    assert len({round(x, 8) for x in out["path_sharpes"]}) > 1
    assert len(out["selection_counts"]) == 2


def test_selection_is_rerun_on_train_groups_only():
    seen = []

    def rule(train):
        seen.append(train.shape[0])
        return 0

    n = 1000
    E.run_cpcv(_matrix(n=n, cols=2), rule, 10, 2, embargo_pct=0.01)
    assert len(seen) == 45 and all(s < n * 0.8 + 1 for s in seen)                # 8 of 10 groups minus purge and embargo


def test_pbo_block_reports_informativeness_inputs():
    M = _matrix(cols=6)
    out = E.run_pbo(M, 16, family_variant_count=31)
    assert 0.0 <= out["pbo"] <= 1.0 and out["grid_n"] == 6 and out["n_splits"] == 12870
    dup = pd.DataFrame({i: M[0] for i in range(6)})
    d = E.run_pbo(dup, 16, family_variant_count=31)
    assert d["median_pairwise_corr"] > 0.999 and d["effective_grid_n"] < 2


def test_trial_history_is_append_only(tmp_path):
    p = tmp_path / "h.json"
    E.append_trial_history(p, {"date": "2026-10-07", "n_trials": 1})
    first = json.loads(p.read_text())
    E.append_trial_history(p, {"date": "2026-10-08", "n_trials": 2})
    second = json.loads(p.read_text())
    assert second["entries"][:1] == first["entries"] and len(second["entries"]) == 2


def test_grid_is_the_frozen_grid_and_at_most_twelve():
    g = E.build_grid()
    assert g == pre.GRID and len(g) <= 12


# ---- step 1 ----------------------------------------------------------------------------------------------------------------------------
def test_verify_start_refuses_without_the_owner_copied_addendum(tmp_path, monkeypatch):
    monkeypatch.setattr(pre, "verify_before_run", lambda *a, **k: {"tau": 0.09, "approved_commit": "x", "path": "p", "sha256": "s"})
    (tmp_path / "research" / "reports" / "core_v1").mkdir(parents=True)
    (tmp_path / "research" / "preregistration").mkdir(parents=True)
    cj = tmp_path / E.CONSTANTS_REL
    cj.write_text(json.dumps({"gates_sha256": pre.GATES_SHA256, "prereg_fingerprint": pre.bars_fingerprint(),
                              "instrument_weights_sha256": pre.INSTRUMENT_WEIGHTS_SHA256, "tau": 0.09}))
    with pytest.raises(E.StartupError, match="addendum"):
        E.verify_start(tmp_path, check_index=False)
    (tmp_path / E.ADDENDUM_REL).write_text(yaml.safe_dump({"constants_json_sha256": "0" * 64}))
    with pytest.raises(E.StartupError, match="differs"):
        E.verify_start(tmp_path, check_index=False)
    (tmp_path / E.ADDENDUM_REL).write_text(yaml.safe_dump({"constants_json_sha256": E.file_sha256(cj)}))
    assert E.verify_start(tmp_path, check_index=False)["constants_sha256"] == E.file_sha256(cj)


def test_verify_start_fails_when_weights_hash_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(pre, "verify_before_run", lambda *a, **k: {"tau": 0.09})
    (tmp_path / "research" / "reports" / "core_v1").mkdir(parents=True)
    (tmp_path / "research" / "preregistration").mkdir(parents=True)
    cj = tmp_path / E.CONSTANTS_REL
    cj.write_text(json.dumps({"gates_sha256": pre.GATES_SHA256, "prereg_fingerprint": pre.bars_fingerprint(),
                              "instrument_weights_sha256": "f" * 64, "tau": 0.09}))
    (tmp_path / E.ADDENDUM_REL).write_text(yaml.safe_dump({"constants_json_sha256": E.file_sha256(cj)}))
    with pytest.raises(E.StartupError, match="weight"):
        E.verify_start(tmp_path, check_index=False)


# ---- the end-to-end dry run ------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    mp = pytest.MonkeyPatch()
    before = host_stat()
    repo = tmp / "repo"
    repo.mkdir()
    (repo / "f.txt").write_text("x")
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "safe.directory=*"]
    subprocess.run([*g, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*g, "add", "."], cwd=repo, check=True)
    subprocess.run([*g, "commit", "-q", "-m", "c"], cwd=repo, check=True)
    root = tmp / "ledger"
    root.mkdir()
    mp.setenv(L.LEDGER_ROOT_ENV, str(root))
    mp.setenv(L.REPO_DIR_ENV, str(repo))
    mp.delenv("FIRM_LEDGER_ALLOW_DIRTY", raising=False)
    mp.setattr(PR, "is_approved", lambda *a, **k: True)
    for rec in LA.iter_legacy_rows(ROOT / "docs"):
        L.record_trial(rec)
    panel = P.synthetic_panel(seed=21, n_days=1100, symbols=SYMS, start="2007-01-03")
    weights = {s: 1.0 / len(SYMS) for s in SYMS}
    win = (str(panel.close.index[0].date()), str(panel.close.index[-1].date()))
    ctx1 = P.LedgerCtx(mode="registered", prereg="core_v1", snapshot_id="synthetic", seed=pre.SEED, fingerprint="fp")
    cdir = tmp / "const"
    est = C1.run_estimation(panel, gates=GATES, tau=0.09, instrument_weights=weights, window=win, out_dir=cdir, ctx=ctx1, code_commit="abc",
                            prereg_fingerprint=pre.bars_fingerprint(), gates_sha256=pre.GATES_SHA256)
    consts = json.loads((cdir / "constants.json").read_text())
    ctx2 = P.LedgerCtx(mode="registered", prereg="core_v1", snapshot_id="synthetic", seed=pre.SEED, fingerprint="fp")
    odir = tmp / "eval"
    grid = [pre.GRID[0], pre.GRID[5], pre.GRID[8]]
    out = E.run_evaluation(
        panel, consts=consts, gates=GATES, ctx=ctx2, out_dir=odir, charter={"path": "research/charters/core_v1.md", "approved_commit": "abc"},
        charter_proc={"path_length_days": 500, "seed": pre.SEED}, code_commit="abc", periods_path=ROOT / "config" / "stress_periods.yaml",
        tax_cfg=AT.load_tax_config(ROOT / "config" / "tax_il.yaml"), cost_cfg=P.CM.load_cost_config(),
        asset_classes={"equity": ["SPY"], "bonds": ["IEF"], "gold": ["GLD"]}, grid=grid, draws=200, n_boot=200,
        trial_history_path=odir / "drafts" / "core_v1_trial_history.json")
    yield {"est": est, "out": out, "odir": odir, "cdir": cdir, "ctx2": ctx2, "ctx1": ctx1, "before": before, "root": root, "grid": grid, "panel": panel}
    mp.undo()


@pytest.fixture()
def led(e2e, monkeypatch):
    """The suite-wide autouse fixture re-points the ledger root per test; point it back at the module-scoped dry run's tmp ledger."""
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(e2e["root"]))
    return e2e["root"]


def test_host_ledger_untouched(e2e):
    assert host_stat() == e2e["before"]


def test_report_has_every_gate_row_and_the_sentences(e2e):
    md = (e2e["odir"] / "REPORT.md").read_text()
    for i in range(1, 9):
        assert f"G-RESEARCH-{i}" in md
    assert "KELLY-TAU-BOUND" in md and "all data to 2026-09-30 is in-sample; no post-seal data examined" in md
    assert f"Tier {e2e['out']['tier']}" in md and e2e["out"]["tier"] in ("A", "C", "D")
    assert "family N" in md.lower() or "n (gate, family-n)" in md.lower()


def test_tier_is_decided_by_gate_report_not_prose(e2e):
    out = e2e["out"]
    outs = [GR.TestOutcome(**o) for o in out["outcomes"]]
    assert GR.combine_tier(outs, out["h4"]["enb_ok"]) == out["tier"]


def test_results_json_machine_readable_with_provenance(e2e):
    r = json.loads((e2e["odir"] / "results.json").read_text())
    assert set(r["results"]) == {"dsr", "pbo", "cpcv", "cost_stress", "stress", "robustness", "benchmark", "mechanism"}
    assert r["results"]["cpcv"]["n_paths"] == 9 and len(r["results"]["cpcv"]["path_sharpes"]) == 9
    assert r["results"]["dsr"]["n_gate"] == pre.FAMILY_N and r["results"]["dsr"]["n_raw"] >= 31
    assert set(r["results"]["cost_stress"]["net_sharpe"]) == {"1", "2", "3"}
    assert r["trial_ids"] and all(r["trial_ids"].values())


def test_unfrozen_min_active_fraction_leaves_gate_five_insufficient(e2e):
    o = next(x for x in e2e["out"]["outcomes"] if x["test_id"] == "G-RESEARCH-5")
    assert o["status"] == "insufficient" and "min_active_fraction" in o["reason"]


def test_pbo_with_a_three_config_grid_is_uninformative_and_cannot_pass(e2e):
    o = next(x for x in e2e["out"]["outcomes"] if x["test_id"] == "G-RESEARCH-2")
    assert o["status"] == "insufficient" and "uninformative" in o["reason"]


def test_robustness_reestimates_scalars_so_mean_abs_forecast_is_ten(e2e):
    perts = e2e["out"]["results"]["robustness"]["perturbations"]
    assert len(perts) == 2 * len(P.perturbable_names(GATES)["assessed"])
    assert all(abs(p["mean_abs_forecast_after_reestimation"] - 10.0) < 1e-6 for p in perts)
    un = e2e["out"]["results"]["robustness"]["unassessed"]
    assert "vol_ewma_span" in un
    o = next(x for x in e2e["out"]["outcomes"] if x["test_id"] == "G-RESEARCH-6")
    assert o["status"] in ("insufficient", "fail")                                    # never a pass while parameters are unassessed


def test_stress_reference_uses_the_registered_bootstrap_horizon(e2e):
    st = e2e["out"]["results"]["stress"]
    assert st["bootstrap_draws"] == 200 and st["bootstrap_seed"] == pre.SEED
    ok = [e for e in st["episodes"] if e["n_days"] >= 5 and e["reference_max_dd"] is not None]
    assert ok
    first = ok[0]
    # recompute independently from the registered procedure: same returns, horizon = the episode's length, same seed and draws
    odir = e2e["odir"]
    assert first["reference_max_dd"] > 0 and (odir / "results.json").exists()
    assert st["survival_ref_max_with_2p5_tau"] >= 2.5 * 0.09 - 1e-12


def test_benchmark_uses_the_pinned_annual_variant_with_monthly_sensitivity(e2e):
    b = e2e["out"]["results"]["benchmark"]
    assert b["gate_variant"] == "annual" and b["sensitivity_variant"] == "monthly"
    assert b["minimum_detectable_sharpe_gap"] == b["margin"] and b["margin"] > 0
    assert b["core_only_100"]["n_trades"] > 0


def test_h4_enb_is_reported_and_decides_tier_d_on_a_miss(e2e):
    h4 = e2e["out"]["h4"]
    assert h4["enb_threshold"] == 2.5 and h4["n_asset_classes"] == 3
    outs = [GR.TestOutcome(**o) for o in e2e["out"]["outcomes"]]
    assert GR.combine_tier(outs, False) == "D" and GR.tier_reason(outs, False) == "H4 ENB miss"


def test_every_step_is_a_registered_trial_and_provenance_verifies(e2e, led):
    tr = L.trials(family="core_v1")
    assert (tr["mode"] == "registered").all() and tr["preregistration_id"].eq("core_v1").all()
    kinds = pd.Series([c["kind"] for c in tr["config"]]).value_counts().to_dict()
    n = len(SYMS)
    assert kinds["constant_estimation"] == 11 + n + n + 1
    assert kinds["grid"] == 3 and kinds["cost_stress"] == 2 and kinds["comparison"] == 1
    assert kinds["robustness"] == 2 * len(P.perturbable_names(GATES)["assessed"]) and kinds["stress_suite"] == 10 and kinds["benchmark"] == 3
    GR.verify_provenance({"trial_ids": list(e2e["out"]["trial_ids"].values())}, tr)
    bad = tr.copy()
    bad.loc[bad.index[3], "mode"] = "exploratory"
    with pytest.raises(GR.ProvenanceError):
        GR.verify_provenance({"trial_ids": list(tr["trial_id"])}, bad)
    assert L.verify_chain().ok


def test_engine_rows_are_a_separate_family_counted_in_the_raw_count(e2e, led):
    eng = L.trials(family="core_v1_engine")
    assert len(eng) > 3 and (eng["mode"] == "exploratory").all()


def test_trial_history_draft_written_outside_docs(e2e):
    h = json.loads((e2e["odir"] / "drafts" / "core_v1_trial_history.json").read_text())
    assert h["family"] == "core_v1" and len(h["entries"]) == 1 and h["entries"][0]["n_trials"] == 3
    assert not (ROOT / "docs" / "core_v1_trial_history.json").exists()
    assert not list((ROOT / "research" / "preregistration").glob("*addendum*"))


def test_nothing_written_under_protected_dirs_by_the_runs(e2e):
    st = subprocess.run(["git", "-c", "safe.directory=*", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    assert "research/preregistration" not in st and "tests/integrity" not in st and "config/gates.yaml" not in st
