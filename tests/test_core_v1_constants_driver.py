"""P3-11 driver (scripts/run_core_v1_constants.py) on a SYNTHETIC panel in tmp dirs with a tmp ledger. No real data is read; the host ledger
file must be byte-for-byte untouched."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import core_v1_preregistered as pre
import run_core_v1_constants as D

from firm.research import core_v1_pipeline as P
from firm.research import ledger as L
from firm.research import run_guards as G

HOST_LEDGER = Path("/local/store/research-ledger/trials.jsonl")
GATES = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())


def host_stat():
    if not HOST_LEDGER.exists():
        return None
    st = HOST_LEDGER.stat()
    return (st.st_size, st.st_mtime_ns)


@pytest.fixture()
def tmp_ledger(tmp_path, monkeypatch):
    r = tmp_path / "ledger"
    r.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(r))
    monkeypatch.setenv("FIRM_LEDGER_ALLOW_DIRTY", "1")
    return r


@pytest.fixture(scope="module")
def synth():
    return P.synthetic_panel(seed=11, n_days=1400, symbols=("AAA", "BBB", "CCC"))


@pytest.fixture()
def run(tmp_path, tmp_ledger, synth):
    before = host_stat()
    out = tmp_path / "out"
    weights = {s: 1.0 / 3 for s in synth.symbols}
    ctx = P.LedgerCtx(mode="exploratory", snapshot_id="synthetic", seed=pre.SEED, fingerprint="fp")
    res = D.run_estimation(synth, gates=GATES, tau=0.09, instrument_weights=weights, window=(str(synth.close.index[0].date()), "2099-01-01"),
                           out_dir=out, ctx=ctx, code_commit="deadbeef", prereg_fingerprint="fp", gates_sha256="g" * 64)
    assert host_stat() == before                       # the host ledger is untouched
    return res, out, ctx, tmp_ledger


def test_constants_json_has_every_required_key(run):
    res, out, *_ = run
    c = json.loads((out / "constants.json").read_text())
    missing = [k for k in D.REQUIRED_KEYS if k not in c]
    assert not missing, missing
    assert c["tau"] == 0.09 and c["seed"] == pre.SEED and c["window"][1] == str(res["constants"]["window"][1])
    assert set(c["scalars"]) == {f"ewmac_{s}" for s in (2, 4, 8, 16, 32, 64)} | {f"breakout_{n}" for n in (20, 40, 80, 160, 320)}
    assert c["in_sample_declaration"] == pre.IN_SAMPLE_DECLARATION and c["post_seal_declaration"] == pre.POST_SEAL_DATA_DECLARATION


def test_writes_only_the_three_report_files_and_nothing_under_preregistration(run):
    out = run[1]
    assert sorted(p.name for p in out.iterdir()) == ["constants.json", "constants.md", "constants_addendum.draft.yaml"]
    assert not list((ROOT / "research" / "preregistration").glob("*addendum*"))
    with pytest.raises(ValueError, match="preregistration"):
        D.write_constants(ROOT / "research" / "preregistration" / "x.json", {}, {})


def test_file_hash_matches_write_constants_and_the_draft_addendum(run, tmp_path):
    res, out, *_ = run
    sha = hashlib.sha256((out / "constants.json").read_bytes()).hexdigest()
    assert sha == res["sha256"]
    add = yaml.safe_load((out / "constants_addendum.draft.yaml").read_text())
    assert add["constants_json_sha256"] == sha and add["status"] == "DRAFT_FOR_OWNER_COPY" and add["code_commit"] == "deadbeef"
    assert add["ledger_trial_ids"] and all(add["ledger_trial_ids"].values())
    p = tmp_path / "x" / "c.json"
    assert D.write_constants(p, {"a": 1}, {"m": 2}) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_pooled_scalar_mean_abs_is_ten_and_post_cap_mean_reported_separately(run):
    c = run[0]["constants"]
    for d in c["scalar_diagnostics"].values():
        assert d["mean_abs_uncapped"] == pytest.approx(10.0, abs=1e-9)
        assert d["mean_abs_capped"] <= 10.0 + 1e-9
    assert any(d["mean_abs_capped"] < 10.0 for d in c["scalar_diagnostics"].values())


def test_fdm_and_idm_obey_caps(run):
    c = run[0]["constants"]
    assert all(1.0 <= v <= c["fdm_cap"] for v in c["fdm"].values()) and all(u >= v - 1e-12 for u, v in zip(c["fdm_uncapped"].values(), c["fdm"].values(), strict=True))
    assert 1.0 <= c["idm"] <= c["idm_cap"] and c["idm"] <= c["idm_uncapped"] + 1e-12


def test_every_step_is_a_ledger_trial_with_kind_constant_estimation(run, synth):
    ctx, ledger_root = run[2], run[3]
    tr = L.trials(family="core_v1")
    assert len(tr) == 11 + 3 + 3 + 1                                   # scalars + speed filters + FDM sets + IDM
    assert all(row["kind"] == "constant_estimation" for row in tr["config"])
    steps = [row["step"] for row in tr["config"]]
    assert steps.count("scalar") == 11 and steps.count("speed_filter") == 3 and steps.count("fdm") == 3 and steps.count("idm") == 1
    assert set(ctx.trial_ids.values()) == set(tr["trial_id"])
    eng = L.trials(family="core_v1_engine")
    assert len(eng) == 3                                               # one sub-system engine run per instrument, disclosed rows
    assert (ledger_root / "trials.jsonl").exists() and L.verify_chain().ok


def test_real_run_context_is_registered_against_the_core_v1_prereg():
    ctx = D.build_ledger_ctx("snap", "fp", 7)
    assert (ctx.mode, ctx.prereg, ctx.family, ctx.seed) == ("registered", "core_v1", "core_v1", 7)
    assert ctx.default_params == pre.DEFAULT_CONFIG and pre.DEFAULT_CONFIG in pre.GRID


def test_tau_and_weights_are_not_chosen_by_the_driver():
    src = (ROOT / "scripts" / "run_core_v1_constants.py").read_text()
    assert "tau=float(facts[\"tau\"])" in src and "pre.INSTRUMENT_WEIGHTS" in src
    tree = ast.parse(src)
    nums = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value in (0.09, 0.9, 0.08, 0.10)]
    assert not nums, "a tau-like literal appears in the driver"


def test_two_instruments_with_different_costs_get_different_speed_sets():
    turn = {"X": {"ewmac_2": 30.0, "ewmac_4": 15.0}, "Y": {"ewmac_2": 30.0, "ewmac_4": 15.0}}
    out = D.apply_cost_speed_filter(turn, {"X": 0.0001, "Y": 0.0009}, {"X": 0.16, "Y": 0.16}, D.EXPECTED_RULE_SHARPE)
    assert out["X"]["ewmac"] == [2, 4] and out["Y"]["ewmac"] == [4] and out["X"] != out["Y"]


def test_data_access_is_not_bypassed():
    for name in ("run_core_v1_constants.py", "run_core_v1_evaluation.py"):
        p = ROOT / "scripts" / name
        if not p.exists():
            continue
        tree = ast.parse(p.read_text())
        bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in {"read_parquet", "read_csv", "read_feather", "read_hdf"}]
        assert not bad, f"{name}: direct file read at {bad}"
        opens = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"]
        assert not opens, f"{name}: bare open() at {opens}"


# ---- guards fire before anything is read -------------------------------------------------------------------------------------------------
@pytest.fixture()
def no_load(monkeypatch):
    """Any attempt to reach the data loader after a guard failure would be a bug."""
    import firm.data.etf_loader as EL

    def boom(*a, **k):
        raise AssertionError("data loader reached before the guards passed")

    monkeypatch.setattr(EL, "load_etf_universe", boom)
    monkeypatch.setattr(D.P, "panel_from_series", boom)


def test_main_refuses_as_root(monkeypatch, tmp_path, no_load):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(G.PreflightError, match="root"):
        D.main(["--manifest", str(tmp_path / "m.json"), "--out", str(tmp_path / "o")])


def test_main_refuses_when_a_sealed_path_is_readable(monkeypatch, tmp_path, no_load):
    monkeypatch.setattr(os, "geteuid", lambda: 1001)
    monkeypatch.setattr(os, "access", lambda p, m: True)          # everything readable: the ACL is missing
    with pytest.raises(G.PreflightError, match="readable"):
        D.main(["--manifest", str(tmp_path / "m.json")])


def _pass_preflight(monkeypatch):
    monkeypatch.setattr(G, "seal_preflight", lambda *a, **k: {"ok": True})


def test_main_refuses_outside_the_window(monkeypatch, tmp_path, no_load):
    import datetime as dt

    _pass_preflight(monkeypatch)
    monkeypatch.setattr(G, "current_utc", lambda: dt.datetime(2026, 10, 6, 15, 0, tzinfo=dt.UTC))   # Tuesday 11:00 ET
    with pytest.raises(G.PreflightError, match="window"):
        D.main(["--manifest", str(tmp_path / "m.json")])


def test_main_refuses_without_a_memory_cap(monkeypatch, tmp_path, no_load):
    import datetime as dt
    import resource

    _pass_preflight(monkeypatch)
    monkeypatch.setattr(G, "current_utc", lambda: dt.datetime(2026, 10, 7, 1, 0, tzinfo=dt.UTC))    # Tuesday 21:00 ET
    monkeypatch.setattr(resource, "getrlimit", lambda r: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    with pytest.raises(G.PreflightError, match="memory cap"):
        D.main(["--manifest", str(tmp_path / "m.json")])


def test_estimation_forecast_turnover_uses_forecasts_not_pnl():
    import inspect

    assert not {"returns", "pnl", "sharpe"} & set(inspect.signature(D.apply_cost_speed_filter).parameters)
    assert np.isfinite(D.EXPECTED_RULE_SHARPE) and D.EXPECTED_RULE_SHARPE > 0
