"""Unit tests of the P1-08 synthetic generator, spec handling and CLI guards (fast)."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from firm.validation import synthetic as syn
from firm.validation.synthetic import (
    STRESS_PARAMS,
    GarchTParams,
    Scenario,
    inject_carry,
    inject_trend,
    make_universe,
    simulate_garch_t,
)

ROOT = Path(__file__).resolve().parents[1]
SYN_SRC = ROOT / "src" / "firm" / "validation" / "synthetic.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("validate_stats_pipeline", ROOT / "scripts" / "validate_stats_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_garch_unit_variance_and_fat_tails():
    z = syn.standardised_t(400_000, 5.0, np.random.default_rng(1))
    assert abs(z.var() - 1.0) < 0.03
    p = GarchTParams(nu=5.0)
    r = simulate_garch_t(200_000, p, seed=2)
    m = r - r.mean()
    kurt = (m**4).mean() / (m**2).mean() ** 2
    assert kurt - 3.0 > 1.0
    sq = r**2 - (r**2).mean()
    assert (sq[1:] * sq[:-1]).mean() / sq.var() > 0.05  # volatility clustering


def test_garch_stationarity_guard():
    with pytest.raises(ValueError):
        simulate_garch_t(100, GarchTParams(alpha=0.1, beta=0.9), seed=1)
    with pytest.raises(ValueError):
        simulate_garch_t(100, GarchTParams(nu=2.0), seed=1)


def test_default_params_finite_fourth_moment():
    assert GarchTParams().finite_fourth_moment() is True
    assert STRESS_PARAMS.finite_fourth_moment() is False
    p = GarchTParams()
    assert abs(np.sqrt(p.omega / (1 - p.alpha - p.beta)) - 0.01) < 0.0003
    r = simulate_garch_t(300_000, p, seed=3)
    assert abs(r.std() - 0.01) < 0.0003 * 3


def _rule_sr(r, h):
    c = np.concatenate([[0.0], np.cumsum(r)])
    pos = np.zeros(len(r))
    for t in range(h, len(r)):
        pos[t] = 1.0 if c[t] - c[t - h] > 0 else -1.0
    sr = pos * r
    return sr[h:].mean() / sr[h:].std(ddof=1) * np.sqrt(252)


@pytest.mark.parametrize("h", [21, 126])
def test_inject_trend_hits_target_sharpe(h):
    r = simulate_garch_t(200_000, GarchTParams(), seed=4)
    x = inject_trend(r, 1.0, h, seed=5)
    assert abs(_rule_sr(x, h) - 1.0) < 0.15


def test_inject_carry_hits_target_sharpe():
    r = simulate_garch_t(200_000, GarchTParams(), seed=6)
    x = inject_carry(r, 1.0, seed=7)
    sr = x.mean() / x.std(ddof=1) * np.sqrt(252)
    assert abs(sr - 1.0) < 0.15


def test_make_universe_correlation():
    U = make_universe(10, 20_000, seed=8, corr=0.3)
    assert U.shape == (20_000, 10)
    c = np.corrcoef(U, rowvar=False)
    off = c[~np.eye(10, dtype=bool)]
    assert abs(off.mean() - 0.3) < 0.06
    U0 = make_universe(10, 20_000, seed=8, corr=0.0)
    assert abs(np.corrcoef(U0, rowvar=False)[~np.eye(10, dtype=bool)].mean()) < 0.06
    U1 = make_universe(5, 5000, seed=9, signal={"kind": "drift", "annual_sr": 3.0, "index": 2})
    assert U1[:, 2].mean() > 5 * np.delete(U1, 2, axis=1).mean(axis=0).max()


def test_seed_reproducible():
    a = simulate_garch_t(500, GarchTParams(), seed=11)
    b = simulate_garch_t(500, GarchTParams(), seed=11)
    c = simulate_garch_t(500, GarchTParams(), seed=12)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.array_equal(make_universe(4, 300, seed=1), make_universe(4, 300, seed=1))


def _tiny(**kw):
    base = {"name": "null_size", "kind": "null", "k_trials": 4, "n_days": 300, "annual_sr": 0.0, "garch": GarchTParams(),
                "B": 60, "n_sims": 3, "base_seed": 100, "alpha": 0.05, "tests": ("rc", "spa", "rw", "dsr", "psr0")}
    base.update(kw)
    return Scenario(**base)


def test_run_scenario_takes_parameters_only_from_spec():
    a = syn.run_scenario(_tiny(), workers=1)
    b = syn.run_scenario(_tiny(), workers=1)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    c = syn.run_scenario(_tiny(base_seed=999), workers=1)
    assert json.dumps(a["raw"], default=str) != json.dumps(c["raw"], default=str)
    d = syn.run_scenario(_tiny(k_trials=6), workers=1)
    assert d["spec"]["k_trials"] == 6
    # CLI refuses overrides that differ from the frozen values
    mod = _load_script()
    with pytest.raises(SystemExit):
        mod.main(["--out", "/nonexistent", "--sims", "7"])
    with pytest.raises(SystemExit):
        mod.main(["--out", "/nonexistent", "--seed", "1"])


def _acc(tmp_path, **over):
    gates = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())
    block = gates["p1_08_acceptance"]
    for k, v in over.items():
        block["size"][k] = v
    p = tmp_path / "gates.yaml"
    p.write_text(yaml.safe_dump(gates))
    return p


def test_load_acceptance_uses_frozen_block_and_hashes_scenarios(tmp_path):
    acc = syn.load_acceptance(ROOT / "config" / "gates.yaml")
    names = {s.name for s in acc["scenarios"]}
    assert {"null_size", "null_pbo", "power_single", "power_multi", "strong_drift"} <= names
    null = next(s for s in acc["scenarios"] if s.name == "null_size")
    assert (null.k_trials, null.n_days, null.B, null.n_sims) == (50, 2520, 2000, 500)
    assert len(acc["scenarios_sha256"]) == 64
    assert acc["bars"]["size"]["max_rejection_rate"] == yaml.safe_load(
        (ROOT / "config" / "gates.yaml").read_text())["p1_08_acceptance"]["size"]["max_rejection_rate"]


def _fake_results(rate, hi=None):
    r = {"k": int(rate * 500), "n": 500, "rate": rate, "lo": max(0.0, rate - 0.01), "hi": hi if hi is not None else rate + 0.01}
    z = {"k": 0, "n": 500, "rate": 0.0, "lo": 0.0, "hi": 0.01}
    full = {"k": 500, "n": 500, "rate": 1.0, "lo": 0.99, "hi": 1.0}
    return {
        "null_size": {"rates": {"reality_check": r, "spa": dict(z), "romano_wolf_fwer": dict(z), "dsr_false_pass": dict(z)},
                      "psr0_ks_p": 0.5},
        "null_pbo": {"pbo_mean": 0.5, "pbo_ci": [0.48, 0.52], "n": 200},
        "strong_drift": {"pbo_mean": 0.001, "pbo_ci": [0.0, 0.003], "n": 500,
                         "rates": {"dsr_pass": dict(full)}, "oracle": {"maxsr_pass": dict(full)}},
        "power_single": {"rates": {"t_test": dict(full), "psr": dict(full)}},
        "power_multi": {"rates": {"reality_check": dict(full), "spa": dict(full), "romano_wolf": dict(full)},
                        "oracle": {"maxt_power": dict(full)}},
    }


def test_evaluate_reads_thresholds_from_gates_yaml(tmp_path):
    acc = syn.load_acceptance(ROOT / "config" / "gates.yaml")
    assert syn.evaluate(_fake_results(0.04), acc) == []
    fails = syn.evaluate(_fake_results(0.10), acc)
    assert any("reality_check" in f for f in fails)
    # changing the gates file changes the verdict: raise the size bar above 0.10
    p = _acc(tmp_path, max_rejection_rate=0.2)
    acc2 = syn.load_acceptance(p)
    assert syn.evaluate(_fake_results(0.10), acc2) == []
    marg = syn.find_marginal(_fake_results(0.065, hi=0.09), acc)
    assert any("reality_check" in m for m in marg)


def test_no_threshold_literals_in_synthetic_source():
    tree = ast.parse(SYN_SRC.read_text())
    allowed_cls = {"GarchTParams"}  # GARCH alpha=0.05 is a DGP parameter, not a bar
    bad = []

    class V(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            if node.name in allowed_cls:
                return
            self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, float) and node.value in (0.07, 0.80, 0.8, 0.05):
                bad.append(node.lineno)

    V().visit(tree)
    assert not bad, f"threshold-like literals at lines {bad}"


def test_quick_smoke_runs_end_to_end(tmp_path, monkeypatch):
    root = tmp_path / "ledger"
    (root / "returns").mkdir(parents=True)
    monkeypatch.setenv("FIRM_RESEARCH_LEDGER_ROOT", str(root))
    monkeypatch.setenv("FIRM_LEDGER_ALLOW_DIRTY", "1")
    host = Path("/local/store/research-ledger/trials.jsonl")
    before = host.stat().st_size if host.exists() else None
    mod = _load_script()
    rc = mod.main(["--out", str(tmp_path / "out"), "--quick", "--workers", "1"])
    assert rc in (0, 1)
    mds = list((tmp_path / "out").glob("synthetic_*.md"))
    assert len(mds) == 1
    text = mds[0].read_text()
    assert "NON-EVIDENCE" in text
    assert list((tmp_path / "out").glob("synthetic_*.json"))
    after = host.stat().st_size if host.exists() else None
    assert before == after
