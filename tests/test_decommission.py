"""P5-01: G-DECOMMISSION combiner truth table and boundaries (synthetic inputs only)."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from firm.lifecycle import decommission as dc
from firm.lifecycle.gates import load_gates_config
from firm.risk.kill_switch import KillSwitchConfig

GATES = load_gates_config(Path(__file__).resolve().parents[1] / "config" / "gates.yaml")
REF = 0.12


def inp(dd=0.0, cusum=False, fid=0, mech=False, ref=REF):
    return dc.DecommissionInputs(dd, ref, cusum, fid, mech)


@pytest.mark.parametrize(("dd_hit", "cusum", "fid_hit", "mech"), list(itertools.product([False, True], repeat=4)))
def test_truth_table_all_16_combinations(dd_hit, cusum, fid_hit, mech):
    i = inp(dd=0.2 if dd_hit else 0.01, cusum=cusum, fid=2 if fid_hit else 1, mech=mech)
    n = dd_hit + cusum + fid_hit + mech
    assert dc.decommission_decision(i, GATES) == ("ok" if n == 0 else "probation" if n == 1 else "zero_weight")


def test_dd_boundary_exactly_1_5x_not_triggered():
    assert dc.decommission_decision(inp(dd=1.5 * REF), GATES) == "ok"
    assert dc.decommission_decision(inp(dd=1.5 * REF + 1e-9), GATES) == "probation"


def test_fidelity_needs_two_consecutive_months():
    assert dc.decommission_decision(inp(fid=1), GATES) == "ok"
    assert dc.decommission_decision(inp(fid=2), GATES) == "probation"
    assert dc.decommission_decision(inp(fid=5), GATES) == "probation"


def test_survival_reference_uses_max_of_p95_and_2p5_tau():
    # p95 below 2.5 tau: reference is 0.10, so dd 0.16 (> 1.5 x 0.10 = 0.15) triggers; with ref=p95=0.05 it would trigger too,
    # but dd 0.14 triggers only against 0.05 and must NOT against the max-based 0.10.
    lucky = KillSwitchConfig(charter_dd_bootstrap_p95=0.05, tau=0.04)
    assert lucky.survival_ref == pytest.approx(0.10)
    assert dc.decommission_decision(inp(dd=0.14, ref=lucky.survival_ref), GATES) == "ok"
    assert dc.decommission_decision(inp(dd=0.16, ref=lucky.survival_ref), GATES) == "probation"
    wide = KillSwitchConfig(charter_dd_bootstrap_p95=0.30, tau=0.04)
    assert dc.decommission_decision(inp(dd=0.4, ref=wide.survival_ref), GATES) == "ok"


def test_multiple_comes_from_gates_yaml(tmp_path):
    import yaml

    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "gates.yaml").read_text())
    raw["g_decommission"]["drawdown_multiple_of_survival_ref"] = 3.0
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert dc.decommission_decision(inp(dd=0.2), load_gates_config(p)) == "ok"
    assert dc.decommission_decision(inp(dd=0.2), GATES) == "probation"
