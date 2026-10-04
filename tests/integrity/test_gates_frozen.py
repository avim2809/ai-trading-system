"""Integrity tests for the frozen gates (ticket P0-08). OWNER-PROTECTED: agents must not edit this file.

Every expected value is hard-coded here on purpose: if gates.yaml is edited, these tests fail unless a human also edits this
protected file. Source of truth is the signed register docs/gate_deviation_register_2026_10.md.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
GATES = ROOT / "config" / "gates.yaml"
REGISTER = ROOT / "docs" / "gate_deviation_register_2026_10.md"


@pytest.fixture(scope="module")
def gates() -> dict:
    return yaml.safe_load(GATES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def register_text() -> str:
    return REGISTER.read_text(encoding="utf-8")


def _gates_hash_module():
    spec = importlib.util.spec_from_file_location("gates_hash", ROOT / "scripts" / "gates_hash.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_yaml_parses_and_has_required_sections(gates):
    required = {
        "meta", "n_rule", "n_counts", "var_sr_family", "prereg_family", "verdict", "g_research", "robustness_parameters",
        "charter", "g_paper", "p3_10_forward_holdout", "g_live_step", "g_decommission", "phase_exits", "monitoring",
        "p1_08_acceptance", "deferred_values",
    }
    assert required <= set(gates), required - set(gates)
    assert {"dsr", "pbo", "cpcv", "cost_stress", "stress", "robustness", "benchmark", "mechanism"} == set(gates["g_research"])


def test_register_is_signed_and_matches_meta(gates, register_text):
    m = re.search(r"^Signed by: (.+?)\s{2,}Date/time \(UTC, from `date -u`\): (\S+)\s*$", register_text, re.MULTILINE)
    assert m, "signature line not found"
    name, stamp = m.group(1).strip(), m.group(2).strip()
    assert set(name) - {"_"} and set(stamp) - {"_"}, "register is not signed"
    assert gates["meta"]["signed_by"] == name
    assert gates["meta"]["signed_utc"] == stamp


def test_var_sr_drawn_from_family_ledger(gates):
    assert gates["g_research"]["dsr"]["var_sr_rule"] == "length_adjusted_floored_at_grid"
    assert gates["prereg_family"] == ["core_v1"]
    assert set(gates["var_sr_family"]) == {
        "legacy_trend_standalone", "alt_premia_T2", "alt_premia_C1",
        "eodhd_s3_trial_1", "eodhd_s3_trial_2", "eodhd_s3_trial_3", "eodhd_s3_trial_4", "eodhd_s3_trial_5",
        "core_v1_grid",
    }
    assert gates["g_research"]["dsr"]["threshold"] == 0.95
    assert gates["g_research"]["dsr"]["per_period_sharpes_only"] is True


def test_pbo_thresholds(gates):
    pbo = gates["g_research"]["pbo"]
    assert pbo["n_blocks_S"] == 16 and pbo["threshold"] == 0.30
    assert pbo["threshold_when_variants_exceed"] == {"variant_count_threshold": 20, "threshold": 0.20}
    assert pbo["informativeness"]["min_effective_grid_n"] == 4
    assert pbo["informativeness"]["max_median_pairwise_column_corr"] == 0.95
    assert pbo["informativeness"]["uninformative_result"] == "insufficient"
    assert pbo["legacy_cscv_pbo_keeps_S"] == 8


def test_cpcv_paths(gates):
    c = gates["g_research"]["cpcv"]
    assert (c["n_groups"], c["k_test_groups"], c["n_paths"], c["min_positive_paths"], c["embargo_pct"]) == (10, 2, 9, 7, 0.01)
    assert c["n_paths"] == c["k_test_groups"] * 45 // c["n_groups"]  # k * C(10,2) / 10 = 9
    steps = c["in_sample_procedure"]
    assert len(steps) == 4
    assert "training groups only" in steps[1] and "training-group net Sharpe" in steps[2]


def test_cost_stress_stress_and_robustness_constants(gates):
    g = gates["g_research"]
    assert g["cost_stress"]["gate_multiplier"] == 2 and g["cost_stress"]["dsr_min"] == 0.90
    assert set(g["cost_stress"]["scales"]) == {"commission", "fees", "half_spread", "impact", "roll"}
    assert g["stress"]["max_loss_multiple_of_charter_max_dd"] == 1.5
    r = g["robustness"]
    assert (r["perturbation"], r["tolerance"], r["one_sided"], r["pass_if_perturbed_net_sharpe_at_least"]) == (0.25, 0.30, True, 0.70)
    assert r["max_fraction_of_days_at_gross_cap"] == 0.20
    assert r["integer_rounding"] == "half_up_min_2"


def _combine(results: dict, mapping: dict) -> str:
    """Reference implementation of the verdict rule as frozen in gates.yaml (P3-08 gate_report must agree with it)."""
    expected = {"dsr", "pbo", "cpcv", "cost_stress", "stress", "robustness", "benchmark", "mechanism"}
    if set(results) != expected or any(v not in {"pass", "confidence_miss", "point_fail"} for v in results.values()):
        return mapping["missing_or_unmapped_gate_result"]
    if any(v == "point_fail" for v in results.values()):
        return mapping["any_point_fail"]
    if any(v == "confidence_miss" for v in results.values()):
        return mapping["no_point_fail_but_any_confidence_miss"]
    return mapping["all_gates_explicitly_pass"]


def test_tier_mapping(gates):
    m = gates["verdict"]["combination"]
    assert m == {
        "all_gates_explicitly_pass": "A", "any_point_fail": "D",
        "no_point_fail_but_any_confidence_miss": "C", "missing_or_unmapped_gate_result": "D",
    }
    assert gates["verdict"]["paper_eligible_tiers"] == ["A"]
    assert gates["verdict"]["passive_outcome_tiers"] == ["C", "D"]
    names = ["dsr", "pbo", "cpcv", "cost_stress", "stress", "robustness", "benchmark", "mechanism"]
    base = {n: "pass" for n in names}
    assert _combine(base, m) == "A"
    for n in names:                                   # every gate x {C-miss, D-fail} cell
        assert _combine({**base, n: "confidence_miss"}, m) == "C", n
        assert _combine({**base, n: "point_fail"}, m) == "D", n
        assert _combine({**base, n: "point_fail", names[0] if n != names[0] else names[1]: "confidence_miss"}, m) == "D", n
        partial = dict(base); partial.pop(n)
        assert _combine(partial, m) == "D", f"missing {n} must fail closed"
        assert _combine({**base, n: "unmapped"}, m) == "D", f"unmapped {n} must fail closed"
    for n in names:                                   # each gate declares explicit predicates (confidence_miss may be null where N/A)
        g = gates["g_research"][n]
        assert "point_fail" in g and "confidence_miss" in g, n
        assert g["point_fail"], n


def test_n_rule_matches_signed_text(gates, register_text):
    row = next(l for l in register_text.splitlines() if l.startswith("| 2 |"))
    norm = lambda s: re.sub(r"[\s`*_]+", " ", s).lower()
    rule = norm(gates["n_rule"])
    for phrase in ("function of an asset's own past price trend", "futures-curve or yield carry",
                   "cross-sectional momentum (s1, s4)", "raw count"):
        assert phrase in norm(row), f"register row 2 lacks: {phrase}"
        assert phrase in rule, f"n_rule lacks: {phrase}"
    c = gates["n_counts"]
    assert (c["family_provisional"], c["raw_with_estimates"], c["ledgered_only_sensitivity"]) == (31, 463, 210)
    for n in ("31", "463", "210"):
        assert n in norm(row)
    assert "463; 210 ledgered-only" in rule


def test_benchmark_decision_table_complete(gates):
    b = gates["g_research"]["benchmark"]
    assert b["rebalance"] == "annual" and b["sensitivity"] == "monthly" and b["correlation_max"] == 0.30
    assert b["live_alpaca_book_excluded"] is True and b["data"] == "pre_seal_only"
    assert len(b["decision_table"]) == 3
    assert (b["mandatory_power_analysis"]["alpha"], b["mandatory_power_analysis"]["power"]) == (0.05, 0.80)


def test_robustness_parameter_list_complete_and_unexempt(gates):
    expected = {
        "vol_span", "vol_blend_long_weight", "vol_long_window_days", "vol_floor_percentile", "ewmac_fast_spans",
        "ewmac_slow_to_fast_ratio", "forecast_cap", "breakout_lookbacks_N", "breakout_smoothing_fraction_of_N", "fdm_cap",
        "idm_cap", "buffer_fraction", "tau", "gross_cap", "speed_cost_max_fraction", "vol_ewma_span", "max_vol_scale",
        "instrument_risk_cap_multiple",
    }
    params = gates["robustness_parameters"]
    assert set(params) == expected, set(params) ^ expected          # set equality: deleting one fails
    for name, spec in params.items():
        assert "exempt" not in spec and "exempt" not in str(spec).lower(), name
    assert gates["g_research"]["robustness"]["parameters_ref"] == "robustness_parameters"


def test_p1_08_acceptance_frozen(gates):
    a = gates["p1_08_acceptance"]
    assert a["size"] == {"alpha": 0.05, "max_rejection_rate": 0.07, "K": 50, "n_sims": 500,
                         "applies_separately_to": ["reality_check", "spa", "romano_wolf_fwer"]}
    assert a["power"]["single_test"] == {"K": 1, "annualised_sr": 1.0, "years": 10, "min_power": 0.80}
    assert a["power"]["multiple_testing"]["true_strategy_annualised_sr"] == 1.3
    assert a["pbo"]["strong_drift"] == {"N": 30, "annualised_sr": 3.0, "T": 1600, "S": 16, "min_seeds": 20,
                                         "mean_pbo_max_exclusive": 0.1}
    assert a["pbo"]["noise_mean_pbo_range"] == [0.4, 0.6]
    assert a["dsr"]["null_false_pass_max"] == 0.05
    s = a["scenarios"]
    assert s["T"] == 2520 and s["garch"] == {"omega": 5.0e-6, "alpha": 0.05, "beta": 0.90, "nu": 6.0, "mu": 0.0}


def test_g_paper_live_step_decommission(gates):
    p = gates["g_paper"]
    assert (p["min_calendar_months"], p["min_review_events"], p["embargo_weeks_before_start"]) == (6, 26, 2)
    f = p["fidelity"]
    assert (f["daily_return_corr_min"], f["tracking_error_max_fraction_of_tau"], f["window_trading_days"]) == (0.95, 0.25, 63)
    assert (p["realised_cost_max_multiple_of_modelled"], p["cost_test_min_fills"], p["cost_test_min_instruments"]) == (1.5, 30, 10)
    assert p["position_breaks"]["max_unreconciled_trading_days"] == 1
    assert p["max_pre_declared_reevaluation_dates"] == 1
    ls = gates["g_live_step"]
    assert ls["steps_fraction_of_planned_capital"] == [0.25, 0.50, 1.00] and ls["min_months_per_step"] == 3
    assert (ls["dd_reference"], ls["dd_multiple"]) == ("survival_ref", 1.0)
    d = gates["g_decommission"]
    assert d["drawdown_multiple_of_survival_ref"] == 1.5 and len(d["triggers"]) == 4


def test_seal_and_unseal_gates_present(gates):
    h = gates["p3_10_forward_holdout"]
    assert h["min_months_of_post_seal_data"] == 12
    assert h["evaluated_once_per_family"] is True
    assert h["required_before_first_live_step"] is False             # amended by the owner on 2026-10-04 (register row 15)
    assert {"forward_net_sharpe_not_below", "forward_max_dd_within"} == set(h["pass_criteria"])
    assert gates["meta"]["all_data_to_2026_09_30_is_in_sample"] is True


def test_charter_and_survival_references(gates):
    c = gates["charter"]["max_dd_procedure"]
    assert (c["draws"], c["path_length_trading_days"], c["percentile"], c["method"]) == (10000, 2520, 95, "stationary_block_bootstrap")
    assert gates["charter"]["references"]["survival_ref"].startswith("max(bootstrap p95 at 2520 days, 2.5")
    assert gates["charter"]["kelly_tau_bound"]["enforce"] is True
    assert gates["phase_exits"]["H4"]["enb_min_etf"] == 2.5 and gates["phase_exits"]["H4"]["enb_min_futures"] == 3.0


def test_gates_hash_matches_signoff():
    gh = _gates_hash_module()
    recorded = gh.recorded_hash(REGISTER)
    if recorded is None:
        pytest.skip("gates hash not recorded in the register yet: owner step `python scripts/gates_hash.py --record` pending")
    assert gh.canonical_hash(GATES) == recorded, "config/gates.yaml changed after sign-off (new signed register version needed)"


def test_gates_hash_matches_prereg(gates):
    gh = _gates_hash_module()
    current = gh.canonical_hash(GATES)
    for prereg in sorted((ROOT / "research" / "preregistration").glob("*.yaml")) if (ROOT / "research" / "preregistration").is_dir() else []:
        data = yaml.safe_load(prereg.read_text(encoding="utf-8")) or {}
        if "gates_sha256" in data:
            assert data["gates_sha256"] == current, prereg.name


def test_deferred_values_are_not_invented(gates):
    # Values no ticket or register row fixed must be left to the named ticket, never guessed here.
    assert gates["robustness_parameters"]["tau"]["value"] is None
    assert gates["robustness_parameters"]["vol_ewma_span"]["value"] is None
    assert "defined_in" in gates["g_research"]["stress"]["min_active_fraction"]
    assert len(gates["deferred_values"]) >= 6
