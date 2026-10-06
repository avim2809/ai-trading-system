"""G-RESEARCH verdict logic (ticket P3-08): every threshold is read from a gates dict (the frozen config/gates.yaml, or a modified
copy), never hard-coded. Synthetic inputs only; no data, no ledger on the host."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from firm.reporting import gate_report as GR
from firm.validation.sharpe_stats import length_adjusted_var_sr

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def gates() -> dict:
    return yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())


def good_results() -> dict:
    return {
        "dsr": {"sharpe": 0.05, "dsr": 0.97, "n_gate": 31, "var_sr": 1e-4},
        "pbo": {"pbo": 0.10, "prob_oos_loss": 0.05, "selected_median_oos_sharpe": 0.04, "grid_n": 12, "effective_grid_n": 6.0,
                "median_pairwise_corr": 0.80, "family_variant_count": 31},
        "cpcv": {"path_sharpes": [0.04, 0.05, 0.03, 0.06, 0.02, 0.05, 0.04, 0.03, 0.05]},
        "cost_stress": {"net_sharpe": {"1": 0.05, "2": 0.04, "3": 0.03}, "dsr": {"1": 0.97, "2": 0.93, "3": 0.9}},
        "stress": {"min_active_fraction": 0.5, "episodes": [
            {"name": "GFC", "status": "ok", "max_drawdown": 0.10, "reference_max_dd": 0.20, "n_days": 100, "n_active_instruments": 10},
            {"name": "COVID", "status": "ok", "max_drawdown": 0.05, "reference_max_dd": 0.10, "n_days": 50, "n_active_instruments": 14}]},
        "robustness": {"chosen_net_sharpe": 0.05,
                       "perturbations": [{"param": "vol_span", "direction": "-25%", "net_sharpe": 0.045},
                                         {"param": "vol_span", "direction": "+25%", "net_sharpe": 0.06}],
                       "unassessed": [], "gross_cap_bound_share": 0.05},
        "benchmark": {"sharpe_candidate": 0.60, "sharpe_bm2": 0.40, "correlation": 0.6, "margin": 0.10, "rationale_claimed": False},
        "mechanism": {"ok": True, "detail": "charter precedes the first core_v1 row"},
    }


def tier(results, gates, enb_ok=True):
    outs = GR.evaluate_g_research(results, gates)
    return GR.combine_tier(outs, enb_ok), outs


def by_id(outs, i):
    return next(o for o in outs if o.test_id == f"G-RESEARCH-{i}")


def test_all_pass_is_tier_a(gates):
    t, outs = tier(good_results(), gates)
    assert [o.status for o in outs] == ["pass"] * 8 and t == "A"
    assert [o.test_id for o in outs] == [f"G-RESEARCH-{i}" for i in range(1, 9)]


def test_one_point_failure_is_tier_d(gates):
    r = good_results()
    r["cpcv"]["path_sharpes"] = [-0.01] * 5 + [0.02] * 4  # median <= 0
    t, outs = tier(r, gates)
    assert by_id(outs, 3).status == "fail" and t == "D"


def test_confidence_only_miss_is_tier_c(gates):
    r = good_results()
    r["dsr"]["dsr"] = 0.90
    t, outs = tier(r, gates)
    assert by_id(outs, 1).status == "insufficient" and t == "C"


def test_nonpositive_sharpe_is_point_fail(gates):
    r = good_results()
    r["dsr"]["sharpe"] = -0.001
    assert by_id(tier(r, gates)[1], 1).status == "fail"


def test_enb_miss_is_tier_d_with_reason(gates):
    outs = GR.evaluate_g_research(good_results(), gates)
    assert GR.combine_tier(outs, enb_ok=False) == "D"
    assert GR.tier_reason(outs, enb_ok=False) == "H4 ENB miss"


def test_threshold_is_read_from_gates_not_hard_coded(gates):
    r = good_results()
    assert tier(r, gates)[0] == "A"
    g2 = copy.deepcopy(gates)
    g2["g_research"]["dsr"]["threshold"] = 0.99
    assert tier(r, g2)[0] == "C"
    g3 = copy.deepcopy(gates)
    g3["g_research"]["cpcv"]["min_positive_paths"] = 9
    r2 = good_results()
    r2["cpcv"]["path_sharpes"][0] = -0.01
    assert tier(r2, gates)[0] == "A" and tier(r2, g3)[0] == "C"
    g4 = copy.deepcopy(gates)
    g4["g_research"]["cost_stress"]["dsr_min"] = 0.95
    assert tier(r, g4)[0] == "C"


def test_missing_gate_result_fails_closed(gates):
    r = good_results()
    del r["mechanism"]
    t, outs = tier(r, gates)
    assert by_id(outs, 8).status == "fail" and t == "D"
    assert GR.combine_tier(outs[:7], True) == "D"  # an outcome list lacking a gate is Tier D too


def test_unknown_status_fails_closed():
    outs = [GR.TestOutcome(f"G-RESEARCH-{i}", {}, "pass", "") for i in range(1, 8)]
    outs.append(GR.TestOutcome("G-RESEARCH-8", {}, "bogus", ""))
    assert GR.combine_tier(outs, True) == "D"


# ---- gate 2 -----------------------------------------------------------------------------------------------------------------
def test_pbo_high_but_oos_loss_low_is_tier_c_not_d(gates):
    r = good_results()
    r["pbo"].update(pbo=0.6, prob_oos_loss=0.3)
    t, outs = tier(r, gates)
    assert by_id(outs, 2).status == "insufficient" and t == "C"


def test_pbo_point_fail_conjunction(gates):
    r = good_results()
    r["pbo"].update(pbo=0.6, prob_oos_loss=0.6)
    assert by_id(tier(r, gates)[1], 2).status == "fail"
    r = good_results()
    r["pbo"]["selected_median_oos_sharpe"] = -0.01
    assert by_id(tier(r, gates)[1], 2).status == "fail"


def test_pbo_threshold_tightens_above_twenty_variants(gates):
    r = good_results()
    r["pbo"].update(pbo=0.25, family_variant_count=31)  # 0.30 would pass; 0.20 applies above 20 variants
    assert by_id(tier(r, gates)[1], 2).status == "insufficient"
    r["pbo"]["family_variant_count"] = 12
    assert by_id(tier(r, gates)[1], 2).status == "pass"


def test_pbo_grid_n_at_most_three_is_uninformative_never_pass(gates):
    r = good_results()
    r["pbo"].update(grid_n=3, effective_grid_n=3.0)
    o = by_id(tier(r, gates)[1], 2)
    assert o.status == "insufficient" and "uninformative" in o.reason


def test_collinear_grid_makes_pbo_uninformative_not_a_failure(gates):
    r = good_results()
    r["pbo"].update(grid_n=12, effective_grid_n=1.0, median_pairwise_corr=0.999, pbo=0.9, prob_oos_loss=0.9)
    t, outs = tier(r, gates)
    assert by_id(outs, 2).status == "insufficient" and t == "C"
    r["pbo"].update(effective_grid_n=8.0, median_pairwise_corr=0.97)  # correlation alone is enough
    assert by_id(tier(r, gates)[1], 2).status == "insufficient"


# ---- gate 3, 4 -------------------------------------------------------------------------------------------------------------
def test_cpcv_fewer_than_seven_positive_paths_is_confidence_miss(gates):
    r = good_results()
    r["cpcv"]["path_sharpes"] = [0.05] * 6 + [-0.001] * 3  # median positive, 6 of 9 positive
    assert by_id(tier(r, gates)[1], 3).status == "insufficient"


def test_cpcv_wrong_path_count_is_not_a_pass(gates):
    r = good_results()
    r["cpcv"]["path_sharpes"] = [0.05] * 8
    assert by_id(tier(r, gates)[1], 3).status != "pass"


def test_cost_stress_2x_rules(gates):
    r = good_results()
    r["cost_stress"]["net_sharpe"]["2"] = -0.001
    assert by_id(tier(r, gates)[1], 4).status == "fail"
    r = good_results()
    r["cost_stress"]["dsr"]["2"] = 0.85
    assert by_id(tier(r, gates)[1], 4).status == "insufficient"


# ---- gate 5 ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("status", ["unusable", "not_applicable", "low_coverage"])
def test_non_ok_stress_episode_is_insufficient(gates, status):
    r = good_results()
    r["stress"]["episodes"][0]["status"] = status
    r["stress"]["episodes"][0]["max_drawdown"] = 0.0
    t, outs = tier(r, gates)
    assert by_id(outs, 5).status == "insufficient" and t == "C"


def test_stress_breach_is_point_fail_using_gates_multiple(gates):
    r = good_results()
    r["stress"]["episodes"][1]["max_drawdown"] = 0.16  # 1.5 x 0.10 = 0.15
    assert by_id(tier(r, gates)[1], 5).status == "fail"
    g = copy.deepcopy(gates)
    g["g_research"]["stress"]["max_loss_multiple_of_charter_max_dd"] = 2.0
    assert by_id(tier(r, g)[1], 5).status == "pass"


def test_unfrozen_min_active_fraction_is_never_a_pass(gates):
    r = good_results()
    r["stress"]["min_active_fraction"] = None
    o = by_id(tier(r, gates)[1], 5)
    assert o.status == "insufficient" and "min_active_fraction" in o.reason


# ---- gate 6 ----------------------------------------------------------------------------------------------------------------
def test_robustness_breach_is_one_sided(gates):
    r = good_results()
    r["robustness"]["perturbations"][0]["net_sharpe"] = 0.034  # < 0.7 * 0.05
    assert by_id(tier(r, gates)[1], 6).status == "fail"
    r["robustness"]["perturbations"][0]["net_sharpe"] = 0.2   # an improvement is not a breach
    assert by_id(tier(r, gates)[1], 6).status == "pass"


def test_gross_cap_share_above_ceiling_fails(gates):
    r = good_results()
    r["robustness"]["gross_cap_bound_share"] = 0.21
    assert by_id(tier(r, gates)[1], 6).status == "fail"


def test_unassessed_parameters_are_insufficient(gates):
    r = good_results()
    r["robustness"]["unassessed"] = ["vol_ewma_span"]
    o = by_id(tier(r, gates)[1], 6)
    assert o.status == "insufficient" and "vol_ewma_span" in o.reason


# ---- gate 7: one case per decision-table row --------------------------------------------------------------------------------
def g7(gates, cand, bm2, corr, margin=0.10, rationale=False):
    r = good_results()
    r["benchmark"].update(sharpe_candidate=cand, sharpe_bm2=bm2, correlation=corr, margin=margin, rationale_claimed=rationale)
    return by_id(tier(r, gates)[1], 7)


def test_gate7_beats_by_margin_passes(gates):
    assert g7(gates, 0.6, 0.4, 0.8).status == "pass"


def test_gate7_beats_by_less_than_margin(gates):
    assert g7(gates, 0.45, 0.4, 0.8).status == "insufficient"                      # corr > 0.3 -> Tier C
    assert g7(gates, 0.45, 0.4, 0.2, rationale=True).status == "pass"              # corr <= 0.3 with rationale
    assert g7(gates, 0.45, 0.4, 0.2, rationale=False).status == "insufficient"     # no rationale: still C


def test_gate7_fails_point_estimate(gates):
    assert g7(gates, 0.3, 0.4, 0.8).status == "fail"                               # corr > 0.3 -> Tier D
    assert g7(gates, 0.3, 0.4, 0.2, rationale=True).status == "pass"
    assert g7(gates, 0.3, 0.4, 0.2, rationale=False).status == "fail"
    assert g7(gates, 0.4, 0.4, 0.8).status == "fail"                               # a tie is not "beats"


def test_gate7_without_a_power_analysis_cannot_pass_on_a_thin_margin(gates):
    assert g7(gates, 0.6, 0.4, 0.8, margin=None).status == "insufficient"
    assert g7(gates, 0.6, 0.4, 0.8, margin=float("nan")).status == "insufficient"


def test_gate7_correlation_limit_comes_from_gates(gates):
    g = copy.deepcopy(gates)
    g["g_research"]["benchmark"]["correlation_max"] = 0.9
    r = good_results()
    r["benchmark"].update(sharpe_candidate=0.3, sharpe_bm2=0.4, correlation=0.8, rationale_claimed=True)
    assert by_id(tier(r, g)[1], 7).status == "pass" and by_id(tier(r, gates)[1], 7).status == "fail"


def test_gate8_mechanism(gates):
    r = good_results()
    r["mechanism"]["ok"] = False
    assert by_id(tier(r, gates)[1], 8).status == "fail"


# ---- Kelly ---------------------------------------------------------------------------------------------------------------
def test_kelly_bound_breach_is_tier_d(gates):
    r = good_results()
    ok = GR.evaluate_kelly_bound({"tau": 0.09, "deflated_sharpe_annual": 0.9}, gates)       # max_tau = 0.5*0.5*0.9 = 0.225
    bad = GR.evaluate_kelly_bound({"tau": 0.09, "deflated_sharpe_annual": 0.3}, gates)      # 0.075 < 0.09
    outs = GR.evaluate_g_research(r, gates)
    assert ok.status == "pass" and bad.status == "fail"
    assert GR.combine_tier([*outs, ok], True) == "A" and GR.combine_tier([*outs, bad], True) == "D"
    assert GR.evaluate_kelly_bound({"tau": 0.09, "deflated_sharpe_annual": None}, gates).status == "fail"


# ---- var_sr behavioural test (P0-08 binding) --------------------------------------------------------------------------------
def test_short_legacy_trials_do_not_inflate_the_length_adjusted_var_sr():
    rng = np.random.default_rng(7)
    t_short, t_cand, k = 250, 8000, 8
    adj, raw = [], []
    for _ in range(300):
        legacy = rng.normal(0.0, 1 / np.sqrt(t_short), k)
        grid = rng.normal(0.0, 1 / np.sqrt(t_cand), 4)
        srs = np.concatenate([legacy, grid])
        n_obs = [t_short] * k + [t_cand] * 4
        adj.append(GR.gate_var_sr(srs, n_obs, t_cand, grid_sharpes=grid))
        raw.append(float(np.var(srs, ddof=1)))
    assert np.mean(adj) < 0.25 * np.mean(raw)                 # the unadjusted variance is dominated by the short trials' noise
    # clipping var(SR_i) - mean(sampling_var) at 0 leaves a small positive bias of order sd(var estimate) ~ 6e-4 here; the
    # adjusted value must stay an order of magnitude under the raw 4e-3 and near the 1/T_cand null (1.25e-4) plus that clip bias
    assert np.mean(adj) < 1e-3


def test_gate_var_sr_is_floored_at_the_grid_variance():
    grid = np.array([0.0, 0.02, -0.02, 0.01])
    v = GR.gate_var_sr(np.array([0.001, 0.0009, 0.0011]), 5000, 5000, grid_sharpes=grid)
    assert v >= float(np.var(grid, ddof=1)) - 1e-15


# ---- provenance ------------------------------------------------------------------------------------------------------------
def ledger_frame(rows):
    return pd.DataFrame(rows, columns=["trial_id", "mode", "status", "family"])


def test_verify_provenance_ok_and_raises_on_unregistered_row():
    led = ledger_frame([("a", "registered", "completed", "core_v1"), ("b", "registered", "completed", "core_v1")])
    GR.verify_provenance({"trial_ids": ["a", "b"]}, led)
    with pytest.raises(GR.ProvenanceError, match="not in the ledger"):
        GR.verify_provenance({"trial_ids": ["a", "zzz"]}, led)
    led2 = ledger_frame([("a", "registered", "completed", "core_v1"), ("b", "exploratory", "completed", "core_v1")])
    with pytest.raises(GR.ProvenanceError, match="not registered"):
        GR.verify_provenance({"trial_ids": ["a", "b"]}, led2)
    led3 = ledger_frame([("a", "registered", "failed", "core_v1")])
    with pytest.raises(GR.ProvenanceError, match="not completed"):
        GR.verify_provenance({"trial_ids": ["a"]}, led3)
    with pytest.raises(GR.ProvenanceError):
        GR.verify_provenance({"trial_ids": []}, led)


# ---- rendering -------------------------------------------------------------------------------------------------------------
def test_render_report_has_every_row_tier_and_declarations(gates):
    outs = GR.evaluate_g_research(good_results(), gates)
    outs.append(GR.evaluate_kelly_bound({"tau": 0.09, "deflated_sharpe_annual": 0.9}, gates))
    meta = {"data_snapshot_id": "snap123", "n_gate": 31, "n_raw": 463, "var_sr": 1e-4, "var_sr_source": "length-adjusted",
            "enb": 3.1, "enb_threshold": 2.5, "tier_label": "family-N", "code_commit": "abc", "prereg_hash": "p" * 8}
    md = GR.render_report(outs, "A", meta, {"BM2 annual": {"sharpe": 0.4}, "core_only_100": {"sharpe": 0.35}})
    for i in range(1, 9):
        assert f"G-RESEARCH-{i}" in md
    assert "Tier A" in md and "family-N" in md and "snap123" in md
    assert "all data to 2026-09-30 is in-sample; no post-seal data examined" in md
    assert "BM2 annual" in md and "core_only_100" in md
    md_d = GR.render_report(outs, "D", {**meta, "tier_reason": "H4 ENB miss"}, {})
    assert "Tier D" in md_d and "H4 ENB miss" in md_d and "passive" in md_d.lower()
