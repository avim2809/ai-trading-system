"""P4-02: per-family return streams and the diversification report (synthetic fixtures only)."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from firm.reporting import diversification_report as DR
from firm.reporting.diversification_report import (
    build_report,
    load_gates,
    render_markdown,
    write_report,
)
from firm.validation import diversification as D

REPO = Path(__file__).resolve().parents[1]
T = 1000


def gates(**h4) -> dict:
    h = {
        "enb_method": DR.ENB_METHOD,
        "enb_min_etf": 2.5,
        "enb_min_futures": 3.0,
        "stress_correlation_flag_threshold": 0.7,
        "stress_correlation_rolling_window": 120,
    }
    h.update(h4)
    return {"phase_exits": {"H4": h}}


def idx(n=T):
    return pd.bdate_range("2010-01-04", periods=n)


def indep(n_cols, n=T, seed=0, scales=None, prefix="s"):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_cols))
    if scales is not None:
        x = x * np.asarray(scales)
    return pd.DataFrame(x, index=idx(n), columns=[f"{prefix}{i}" for i in range(n_cols)])


def mask(n=T, frac=0.15, seed=1):
    rng = np.random.default_rng(seed)
    m = np.zeros(n, dtype=bool)
    m[rng.choice(n, int(n * frac), replace=False)] = True
    return pd.Series(m, index=idx(n))


def report(fam=None, ac=None, g=None, **kw):
    fam = indep(4, seed=2, prefix="f") if fam is None else fam
    ac = indep(4, seed=3, prefix="c") if ac is None else ac
    return build_report(fam, ac, ["t1", "t2"], mask(max(len(fam), len(ac))), g or gates(), **kw)


# ---- ENB / PR / MP ----------------------------------------------------------------------------------------------
def test_enb_identity_matrix_equals_n():
    # exactly independent unit-variance streams (identity covariance); sample covariances of random draws have
    # near-degenerate eigenvalues, where the PCA basis is arbitrary and Meucci's ENB is biased DOWN (see report doubts).
    n = 10
    assert D.effective_number_of_bets(np.eye(n), np.ones(n)) == pytest.approx(n, rel=0.05)


def test_sample_enb_of_independent_streams_never_exceeds_n():
    rep = report(ac=indep(6, n=4000, seed=5, prefix="c"))
    assert 1.0 <= rep.enb_asset_class <= 6.0 + 1e-9


def test_enb_perfectly_correlated_equals_one():
    x = indep(1, seed=4)["s0"]
    ac = pd.DataFrame({"a": x, "b": 2 * x, "c": 0.5 * x})
    rep = report(ac=ac)
    assert rep.enb_asset_class == pytest.approx(1.0, abs=1e-6)
    assert not rep.enb_pass


def test_participation_ratio_bounds():
    for seed in range(4):
        rep = report(fam=indep(5, seed=seed, prefix="f"))
        assert 1.0 <= rep.participation_ratio <= 5.0
    x = indep(1, seed=9)["s0"]
    same = pd.DataFrame({"a": x, "b": x, "c": x}, index=x.index)
    assert report(fam=same).participation_ratio == pytest.approx(1.0)


def test_mp_edge_formula():
    rep = report(fam=indep(10, n=1000, seed=6, prefix="f"))
    assert rep.mp_edge == pytest.approx((1 + math.sqrt(10 / 1000)) ** 2) == pytest.approx(1.21)
    assert 0 <= rep.n_eigs_above_mp <= 10


# ---- Fisher z / calm-stress -------------------------------------------------------------------------------------
def test_fisher_z_known_value():
    z, p = D.fisher_z_compare(0.5, 100, 0.3, 100)
    z_hand = (math.atanh(0.5) - math.atanh(0.3)) / math.sqrt(1 / 97 + 1 / 97)
    assert z == pytest.approx(z_hand, abs=1e-6)
    assert p == pytest.approx(math.erfc(abs(z_hand) / math.sqrt(2)), abs=1e-6)
    with pytest.raises(ValueError):
        D.fisher_z_compare(1.0, 100, 0.3, 100)


def _null_frame(rng, k=6, n=400):
    return pd.DataFrame(
        rng.standard_normal((n, k)), index=idx(n), columns=[f"x{i}" for i in range(k)]
    )


def test_calm_stress_bh_null():
    rng = np.random.default_rng(123)
    reps, hits = 200, 0
    m = pd.Series(np.arange(400) % 7 == 0, index=idx(400))  # ~57 stress days
    for _ in range(reps):
        out = D.calm_stress_corr_test(_null_frame(rng), m, fdr_q=0.05)
        hits += bool(out["reject"].any())
    se = math.sqrt(0.05 * 0.95 / reps)
    assert hits / reps <= 0.05 + 3 * se


def test_calm_stress_power():
    rng = np.random.default_rng(321)
    n_calm, n_stress = 1000, 120
    m = pd.Series([False] * n_calm + [True] * n_stress, index=idx(n_calm + n_stress))
    hits = 0
    for _ in range(200):
        a = rng.standard_normal(n_calm + n_stress)
        b = rng.standard_normal(n_calm + n_stress)
        b[n_calm:] = 0.6 * a[n_calm:] + 0.8 * b[n_calm:]
        out = D.calm_stress_corr_test(pd.DataFrame({"a": a, "b": b}, index=m.index), m)
        hits += bool(out["reject"].iloc[0])
    assert hits / 200 >= 0.90
    assert list(out.columns) == ["pair", "r_calm", "r_stress", "z", "p", "p_adj", "reject"]


def test_calm_stress_rejects_short_overlap_and_tiny_regimes():
    df = _null_frame(np.random.default_rng(0), k=2, n=200)
    with pytest.raises(ValueError):
        D.calm_stress_corr_test(df, pd.Series(np.arange(200) % 5 == 0, index=df.index))
    df = _null_frame(np.random.default_rng(0), k=2, n=400)
    with pytest.raises(ValueError):
        D.calm_stress_corr_test(df, pd.Series(np.arange(400) < 2, index=df.index))


def test_bh_adjusted_p_matches_accept_mask():
    p = np.array([0.001, 0.02, 0.03, 0.2, 0.9])
    adj = D._bh_adjusted_p(p)
    assert (adj >= p).all() and (adj <= 1).all()
    np.testing.assert_array_equal(adj <= 0.05, D.bh_adjust(p, 0.05))


# ---- gate ENB ---------------------------------------------------------------------------------------------------
def test_gate_enb_weight_dependent():
    # 4 uncorrelated class P&L streams with variance (= risk) shares 0.85/0.05/0.05/0.05
    shares = np.array([0.85, 0.05, 0.05, 0.05])
    ac = indep(4, n=5000, seed=7, scales=np.sqrt(shares), prefix="c")
    rep = report(fam=indep(4, n=5000, seed=8, prefix="f"), ac=ac)
    assert rep.enb_asset_class < 2.0  # well below the 2.5 bar
    assert not rep.enb_pass
    assert rep.enb_asset_class_corr_entropy == pytest.approx(
        4.0, abs=0.05
    )  # the weightless diagnostic says "diversified"


def test_diversification_ratio_known_value():
    n = 1000
    x = np.tile([1.0, -1.0], n // 2)
    y = np.tile([1.0, 1.0, -1.0, -1.0], n // 4)  # exactly uncorrelated with x, same std
    ac = pd.DataFrame({"a": x, "b": y}, index=idx(n))
    rep = report(fam=indep(3, n=n, seed=1, prefix="f"), ac=ac)
    assert rep.diversification_ratio == pytest.approx(2 / math.sqrt(2), rel=1e-9)


def test_stress_corr_above_0p7_flagged():
    rng = np.random.default_rng(11)
    n = T
    m = mask(n, frac=0.2, seed=3)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    b = np.where(
        m.to_numpy(), 0.9 * a + math.sqrt(1 - 0.81) * b, b
    )  # a,b nearly collinear only in stress
    fam = pd.DataFrame({"a": a, "b": b, "c": c}, index=idx(n))
    rep = build_report(fam, indep(4, seed=3, prefix="c"), ["t"], m, gates())
    flagged = {tuple(sorted(p[:2])): p[2] for p in rep.stress_corr_flags}
    assert ("a", "b") in flagged and flagged[("a", "b")] > 0.7
    assert ("a", "c") not in flagged and ("b", "c") not in flagged
    assert "a" in render_markdown(rep) and "STRESS CORRELATION" in render_markdown(rep)


def test_rolling_diagnostics_shape():
    rep = report(g=gates(stress_correlation_rolling_window=100))
    assert len(rep.rolling_corr) == T
    assert list(rep.rolling_corr.columns) == ["mean_pairwise_corr_family", "enb_asset_class"]
    assert (
        rep.rolling_corr.iloc[:99].isna().all().all()
        and rep.rolling_corr.iloc[99:].notna().all().all()
    )


def test_deferred_rolling_window_is_not_invented():
    g = gates(stress_correlation_rolling_window={"defined_in": "P4-02"})
    rep = report(g=g)
    assert rep.rolling_corr.empty
    assert "not computed" in render_markdown(rep)
    assert not report(g=g, rolling_window=100).rolling_corr.empty


# ---- inputs and wiring ------------------------------------------------------------------------------------------
def test_trial_returns_passed_to_effective_trials(monkeypatch):
    seen = {}

    def fake(df):
        seen["df"] = df
        return {"onc": 1.0, "enb": 2.0, "mp_eigen": 1.0, "raw": float(df.shape[1])}

    monkeypatch.setattr(DR, "effective_n_all", fake)
    fam = indep(4, seed=2, prefix="f")
    rep = report(fam=fam)
    assert seen["df"] is fam and rep.effective_n["enb"] == 2.0


def test_enb_method_read_from_gates_yaml():
    assert report(g=gates()).enb_threshold == 2.5
    with pytest.raises(ValueError):
        report(g=gates(enb_method="corr_entropy"))
    with pytest.raises(ValueError):
        report(g=gates(enb_method=None))


def test_gate_threshold_read_from_gates_yaml(tmp_path):
    ac = indep(4, seed=3, prefix="c")
    base = copy.deepcopy(gates())
    p = tmp_path / "gates.yaml"
    lo, hi = copy.deepcopy(base), copy.deepcopy(base)
    lo["phase_exits"]["H4"]["enb_min_etf"] = 0.5
    hi["phase_exits"]["H4"]["enb_min_etf"] = 50.0
    p.write_text(yaml.safe_dump(lo))
    assert report(ac=ac, g=load_gates(p)).enb_pass
    p.write_text(yaml.safe_dump(hi))
    assert not report(ac=ac, g=load_gates(p)).enb_pass
    assert report(ac=ac, g=lo, instrument_type="futures").enb_threshold == 3.0
    src = Path(DR.__file__).read_text()
    assert not re.search(r"\b2\.5\b|\b3\.0\b|\b0\.7\b", src), (
        "gate thresholds must not be hard-coded in the module"
    )


def test_real_gates_yaml_loads_and_window_is_deferred():
    g = load_gates(REPO / "config" / "gates.yaml")
    rep = report(g=g)
    assert rep.enb_threshold == g["phase_exits"]["H4"]["enb_min_etf"]
    assert rep.rolling_corr.empty  # window deferred in gates.yaml (owner action)


def test_report_requires_trial_ids():
    fam, ac, m = indep(4, seed=2, prefix="f"), indep(4, seed=3, prefix="c"), mask()
    with pytest.raises(ValueError):
        build_report(fam, ac, [], m, gates())
    with pytest.raises(ValueError):
        build_report(fam, None, ["t"], m, gates())
    with pytest.raises(ValueError):
        build_report(fam, ac.iloc[:, :1], ["t"], m, gates())


def test_overlap_below_250_raises():
    with pytest.raises(ValueError):
        report(fam=indep(4, n=200, seed=2, prefix="f"), ac=indep(4, n=200, seed=3, prefix="c"))
    fam = indep(3, n=600, seed=2, prefix="f")
    fam.iloc[:400, 2] = np.nan  # pair (f0,f2) overlaps only 200 days
    with pytest.raises(ValueError):
        build_report(fam, indep(4, n=600, seed=3, prefix="c"), ["t"], mask(600), gates())


def test_family_enb_marked_uninformative_below_three_families():
    rep = report(fam=indep(2, seed=2, prefix="f"))
    assert not rep.family_enb_informative and rep.enb_family <= 2.0 + 1e-9
    assert "uninformative" in render_markdown(rep)
    assert report(fam=indep(4, seed=2, prefix="f")).family_enb_informative


# ---- rendering and output ---------------------------------------------------------------------------------------
def test_markdown_stop_line_only_on_miss():
    miss = render_markdown(
        report(ac=pd.DataFrame({"a": indep(1, seed=4)["s0"], "b": indep(1, seed=4)["s0"]}))
    )
    assert "no post-hoc universe changes" in miss.lower() and "STOP" in miss
    ok = render_markdown(report(g=gates(enb_min_etf=0.5)))
    assert "no post-hoc universe changes" not in ok.lower()
    for text in (miss, ok):
        assert "t1" in text and "t2" in text and "sleeve" not in text.lower()


def test_write_report_outputs_md_and_json(tmp_path):
    rep = report()
    out = write_report(rep, tmp_path / "core_v1")
    md, js = out["markdown"], out["json"]
    assert md.name == "diversification.md" and js.name == "diversification.json"
    data = json.loads(js.read_text())
    assert data["trial_ids"] == ["t1", "t2"] and data["enb_pass"] == bool(rep.enb_pass)
    assert "enb_asset_class" in data


def test_no_live_imports_and_no_sleeve_wording():
    import subprocess
    import sys

    src = REPO / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(src)!r}); import firm.reporting.diversification_report; "
        "bad=[m for m in sys.modules if m.split('.')[:2] in (['firm','live'],['firm','agents'],['firm','api'],"
        "['firm','runtime'])]; print(bad); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "sleeve" not in Path(DR.__file__).read_text().lower()
