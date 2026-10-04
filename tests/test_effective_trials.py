"""Effective number of trials and the gate N rule (P1-03, OD-09)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from firm.validation import effective_trials as ET


def blocks(k: int, size: int = 6, T: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((T, 1))
    b = np.repeat(rng.standard_normal((T, k)), size, axis=1)
    e = rng.standard_normal((T, k * size))
    x = np.sqrt(0.05) * g + np.sqrt(0.85) * b + np.sqrt(0.10) * e  # within 0.9, between 0.05
    return pd.DataFrame(x, columns=[f"t{i}" for i in range(k * size)])


@pytest.mark.parametrize("k", [3, 5, 8])
def test_k_known_clusters_recovered_within_20pct(k):
    R = blocks(k)
    assert ET.effective_n(R, "onc") == k
    assert ET.effective_n(R, "mp_eigen") == k
    pr = ET.effective_n_all(R)
    assert 0.8 * k <= pr["mp_eigen"] <= 1.2 * k
    er = ET.effective_n(R, "enb")  # effective_rank: noise eigenvalues add entropy, ~1.5k here
    assert 1.2 * k <= er <= 1.8 * k
    from firm.validation.diversification import participation_ratio

    assert 0.8 * k <= participation_ratio(R.corr().to_numpy()) <= 1.2 * k
    assert pr["raw"] == 6 * k


def test_onc_clusters_are_the_blocks():
    cl = ET.onc_clusters(blocks(3).corr())
    assert {frozenset(c) for c in cl} == {frozenset(f"t{i}" for i in range(b * 6, b * 6 + 6)) for b in range(3)}


def test_identity_returns_n_equals_columns():
    R = pd.DataFrame(np.random.default_rng(3).standard_normal((2000, 20)))
    assert ET.effective_n(R, "enb") == pytest.approx(20, rel=0.15)
    from firm.validation.diversification import participation_ratio

    assert participation_ratio(R.corr().to_numpy()) == pytest.approx(20, rel=0.15)


def test_identical_series_give_one():
    s = np.random.default_rng(4).standard_normal(500)
    R = pd.DataFrame({f"c{i}": s for i in range(6)})
    from firm.validation.diversification import effective_rank, participation_ratio

    assert ET.effective_n(R, "enb") == pytest.approx(1.0)
    assert ET.effective_n(R, "mp_eigen") == 1
    assert ET.effective_n(R, "onc") == 1
    assert participation_ratio(R.corr().to_numpy()) == pytest.approx(1.0)
    assert effective_rank(R.corr().to_numpy()) == pytest.approx(1.0)


def test_constant_columns_dropped_but_counted_raw():
    R = blocks(3)
    R["const"] = 0.0
    out = ET.effective_n_all(R)
    assert out["raw"] == 19 and out["onc"] == 3


def test_overlap_rule():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(400)
    R = pd.DataFrame({"a": a, "b": a + 0.5 * rng.standard_normal(400), "c": rng.standard_normal(400)})
    R.loc[:150, "b"] = np.nan  # 249 overlapping rows with a
    assert np.isnan(ET.pairwise_corr(R)["a"]["b"])
    R2 = R.copy()
    R2.loc[150, "b"] = a[150]
    R2.loc[150, "b"] = a[150] + 0.1
    assert not np.isnan(ET.pairwise_corr(R2)["a"]["b"])  # 250 rows
    assert not np.isnan(ET.pairwise_corr(R)["a"]["c"])


def test_psd_projection_logged(caplog):
    bad = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    with caplog.at_level(logging.INFO, logger=ET.log.name):
        out = ET._nearest_psd_corr(bad)
    assert np.linalg.eigvalsh(out).min() >= -1e-10 and np.allclose(np.diag(out), 1.0)
    assert any("PSD" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ gate_n

def _ledger(n_legacy=210, n_est=253, n_ret=12, n_failed=2, n_unreg=5, n_api=3) -> pd.DataFrame:
    rows = []

    def add(n, **kw):
        for _ in range(n):
            base = {"trial_id": f"t{len(rows)}", "mode": "exploratory", "status": "completed",
                    "returns_path": None, "count_is_estimate": False, "n_variants": 1, "source_file": None}
            rows.append({**base, **kw})

    add(1, mode="legacy", n_variants=n_legacy, source_file="docs/x.json")
    add(1, mode="legacy", n_variants=n_est, count_is_estimate=True, source_file="census.csv")
    for i in range(n_ret):
        add(1, returns_path=f"returns/ret{i}.parquet")
    add(n_failed, status="failed")
    add(n_unreg, mode="unregistered")
    add(n_api, mode="unregistered", source_file="inbox/api-1.jsonl")
    return pd.DataFrame(rows)


def _rets(n=12) -> pd.DataFrame:
    return pd.DataFrame(np.random.default_rng(6).standard_normal((300, n)), columns=[f"ret{i}" for i in range(n)])


def test_gate_n_arithmetic():
    tc = ET.gate_n(_ledger(), _rets())
    assert tc.raw_returns_bearing == 12 and tc.legacy_ledgered == 210 and tc.legacy_estimate == 253
    assert tc.failed_or_no_returns == 2 and tc.unregistered == 5 and tc.api == 3
    assert tc.gate_n == max(tc.effective_onc, tc.effective_enb, tc.effective_mp, 12) + 210 + 253 + 2 + 5 + 3
    assert tc.gate_n >= tc.raw_returns_bearing
    assert tc.sensitivity_ledger_only == 210 + 12 + 2 + 5 + 3
    total = (tc.raw_returns_bearing + tc.legacy_ledgered + tc.legacy_estimate + tc.failed_or_no_returns
             + tc.unregistered + tc.api)
    assert total == int(_ledger()["n_variants"].sum())  # every row in exactly one bucket


def test_gate_n_on_signed_census_fixture_at_least_460():
    tc = ET.gate_n(_ledger(n_ret=0, n_failed=0, n_unreg=0, n_api=0), _rets(0))
    assert tc.gate_n >= 210 + 253 and tc.gate_n == 463
    assert tc.sensitivity_ledger_only == 210


def test_failed_exploratory_row_increases_gate_n_by_one():
    base = _ledger()
    extra = pd.concat([base, pd.DataFrame([{**base.iloc[2].to_dict(), "trial_id": "z", "returns_path": None,
                                            "status": "failed"}])], ignore_index=True)
    assert ET.gate_n(extra, _rets()).gate_n == ET.gate_n(base, _rets()).gate_n + 1


def test_returns_bearing_unregistered_row_not_double_counted():
    base = _ledger(n_ret=0, n_failed=0, n_unreg=0, n_api=0)
    row = {**base.iloc[0].to_dict(), "trial_id": "u", "mode": "unregistered", "n_variants": 1,
           "count_is_estimate": False, "returns_path": "returns/u.parquet"}
    tc = ET.gate_n(pd.concat([base, pd.DataFrame([row])], ignore_index=True), _rets(1).set_axis(["u"], axis=1))
    assert tc.raw_returns_bearing == 1 and tc.unregistered == 0 and tc.gate_n == 463 + 1


def test_never_returns_smaller_than_raw_in_gate():
    for k in (1, 3, 8):
        R = blocks(k)
        led = pd.DataFrame({"trial_id": list(R.columns), "mode": "exploratory", "status": "completed",
                            "returns_path": "r", "count_is_estimate": False, "n_variants": 1,
                            "source_file": None})
        tc = ET.gate_n(led, R)
        assert tc.gate_n >= len(R.columns) > tc.effective_onc


def test_empty_ledger():
    tc = ET.gate_n(pd.DataFrame(columns=["mode", "status", "returns_path", "count_is_estimate", "n_variants",
                                         "source_file"]), pd.DataFrame())
    assert tc.gate_n == 0
