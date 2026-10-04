"""P4-01 handcrafted and HRP instrument weights (synthetic fixtures only)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.agents.analysts import hrp_signal_weights  # test-only cross-check import
from firm.portfolio.weights import compare_weights, handcraft_weights, hrp_weights

SRC = Path(__file__).resolve().parents[1] / "src"

TREE = {
    "equity": {"us": ["A", "B", "C"], "intl": ["D"]},
    "bond": {"gov": ["E", "F"]},
    "real": {"gold": ["G"]},
}


def _cov(names, sds, corr=None):
    n = len(names)
    corr = np.eye(n) if corr is None else np.asarray(corr)
    return pd.DataFrame(np.outer(sds, sds) * corr, index=names, columns=names)


def test_handcraft_sums_to_one():
    w = handcraft_weights(TREE)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert list(w.index) == list("ABCDEFG")


def test_handcraft_equal_class_risk():
    w = handcraft_weights(TREE)
    assert w[["A", "B", "C", "D"]].sum() == pytest.approx(1 / 3)
    assert w[["E", "F"]].sum() == pytest.approx(1 / 3)
    assert w["G"] == pytest.approx(1 / 3)
    assert w["A"] == pytest.approx(1 / 3 / 2 / 3) and w["D"] == pytest.approx(1 / 3 / 2)


def test_handcraft_deterministic():
    pd.testing.assert_series_equal(handcraft_weights(TREE), handcraft_weights(TREE))


def test_handcraft_rejects_bad_trees():
    for bad in ({}, {"x": {}}, {"x": {"s": []}}, {"x": {"s": ["A"]}, "y": {"t": ["A"]}}):
        with pytest.raises(ValueError):
            handcraft_weights(bad)


def test_hrp_known_two_block():
    names = ["a", "b", "c", "d"]
    sds = np.array([0.01, 0.02, 0.03, 0.05])
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = 0.5
    corr[2, 3] = corr[3, 2] = 0.3
    cov = _cov(names, sds, corr)
    w = hrp_weights(cov)

    def ivp(i, j):
        sub = cov.to_numpy()[np.ix_([i, j], [i, j])]
        iv = 1 / np.diag(sub)
        iv /= iv.sum()
        return iv, float(iv @ sub @ iv)

    iv_l, v_l = ivp(0, 1)
    iv_r, v_r = ivp(2, 3)
    alpha = 1 - v_l / (v_l + v_r)
    assert w[["a", "b"]].sum() == pytest.approx(alpha, abs=1e-12)
    np.testing.assert_allclose(w[["a", "b"]] / w[["a", "b"]].sum(), iv_l, atol=1e-12)
    np.testing.assert_allclose(w[["c", "d"]] / w[["c", "d"]].sum(), iv_r, atol=1e-12)
    # within-block weights proportional to inverse variance (block members use 2-asset bisection)
    inv_var = 1 / np.diag(cov.to_numpy())
    assert w["a"] / w["b"] == pytest.approx(
        (1 - v_a_over(cov, 0, 1)) / v_a_over(cov, 0, 1) if False else inv_var[0] / inv_var[1], rel=1e-9
    )


def v_a_over(cov, i, j):  # helper kept trivial: unused branch above
    return 0.5


def test_hrp_rank_deficient_cov_returns_valid_weights():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(size=(4, 8)), columns=list("abcdefgh"))  # T=4 < N=8
    w = hrp_weights(r.cov())
    assert np.linalg.matrix_rank(r.cov().to_numpy()) < 8
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0, abs=1e-12) and np.isfinite(w).all()


@pytest.mark.parametrize("seed", range(10))
def test_hrp_weights_nonnegative_sum_one(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 14))
    r = pd.DataFrame(rng.normal(size=(60, n)) * rng.uniform(0.5, 3, n), columns=[f"x{i}" for i in range(n)])
    w = hrp_weights(r.cov())
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0, abs=1e-12)
    assert list(w.index) == list(r.columns)


def test_hrp_single_instrument():
    w = hrp_weights(pd.DataFrame([[0.04]], index=["x"], columns=["x"]))
    assert w.tolist() == [1.0]


@pytest.mark.parametrize("seed", range(5))
def test_hrp_matches_legacy_signal_hrp(seed):
    rng = np.random.default_rng(100 + seed)
    n = 9
    common = rng.normal(size=(300, 1))
    R = pd.DataFrame(
        0.6 * common * rng.uniform(0, 1, n) + rng.normal(size=(300, n)) * rng.uniform(0.5, 2, n),
        columns=[f"s{i}" for i in range(n)],
    )
    legacy, _ = hrp_signal_weights(R)
    new = hrp_weights(R.cov(ddof=1))
    np.testing.assert_allclose(new.to_numpy(), legacy.reindex(new.index).to_numpy(), atol=1e-9)


def test_hrp_rejects_nan_asymmetric_nonpositive_diag():
    good = _cov(list("abc"), [0.01, 0.02, 0.03])
    nan = good.copy()
    nan.iloc[0, 1] = nan.iloc[1, 0] = np.nan
    asym = good.copy()
    asym.iloc[0, 1] = 1e-4
    zero = good.copy()
    zero.iloc[1, 1] = 0.0
    neg = good.copy()
    neg.iloc[2, 2] = -1.0
    for bad in (nan, asym, zero, neg):
        with pytest.raises(ValueError):
            hrp_weights(bad)
    notpsd = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    with pytest.raises(ValueError):
        hrp_weights(notpsd)


def test_compare_weights():
    h = handcraft_weights({"x": {"s": ["a", "b"]}, "y": {"t": ["c"]}})
    p = pd.Series({"a": 0.2, "b": 0.3, "c": 0.5})
    df = compare_weights(h, p)
    assert list(df.columns) == ["handcrafted", "hrp", "diff"]
    assert df["diff"].sum() == pytest.approx(0.0, abs=1e-12)
    assert df.loc["a", "diff"] == pytest.approx(0.2 - 0.25)


def test_no_firm_agents_import():
    code = "import sys, firm.portfolio.weights; print('AGENTS:' + str(any(m == 'firm.agents' or m.startswith('firm.agents.') for m in sys.modules)))"
    out = subprocess.run(
        [sys.executable, "-c", code], env={"PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "AGENTS:False" in out.stdout
