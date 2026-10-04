"""P3-05 forecast combination and FDM tests (synthetic data only)."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.portfolio.forecast_combine import (
    FDM_CAP,
    FORECAST_CAP,
    combine_forecasts,
    fdm,
    group_equal_weights,
    pooled_forecast_correlation,
    validate_weights,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def test_constants():
    assert FDM_CAP == 2.5 and FORECAST_CAP == 20.0


def test_fdm_hand_computed():
    w = np.array([0.5, 0.5])
    rho = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert abs(fdm(w, rho) - 1 / np.sqrt(0.75)) < 1e-12
    assert abs(fdm(w, rho) - 1.1547005383792515) < 1e-12


def test_fdm_uncorrelated_two_rules():
    assert abs(fdm(np.array([0.5, 0.5]), np.eye(2)) - np.sqrt(2)) < 1e-12


def test_fdm_perfect_correlation_is_one():
    assert abs(fdm(np.array([0.3, 0.7]), np.ones((2, 2))) - 1.0) < 1e-12


def test_fdm_cap():
    n = 20
    assert fdm(np.full(n, 1 / n), np.eye(n)) == FDM_CAP
    assert fdm(np.full(n, 1 / n), np.eye(n), cap=10.0) == pytest.approx(np.sqrt(20))


def test_negative_correlation_floored():
    w = np.array([0.5, 0.5])
    neg = np.array([[1.0, -0.5], [-0.5, 1.0]])
    assert fdm(w, neg) == fdm(w, np.eye(2))
    # without the floor the quadratic form shrinks to 0.5 and the FDM grows
    assert fdm(w, neg, floor_rho_at_zero=False) == pytest.approx(1 / np.sqrt(0.25 + 0.25 - 0.25))


def test_fdm_drops_zero_weight_rules_and_checks_shape():
    w = np.array([0.5, 0.5, 0.0])
    rho = np.array([[1, 0.5, 0.9], [0.5, 1, 0.9], [0.9, 0.9, 1]])
    assert abs(fdm(w, rho) - 1 / np.sqrt(0.75)) < 1e-12
    with pytest.raises(ValueError):
        fdm(np.array([0.5, 0.5]), np.eye(3))


def test_avg_abs_combined_about_10():
    rng = np.random.default_rng(0)
    k, n = 6, 50000
    rho = np.full((k, k), 0.6) + 0.4 * np.eye(k)
    z = rng.multivariate_normal(np.zeros(k), rho, size=n)
    z *= 10.0 / np.abs(z).mean(axis=0)  # every rule has mean |f| = 10
    df = pd.DataFrame(z, columns=[f"r{i}" for i in range(k)])
    weights = {c: 1 / k for c in df.columns}
    emp = df.corr().to_numpy()
    f = fdm(np.array(list(weights.values())), emp)
    out = combine_forecasts(df, weights, f)
    assert abs(out.abs().mean() - 10.0) < 1.0
    assert out.abs().max() <= FORECAST_CAP


def test_combine_then_floor_not_floor_then_combine():
    df = pd.DataFrame({"a": [-10.0, 10.0], "b": [10.0, -10.0]})
    w = {"a": 0.5, "b": 0.5}
    out = combine_forecasts(df, w, 1.0, floor=0.0)
    assert (out == 0.0).all()
    pos = pd.DataFrame({"a": [10.0], "b": [10.0]})
    assert combine_forecasts(pos, w, 1.0, floor=0.0).iloc[0] == 10.0
    assert combine_forecasts(pd.DataFrame({"a": [-30.0], "b": [-30.0]}), w, 1.0).iloc[0] == -FORECAST_CAP


def test_group_equal_weights():
    gw = {"ewmac": 0.6, "breakout": 0.4}
    surv = {"ewmac": [f"e{i}" for i in range(5)], "breakout": ["b20", "b40"]}
    out = group_equal_weights(gw, surv)
    assert all(out[f"e{i}"] == pytest.approx(0.12) for i in range(5))
    assert out["b20"] == out["b40"] == pytest.approx(0.2)
    assert sum(out.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        group_equal_weights(gw, {"ewmac": ["e0"], "breakout": []})
    with pytest.raises(ValueError):
        group_equal_weights(gw, {"ewmac": ["e0"]})
    with pytest.raises(ValueError):
        group_equal_weights(gw, {"ewmac": ["x"], "breakout": ["x"]})


def test_weights_validation():
    validate_weights({"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError):
        validate_weights({"a": 1.5, "b": -0.5})
    with pytest.raises(ValueError):
        validate_weights({"a": 0.5, "b": 0.4})
    with pytest.raises(ValueError):
        validate_weights({})
    with pytest.raises(ValueError):
        combine_forecasts(pd.DataFrame({"a": [1.0]}), {"a": 0.5}, 1.0)
    with pytest.raises(ValueError):
        combine_forecasts(pd.DataFrame({"a": [1.0]}), {"a": 0.5, "b": 0.5}, 1.0)  # b missing


def test_nan_propagates():
    df = pd.DataFrame({"a": [5.0, np.nan, 5.0], "b": [5.0, 5.0, 5.0], "z": [np.nan] * 3})
    out = combine_forecasts(df, {"a": 0.5, "b": 0.5, "z": 0.0}, 1.0)
    assert out.iloc[0] == 5.0 and np.isnan(out.iloc[1]) and out.iloc[2] == 5.0


def _fc(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-01", periods=n)
    base = rng.normal(size=n)
    return pd.DataFrame(
        {c: base * k + rng.normal(size=n) for c, k in (("a", 1.0), ("b", 0.5), ("c", 0.0))}, index=idx
    )


def test_pooled_correlation_uses_window_only():
    d = {"X": _fc(1, 1000), "Y": _fc(2, 1000)}
    start, end = d["X"].index[0], d["X"].index[599]
    c0 = pooled_forecast_correlation(d, start, end)
    longer = {k: pd.concat([v, _fc(9, 300).set_axis(pd.bdate_range(v.index[-1] + pd.Timedelta(days=1), periods=300))]) * 1.0
              for k, v in d.items()}
    pd.testing.assert_frame_equal(pooled_forecast_correlation(longer, start, end), c0)
    manual = (d["X"].loc[start:end].corr() + d["Y"].loc[start:end].corr()) / 2
    pd.testing.assert_frame_equal(c0, manual)
    assert list(c0.index) == list(c0.columns) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        pooled_forecast_correlation(d, start, d["X"].index[199])  # 200 < 250 overlap


def test_pooled_correlation_pairwise_overlap():
    x = _fc(3, 600)
    x.loc[x.index[:400], "c"] = np.nan  # pair overlap with c is only 200
    with pytest.raises(ValueError):
        pooled_forecast_correlation({"X": x}, x.index[0], x.index[-1])


def test_no_weight_fitting_api():
    for fn in (combine_forecasts, fdm, validate_weights, group_equal_weights, pooled_forecast_correlation):
        names = " ".join(inspect.signature(fn).parameters)
        assert "return" not in names and "pnl" not in names and "sharpe" not in names


def test_import_light():
    code = (
        "import sys, firm.portfolio.forecast_combine;"
        "bad=[m for m in sys.modules if m.split('.')[:2] in "
        "(['firm','live'],['firm','api'],['firm','runtime'])];"
        "print('BAD:'+','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env={"PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "BAD:" in out.stdout.splitlines(), out.stdout
    init = (SRC / "firm" / "portfolio" / "__init__.py").read_text()
    assert "forecast_combine" not in init
