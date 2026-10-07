"""P4-03: portfolio vol targeting, leverage and exposure limits (synthetic fixtures only)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from firm.risk.limits import (
    LimitResult,
    RiskLimits,
    apply_caps,
    apply_limits,
    ewma_realised_vol,
    gross_cap_fraction,
    load_limits,
    risk_contributions,
    vol_scale,
)

SRC = Path(__file__).resolve().parents[1] / "src"
REPO = Path(__file__).resolve().parents[1]
TAU = 0.10  # arbitrary test value; no tau is chosen by the module or by config/risk.yaml


def make_limits(names, classes, share=None, **kw) -> RiskLimits:
    n = len(names)
    base = {
        "tau": TAU,
        "vol_ewma_span": 20,
        "instrument_type": "etf",
        "max_gross": 1.0,
        "long_only": True,
        "asset_class": dict(zip(names, classes, strict=True)),
        "handcraft_share": share if share is not None else {s: 1.0 / n for s in names},
    }
    base.update(kw)
    return RiskLimits(**base)


def diag_cov(names, vols) -> pd.DataFrame:
    return pd.DataFrame(np.diag(np.square(vols)), index=names, columns=names)


def five():
    names = list("ABCDE")
    return names, names, diag_cov(names, [0.1] * 5)  # one class per instrument, equal vols


def class_rc(w, cov, limits):
    rc = risk_contributions(w, cov)
    return rc.groupby(pd.Series(limits.asset_class)).sum()


# ---- vol scalar -------------------------------------------------------------------------------------------------
def test_forecast_proportionality_preserved():
    names, classes, cov = five()
    lim = make_limits(names, classes)
    raw = pd.Series([0.12, 0.10, 0.10, 0.08, 0.06], index=names)
    a = apply_limits(raw, cov, lim, sigma_ewma=TAU)
    b = apply_limits(raw / 2, cov, lim, sigma_ewma=TAU)
    assert (
        not a.gross_cap_bound
        and not any(a.class_cap_bound.values())
        and not any(a.instrument_cap_bound.values())
    )
    np.testing.assert_allclose(b.weights.to_numpy(), a.weights.to_numpy() / 2, atol=1e-12)
    np.testing.assert_allclose(
        a.weights.to_numpy(), raw.to_numpy(), atol=1e-12
    )  # no renormalisation to tau


def test_vol_scale_capped_at_1p5():
    names, classes, _ = five()
    lim = make_limits(names, classes)
    assert vol_scale(1e-6, lim) == 1.5
    assert vol_scale(TAU / 1.2, lim) == pytest.approx(1.2)


def test_vol_scale_converges_to_half():
    span = 20
    names, classes, _ = five()
    lim = make_limits(names, classes, vol_ewma_span=span)
    daily = TAU / np.sqrt(252)
    calm = np.tile([1.0, -1.0], 50) * daily * 0.5  # vol 0.5 tau
    hot = np.tile([1.0, -1.0], 3 * span) * daily * 2.0  # vol exactly 2 tau afterwards
    r = pd.Series(np.concatenate([calm, hot]))
    s_early = vol_scale(ewma_realised_vol(r.iloc[: len(calm)], span), lim)
    assert s_early == 1.5  # sigma = 0.5 tau, capped
    s_late = vol_scale(ewma_realised_vol(r, span), lim)
    assert s_late == pytest.approx(
        0.5, abs=0.025
    )  # 3 spans later the old regime is < 5% of the weight


def test_ewma_vol_needs_span_observations():
    with pytest.raises(ValueError):
        ewma_realised_vol(pd.Series([0.01] * 5), span=20)


def test_zero_vol_returns_zero_weights_and_logs(caplog):
    names, classes, cov = five()
    lim = make_limits(names, classes)
    with caplog.at_level(logging.WARNING, logger="firm.risk.limits"):
        assert vol_scale(0.0, lim) == 0.0
        res = apply_limits(pd.Series(0.1, index=names), cov, lim, sigma_ewma=0.0)
    assert (res.weights == 0).all() and res.scale == 0.0
    assert any("sigma_ewma" in m for m in caplog.messages)


# ---- gross cap / validation -------------------------------------------------------------------------------------
def test_gross_cap_binds_and_flags():
    names, classes, cov = five()
    lim = make_limits(names, classes)
    res = apply_caps(pd.Series(0.4, index=names), cov, lim)
    assert res.gross_cap_bound
    assert res.weights.abs().sum() == pytest.approx(1.0, abs=1e-12)
    assert "gross" in res.breaches_clipped


def test_max_gross_above_source_cap_raises():
    names, classes, _ = five()
    with pytest.raises(ValueError):
        make_limits(names, classes, max_gross=1.5)
    with pytest.raises(ValueError):
        make_limits(
            names,
            classes,
            instrument_type="futures",
            max_gross=5.0,
            margin_rate=dict.fromkeys(names, 0.1),
        )
    make_limits(
        names,
        classes,
        instrument_type="futures",
        max_gross=4.0,
        margin_rate=dict.fromkeys(names, 0.1),
    )


# ---- risk-contribution caps -------------------------------------------------------------------------------------
def test_risk_contributions_sum_to_one():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=list("ab"), columns=list("ab"))
    rc = risk_contributions(pd.Series([0.3, 0.5], index=list("ab")), cov)
    assert rc.sum() == pytest.approx(1.0)


def test_class_risk_cap_40pct():
    names = ["A1", "A2", "B1", "B2", "C1", "D1"]
    classes = ["A", "A", "B", "B", "C", "D"]
    w = pd.Series([0.175, 0.175, 0.1625, 0.1625, 0.1625, 0.1625], index=names)
    # class A: 35% notional; choose the other vols so that A's risk share is exactly 55%
    var_a = 2 * 0.175**2 * 0.30**2
    var_o = var_a * 0.45 / 0.55
    sig_o = np.sqrt(var_o / (4 * 0.1625**2))
    cov = diag_cov(names, [0.30, 0.30] + [sig_o] * 4)
    lim = make_limits(names, classes)
    before = class_rc(w, cov, lim)
    assert before["A"] == pytest.approx(0.55) and w[["A1", "A2"]].sum() == pytest.approx(0.35)
    res = apply_caps(w, cov, lim)
    after = class_rc(res.weights, cov, lim)
    assert (after <= 0.40 + 1e-9).all()
    assert after["A"] == pytest.approx(0.40, abs=1e-8)
    assert res.class_cap_bound["A"] and not res.class_cap_bound["B"]
    assert (res.weights >= 0).all() and (res.weights > 0).all()  # never drops instruments


def test_bond_class_not_clipped_on_notional():
    names = ["S1", "S2", "E", "G", "C"]
    classes = ["bond", "bond", "eq", "gold", "com"]
    w = pd.Series([0.3, 0.3, 0.4 / 3, 0.4 / 3, 0.4 / 3], index=names)
    var_bond = 2 * 0.3**2 * 0.05**2
    sig_o = np.sqrt(var_bond) / (0.4 / 3)  # each other class has the same variance as bonds
    cov = diag_cov(names, [0.05, 0.05, sig_o, sig_o, sig_o])
    lim = make_limits(names, classes)
    rc = class_rc(w, cov, lim)
    assert rc["bond"] == pytest.approx(0.25) and w[["S1", "S2"]].sum() == pytest.approx(0.60)
    res = apply_caps(w, cov, lim)
    np.testing.assert_allclose(res.weights.to_numpy(), w.to_numpy(), atol=1e-12)
    assert not res.class_cap_bound["bond"] and not res.gross_cap_bound


def test_instrument_risk_cap_2x_handcraft():
    names = list("ABCD")
    cov = diag_cov(names, [0.1] * 4)
    lim = make_limits(
        names, names, max_class_risk_share=1.0
    )  # isolate the instrument cap; shares 0.25 -> cap 0.5
    w = pd.Series([0.30, 0.05, 0.05, 0.05], index=names)  # A has ~0.9 of the risk
    assert risk_contributions(w, cov)["A"] > 0.5
    res = apply_caps(w, cov, lim)
    rc = risk_contributions(res.weights, cov)
    assert (rc <= 0.5 + 1e-9).all() and rc["A"] == pytest.approx(0.5, abs=1e-8)
    assert res.instrument_cap_bound["A"] and not res.instrument_cap_bound["B"]
    assert (res.weights > 0).all()


def test_missing_handcraft_share_raises():
    names, classes, cov = five()
    lim = make_limits(names, classes, share={s: 0.25 for s in names[:-1]})
    with pytest.raises(ValueError):
        apply_caps(pd.Series(0.1, index=names), cov, lim)


def test_infeasible_class_cap_is_flagged_not_raised(caplog):
    # all risk in one class: a risk SHARE cannot be reduced by scaling that class alone
    names = ["A1", "A2"]
    cov = diag_cov(names, [0.1, 0.1])
    lim = make_limits(names, ["A", "A"])
    with caplog.at_level(logging.WARNING, logger="firm.risk.limits"):
        res = apply_caps(pd.Series([0.2, 0.2], index=names), cov, lim)
    np.testing.assert_allclose(res.weights.to_numpy(), [0.2, 0.2])
    assert any("infeasible" in m for m in caplog.messages)


# ---- long-only, margin ------------------------------------------------------------------------------------------
def test_long_only_no_negative():
    names, classes, cov = five()
    lim = make_limits(names, classes)
    res = apply_caps(pd.Series([0.1, -0.1, 0.1, 0.1, 0.1], index=names), cov, lim)
    assert (res.weights >= 0).all() and res.weights["B"] == 0.0
    short_ok = make_limits(names, classes, long_only=False)
    assert (
        apply_caps(pd.Series([0.1, -0.1, 0.1, 0.1, 0.1], index=names), cov, short_ok).weights["B"]
        < 0
    )


def test_margin_noop_for_etf():
    names, classes, cov = five()
    lim = make_limits(names, classes)
    assert lim.margin_limit is None
    res = apply_caps(pd.Series(0.18, index=names), cov, lim)  # gross 0.9, within cap
    np.testing.assert_allclose(res.weights.to_numpy(), 0.18)
    assert "margin" not in res.breaches_clipped
    with pytest.raises(ValueError):
        make_limits(names, classes, margin_limit=0.3)


def test_margin_binds_for_futures_fixture():
    names, classes, cov = five()
    lim = make_limits(
        names,
        classes,
        instrument_type="futures",
        max_gross=4.0,
        margin_rate=dict.fromkeys(names, 0.10),
    )
    assert lim.margin_limit == 0.30
    res = apply_caps(
        pd.Series(0.7, index=names), cov, lim
    )  # gross 3.5 <= 4 but margin use 0.35 > 0.30
    assert not res.gross_cap_bound and "margin" in res.breaches_clipped
    assert (res.weights.abs() * 0.10).sum() == pytest.approx(0.30)
    with pytest.raises(ValueError):  # futures need margin rates
        make_limits(names, classes, instrument_type="futures", max_gross=4.0)


def test_apply_caps_idempotent():
    names = ["A1", "A2", "B1", "B2", "C1", "D1"]
    classes = ["A", "A", "B", "B", "C", "D"]
    w = pd.Series([0.45, 0.45, 0.3, 0.3, 0.3, 0.3], index=names)
    cov = diag_cov(names, [0.30, 0.30, 0.1, 0.1, 0.1, 0.1])
    lim = make_limits(names, classes)
    once = apply_caps(w, cov, lim)
    assert once.class_cap_bound["A"] and once.gross_cap_bound
    twice = apply_caps(once.weights, cov, lim)
    np.testing.assert_allclose(twice.weights.to_numpy(), once.weights.to_numpy(), atol=1e-9)
    assert not any(twice.class_cap_bound.values()) and not twice.gross_cap_bound


def test_apply_limits_scale_up_then_gross_cap():
    names, classes, cov = five()
    lim = make_limits(names, classes)
    res = apply_limits(
        pd.Series(0.12, index=names), cov, lim, sigma_ewma=TAU / 3
    )  # scale 1.5 -> gross 0.9
    assert res.scale == 1.5 and not res.gross_cap_bound
    assert res.weights.sum() == pytest.approx(0.9)
    res2 = apply_limits(
        pd.Series(0.16, index=names), cov, lim, sigma_ewma=TAU / 3
    )  # 1.2 gross -> capped
    assert res2.gross_cap_bound and res2.weights.sum() == pytest.approx(1.0)


def test_nan_weights_raise():
    names, classes, cov = five()
    with pytest.raises(ValueError):
        apply_caps(
            pd.Series([0.1, np.nan, 0.1, 0.1, 0.1], index=names), cov, make_limits(names, classes)
        )


def test_gross_cap_fraction():
    mk = lambda b: LimitResult(pd.Series(dtype=float), 1.0, b, {}, {}, [])
    flags = [True, False, False, True, False, False, False, True, False, False]
    assert gross_cap_fraction([mk(b) for b in flags]) == pytest.approx(3 / 10)
    with pytest.raises(ValueError):
        gross_cap_fraction([])


# ---- config loader ----------------------------------------------------------------------------------------------
def _write(tmp_path, **over):
    cfg = {
        "tau": 0.10,
        "instrument_type": "etf",
        "max_gross": 1.0,
        "max_vol_scale": 1.5,
        "vol_ewma_span": 30,
        "max_class_risk_share": 0.40,
        "max_instrument_risk_mult": 2.0,
        "long_only": True,
        "margin_limit": None,
        "margin_rate": None,
        "universe": str(REPO / "config" / "universe_etf.yaml"),
    }
    cfg.update(over)
    cfg = {k: v for k, v in cfg.items() if v != "__drop__"}
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_load_limits_roundtrip(tmp_path):
    uni = yaml.safe_load((REPO / "config" / "universe_etf.yaml").read_text())
    share = {s: 0.1 for s in ["SPY", "IEF"]}
    lim = load_limits(_write(tmp_path), handcraft_share=share)
    assert lim.tau == 0.10 and lim.vol_ewma_span == 30 and lim.instrument_type == "etf"
    assert lim.asset_class["SPY"] == "equity_us"
    assert set(lim.asset_class) == {s for v in uni["asset_classes"].values() for s in v}


@pytest.mark.parametrize(
    "over",
    [{"tau": None}, {"tau": "__drop__"}, {"vol_ewma_span": None}, {"vol_ewma_span": "__drop__"}],
)
def test_missing_tau_raises(tmp_path, over):
    with pytest.raises(ValueError):
        load_limits(_write(tmp_path, **over), handcraft_share={"SPY": 1.0})


def test_shipped_risk_yaml_chooses_no_tau_or_span():
    cfg = yaml.safe_load((REPO / "config" / "risk.yaml").read_text())
    assert cfg["tau"] is None and cfg["vol_ewma_span"] is None
    with pytest.raises(ValueError):
        load_limits(REPO / "config" / "risk.yaml", handcraft_share={"SPY": 1.0})


def test_load_limits_validates_gross(tmp_path):
    with pytest.raises(ValueError):
        load_limits(_write(tmp_path, max_gross=1.5), handcraft_share={"SPY": 1.0})


# ---- isolation --------------------------------------------------------------------------------------------------
def test_no_live_imports():
    code = (
        f"import sys; sys.path.insert(0, {str(SRC)!r}); import firm.risk.limits; "
        "bad=[m for m in sys.modules if m=='firm.live' or m.startswith('firm.live.') or m=='firm.agents' "
        "or m.startswith('firm.agents.') or m=='firm.portfolio.optimizer' or m=='firm.api' or m=='firm.runtime']; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr


# ---- batch 20: optional research-backtest options of apply_caps (defaults unchanged) ----------------------------------------------------------
def _random_case(rng, n=8, k_classes=4):
    names = [f"X{i}" for i in range(n)]
    classes = [f"c{i % k_classes}" for i in range(n)]
    a = rng.normal(size=(n, n + 10)) * 0.01
    vols = rng.uniform(0.5, 3.0, n)
    cov = pd.DataFrame(a @ a.T * np.outer(vols, vols) * 252, index=names, columns=names)
    w = pd.Series(rng.uniform(0.02, 0.3, n), index=names)
    return names, classes, cov, w


def test_closed_form_solver_matches_bisection_on_random_cases():
    rng = np.random.default_rng(11)
    n_bound = 0
    for _ in range(60):
        names, classes, cov, w = _random_case(rng)
        lim = make_limits(names, classes, max_instrument_risk_mult=2.0, max_class_risk_share=0.6)
        try:
            ref = apply_caps(w, cov, lim)
        except RuntimeError:
            continue                                   # slow-fixed-point case: covered below
        fast = apply_caps(w, cov, lim, solver="closed_form")
        np.testing.assert_allclose(fast.weights.to_numpy(), ref.weights.to_numpy(), rtol=1e-8, atol=1e-10)
        assert fast.gross_cap_bound == ref.gross_cap_bound and fast.instrument_cap_bound == ref.instrument_cap_bound
        n_bound += any(ref.instrument_cap_bound.values()) or any(ref.class_cap_bound.values())
    assert n_bound >= 10                               # the comparison exercised binding caps


def test_closed_form_scale_puts_the_share_exactly_on_the_cap():
    rng = np.random.default_rng(5)
    names, classes, cov, w = _random_case(rng, n=6, k_classes=6)
    lim = make_limits(names, classes, max_instrument_risk_mult=1.0, max_class_risk_share=1.0)
    res = apply_caps(w, cov, lim, solver="closed_form", tol=1e-12, max_sweeps=500, on_nonconvergence="return")
    rc = risk_contributions(res.weights, cov)
    assert (rc <= 1.0 / 6.0 + 1e-6).all()


def test_nonconvergence_raises_by_default_and_returns_the_last_iterate_on_request():
    rng = np.random.default_rng(0)
    names = [f"X{i}" for i in range(14)]
    classes = [f"c{i}" for i in range(14)]
    for _ in range(300):
        a = rng.normal(size=(14, 34)) * 0.01
        vols = rng.uniform(0.5, 3, 14)
        cov = pd.DataFrame(a @ a.T * np.outer(vols, vols) * 252, index=names, columns=names)
        w = pd.Series(rng.uniform(0.01, 0.25, 14), index=names)
        lim = make_limits(names, classes, share={s: 1.0 / 14 for s in names}, max_instrument_risk_mult=1.2, max_class_risk_share=1.0)
        try:
            apply_caps(w, cov, lim)
        except RuntimeError:
            res = apply_caps(w, cov, lim, solver="closed_form", max_sweeps=3, on_nonconvergence="return")
            assert "nonconverged" in res.breaches_clipped and (res.weights <= w + 1e-12).all()      # only ever scaled down
            with pytest.raises(RuntimeError, match="did not converge in 3 sweeps"):
                apply_caps(w, cov, lim, solver="closed_form", max_sweeps=3)
            return
    pytest.fail("no non-converging case found; tighten the instrument cap multiple")


def test_apply_caps_option_validation():
    names, classes, cov = five()
    lim = make_limits(names, classes)
    w = pd.Series(0.1, index=names)
    with pytest.raises(ValueError, match="solver"):
        apply_caps(w, cov, lim, solver="newton")
    with pytest.raises(ValueError, match="on_nonconvergence"):
        apply_caps(w, cov, lim, on_nonconvergence="ignore")
